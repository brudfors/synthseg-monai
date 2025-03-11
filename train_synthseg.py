"""SynthSeg training in MONAI
"""
import math
import os
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import matplotlib.pyplot as plt
import shutil

from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureTyped,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityd,
    RandFlipd,
)
from monai.data import CacheDataset, DataLoader, decollate_batch, partition_dataset
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

import transforms_synthseg as transforms
import utils_synthseg as utils

seed = 0
set_determinism(seed=seed)
torch.backends.cudnn.benchmark = True


# ==================================
# Parameters
# ==================================
testing = True
if testing:
    validation_steps = 100
    spatial_size = (96,) * 3
    patch_size = None
    dout = "./results-testing"
    if os.path.isdir(dout):  shutil.rmtree(dout)
else:
    validation_steps = 1000
    spatial_size = (256,) * 3
    patch_size = (160,) * 3
    dout = "./results"
if not os.path.isdir(dout):  os.makedirs(dout, exist_ok=True)
restart_meta_pth = os.path.join(dout, 'restart_meta_data.pkl')
dir_labels = "./data/training_label_maps"
model_pth = os.path.join(dout, 'model_latest.pth')
model_pth = None if not os.path.exists(model_pth) else model_pth
batch_size = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
steps = 1000000

# ==================================
# Distributed Data Parallel (DDP)
# ==================================
if "LOCAL_RANK" in os.environ:
    print("Setting up DDP...", end="")
    ddp = True
    local_rank = int(os.environ["LOCAL_RANK"])
    # initialize the distributed training process, every GPU runs in a process
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
    num_gpus = dist.get_world_size()    
    print("done!")
else:
    ddp = False
    num_gpus = 1
torch.cuda.set_device(device)


# ==================================
# Get labels
# Train/val split happens here
# ==================================
labels = [str(d) for d in sorted(Path(dir_labels).rglob('*.nii.gz'))]
data = [{"label": label_name} for label_name in labels]
train_files, val_files = data[:14], data[14:]
num_train = len(train_files)
num_val = len(val_files)
steps_per_epoch = num_train
validation_epoch = round(validation_steps/steps_per_epoch)
if ddp:
    # DDP: Partition per device
    train_files = partition_dataset(
        data=train_files,
        num_partitions=num_gpus,
        shuffle=False,
        seed=0,
        drop_last=False,
        even_divisible=True,
    )[dist.get_rank()]
# Label info
target_labels = list(transforms.MapLabelsSynthSeg.label_mapping().values())
n_labels = len(target_labels)
    

# ==================================
# Get transforms
# ==================================
train_transforms = Compose(
    [
        LoadImaged(keys="label"),
        EnsureChannelFirstd(keys="label"),
        Orientationd(keys="label", axcodes="RAS"),            
        transforms.MapLabelsSynthSeg(),
        transforms.Resize(spatial_size, testing),
        EnsureTyped(keys="label", dtype=torch.int16, device=device),            
        transforms.SynthSegd(params=utils.get_synth_params(target_labels, train=True), patch_size=patch_size),
        ScaleIntensityd(keys="image"),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),                          
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys="label"),
        EnsureChannelFirstd(keys="label"),
        Orientationd(keys="label", axcodes="RAS"),
        transforms.MapLabelsSynthSeg(),
        transforms.Resize(spatial_size, testing),
        EnsureTyped(keys="label", dtype=torch.int16, device=device),            
        transforms.SynthSegd(params=utils.get_synth_params(target_labels, train=False)),
        ScaleIntensityd(keys="image"),                    
    ]
)


# ==================================
# Get data loaders
# ==================================
train_loader = DataLoader(
    CacheDataset(
        data=train_files, 
        transform=train_transforms, 
        cache_rate=1.0,
    ), 
    batch_size=batch_size, 
    shuffle=True,
)
val_loader = DataLoader(
    CacheDataset(
        data=val_files, 
        transform=val_transforms, 
        cache_rate=1.0,
    ), 
    batch_size=1, 
    shuffle=False,
)
max_epochs = steps // num_train


# ==================================
# Get model
# ==================================
out_channels = n_labels + 1
model = utils.get_model(out_channels)
model = model.to(device)
if model_pth != None:
    checkpoint = torch.load(model_pth)
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)
if ddp:
    model = DistributedDataParallel(
        model,
        device_ids=[device],
        find_unused_parameters=False,
    )

    
# ==================================
# Get loss, optimiser and metrics
# ==================================
loss_function = DiceLoss(
    to_onehot_y=True, softmax=True, include_background=True, 
    smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=True,
)
optimizer = torch.optim.Adam(model.parameters(), lr=math.sqrt(batch_size * num_gpus) * 1e-4)
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")


# ==================================
# Various
# ==================================
post_pred = Compose([Activations(softmax=True), AsDiscrete(threshold=0.5)])
fig, axs = plt.subplots(4, 3, figsize=(9, 12))
cmap, norm = utils.get_label_cmap(n_labels=out_channels)
# Load/init meta data dict
if os.path.isfile(restart_meta_pth) and model_pth != None:
    with open(restart_meta_pth, 'rb') as f:
        meta_data = pickle.load(f)
    if "epoch" not in meta_data:  meta_data["epoch"] = 0
    if "best_metric" not in meta_data:  meta_data["best_metric"] = -1
    if "loss_values" not in meta_data:  meta_data["loss_values"] = []
    if "metric_values" not in meta_data:  meta_data["metric_values"] = []
else:
    meta_data = dict(
        epoch=0,
        best_metric=-1,
        loss_values=[],
        metric_values=[],
    )
best_metric = meta_data["best_metric"]
loss_values = meta_data["loss_values"]
metric_values = meta_data["metric_values"]

    
# ==================================
# Run training
# ==================================
print("-" * 64)
print("Starting training from epoch {}".format(meta_data["epoch"]))
print("-" * 64)

for epoch in range(meta_data["epoch"], max_epochs):   
        
    # Save latest meta data
    meta_data["epoch"] = epoch
    meta_data["best_metric"] = best_metric
    meta_data["loss_values"] = loss_values
    meta_data["metric_values"] = metric_values
    with open(restart_meta_pth, 'wb') as f:
        pickle.dump(meta_data, f)
    
    # ----------------------------------
    # Train  
    # ----------------------------------
    model.train()
    epoch_loss = 0
    save_fig_ix = np.random.randint(0, len(train_loader))  # For printing a random debug figure

    step_epoch = 0
    for ix, batch_data in enumerate(train_loader):       

        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        step_epoch += 1
        batch_ix = np.random.randint(0, inputs.shape[0])
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)                   
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if (epoch % validation_epoch == 0 or epoch == max_epochs - 1) and (save_fig_ix == ix):
            with torch.no_grad():
                post_outputs = [post_pred(i) for i in decollate_batch(outputs)]
            axs[0, 0].imshow(utils.extract_slice(inputs[batch_ix][0]).cpu().numpy(), cmap="gray")
            axs[0, 0].set_title('train image'); axs[0, 0].axis('off')
            axs[0, 1].imshow(utils.extract_slice(labels[batch_ix][0]).cpu().numpy(), cmap=cmap, norm=norm, interpolation="nearest")
            axs[0, 1].set_title('train label'); axs[0, 1].axis('off')
            axs[0, 2].imshow(utils.extract_slice(post_outputs[batch_ix].argmax(dim=0)).cpu().numpy(), cmap=cmap, norm=norm, interpolation="nearest")
            axs[0, 2].set_title('train prediction'); axs[0, 2].axis('off')
       
    epoch_loss /= step_epoch
    print(f"EPOCH={epoch + 1:{' '}{len(str(max_epochs))}}/{max_epochs} |__LOSS (N={num_train})={epoch_loss:.4f}  |  {utils.get_timestamp()}" )
    loss_values.append(epoch_loss)
        
    if epoch % validation_epoch == 0 or epoch == max_epochs - 1:
        # ----------------------------------
        # Val
        # ----------------------------------
        model.eval()
        save_fig_ix = np.random.randint(0, len(val_loader))
                
        with torch.no_grad():
            
            with torch.random.fork_rng(enabled=seed):
                torch.random.manual_seed(seed)
                
                for ix, batch_data in enumerate(val_loader):

                    inputs, labels = (
                        batch_data["image"].to(device),
                        batch_data["label"].to(device),
                    )
                    batch_ix = np.random.randint(0, inputs.shape[0])

                    outputs = utils.inference(inputs, model, patch_size=patch_size)
                    post_outputs = [post_pred(i) for i in decollate_batch(outputs)]

                    dice_metric(y_pred=post_outputs, y=labels)
                    dice_metric_batch(y_pred=post_outputs, y=labels)

                    if ix == 0:
                        inputs_slice = utils.extract_slice(inputs[batch_ix][0]).cpu().numpy()
                        axs[1, 0].imshow(inputs_slice, cmap="gray")
                        axs[1, 0].set_title('val image now'); axs[1, 0].axis('off')
                        labels_slice = utils.extract_slice(labels[batch_ix][0]).cpu().numpy()
                        axs[1, 1].imshow(labels_slice, cmap=cmap, norm=norm, interpolation="nearest")
                        axs[1, 1].set_title('val label now'); axs[1, 1].axis('off')
                        pred_slice = utils.extract_slice(post_outputs[batch_ix].argmax(dim=0)).cpu().numpy()
                        axs[1, 2].imshow(pred_slice, cmap=cmap, norm=norm, interpolation="nearest")
                        axs[1, 2].set_title('val prediction now'); axs[1, 2].axis('off')

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            metric_batch = dice_metric_batch.aggregate()        
            metric_values.append(metric)
            # reset the status for next validation round
            dice_metric.reset()
            dice_metric_batch.reset()                        
            # print metric to terminal
            print(f"EPOCH={epoch + 1:{' '}{len(str(max_epochs))}}/{max_epochs} |____METRIC (N={num_val})={metric:.4f}")
            for i in range(0, len(metric_batch), 10): 
                print( " " * (13 + len(str(max_epochs))) + "|______" + ", "
                      .join([f"{k + i:3.0f}={v:0.3f}".format(k, v) for k, v in enumerate(metric_batch[i:i + 10])]))

            if metric > best_metric:
                best_metric = metric                
                torch.save(model.state_dict(), os.path.join(dout, "model_best.pth"))   
                # show best model's prediction
                axs[2, 0].imshow(inputs_slice, cmap="gray")
                axs[2, 0].set_title('val image best'); axs[1, 0].axis('off')
                axs[2, 1].imshow(labels_slice, cmap=cmap, norm=norm, interpolation="nearest")
                axs[2, 1].set_title('val label best'); axs[1, 1].axis('off')
                axs[2, 2].imshow(pred_slice, cmap=cmap, norm=norm, interpolation="nearest")
                axs[2, 2].set_title('val prediction best'); axs[1, 2].axis('off')

            utils.plot_loss_and_metric(axs, loss_values, metric_values, validation_epoch)
        
            plt.suptitle(f"EPOCH={epoch}, LOSS={epoch_loss:.4f}, METRIC_NOW={metric:.4f}, METRIC_BEST={best_metric:.4f}")
            fig.tight_layout()
            plt.savefig(os.path.join(dout, "outputs.png"))
  
    torch.save(model.state_dict(), os.path.join(dout, "model_latest.pth"))        
