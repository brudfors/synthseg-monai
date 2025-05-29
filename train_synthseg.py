"""Code for SynthSeg training in MONAI.
"""
import math
import os
from pathlib import Path
import nvidia_smi
import numpy as np
import torch
import time
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import matplotlib.pyplot as plt
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureTyped,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    RandFlipd,
)
from monai.data import (
    CacheDataset,
    DataLoader,
    decollate_batch,
    partition_dataset,
)
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

import transforms_synthseg as transforms
import utils_synthseg as utils

seed = 0
set_determinism(seed=seed)
torch.backends.cudnn.benchmark = True
nvidia_smi.nvmlInit()
gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)


# Settings
dir_input = "./training_label_maps" # Folder path of training labels
dir_results = "./results" # Folder path where to write results (checkpoint, best model, figures)
batch_size = 1
total_steps = 400000 # total_steps = epochs x len(train_files) ### 500 000 used in SynthSeg body paper ###
validation_steps = 2000 # steps between each validation
spatial_size = None # Resize input to this size, eg (256,) * 3, use None for no resize
patch_size = None # Training patch size, eg (128,) * 3, use None for training on full input
pth_checkpoint = os.path.join(dir_results, 'checkpoint.pkl')
pth_checkpoint_prev = os.path.join(dir_results, 'checkpoint_prev.pkl')
if not os.path.isdir(dir_results):  os.makedirs(dir_results, exist_ok=True)  
device = torch.device("cuda:0")

# Distributed Data Parallel (DDP)
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

# Get labels
labels = [str(d) for d in sorted(Path(dir_input).rglob('*.nii.gz'))]
data = [{"label": label_name} for label_name in labels]
train_files, val_files = data, data # train/val split is the same, but for validation minimal random transformations are applied (and random seed is fixed)
# train_files, val_files = data[:1], data[:1] #
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
        seed=seed,
        drop_last=False,
        even_divisible=True,
    )[dist.get_rank()]
# Label info
target_labels = list(transforms.MapLabelsSynthSeg.label_mapping().values())
n_labels = len(target_labels)
    
# Get transforms
train_transforms = Compose(
    [
        LoadImaged(keys="label"),
        EnsureChannelFirstd(keys="label"),
        Orientationd(keys="label", axcodes="RAS"),            
        transforms.MapLabelsSynthSeg(),
        transforms.ResizeTransform(keys=["label"], spatial_size=spatial_size, method="pad_crop"),        
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
        transforms.ResizeTransform(keys=["label"], spatial_size=spatial_size, method="pad_crop"),        
        EnsureTyped(keys="label", dtype=torch.int16, device=device),            
        transforms.SynthSegd(params=utils.get_synth_params(target_labels, train=False)),
        ScaleIntensityd(keys="image"),                    
    ]
)

# Get data loaders
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
epochs = total_steps // num_train

# Get model
out_channels = n_labels + 1
model = utils.get_model(out_channels)
model = model.to(device)
   
# Get loss, optimiser and metrics
loss_function = DiceLoss(
    to_onehot_y=True, softmax=True, include_background=True, 
    smooth_nr=1e-5, smooth_dr=1e-5, squared_pred=True,
)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

optimizer = torch.optim.Adam(model.parameters(), lr=math.sqrt(batch_size * num_gpus) * 1e-4)

# Load checkpoint
if os.path.isfile(pth_checkpoint):   
    # Is checkpoint
    print(f"Loading checkpoint from {pth_checkpoint}")
    checkpoint = torch.load(pth_checkpoint)
    checkpoint["model_state_dict"] = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if "epoch" not in checkpoint:  checkpoint["epoch"] = 0
    if "best_metric" not in checkpoint:  checkpoint["best_metric"] = -1
    if "loss_values" not in checkpoint:  checkpoint["loss_values"] = []
    if "metric_values" not in checkpoint:  checkpoint["metric_values"] = []
    if "train_duration" not in checkpoint:  checkpoint["train_duration"] = 0
else:
    # No checkpoint
    checkpoint = dict(
        epoch=0,
        best_metric=-1,
        loss_values=[],
        metric_values=[],
        model_state_dict=None,
        optimizer_state_dict=None,
        train_duration=0,
    )    
epoch = checkpoint["epoch"]
best_metric = checkpoint["best_metric"]
loss_values = checkpoint["loss_values"]
metric_values = checkpoint["metric_values"]
train_duration = checkpoint["train_duration"]

# Various
if ddp:
    model = DistributedDataParallel(
        model,
        device_ids=[device],
        find_unused_parameters=False,
    )
post_pred = Compose([Activations(softmax=True), AsDiscrete(threshold=0.5)])
fig, axs = plt.subplots(4, 3, figsize=(9, 12))
cmap, norm = utils.get_label_cmap(n_labels=out_channels)
scaler = torch.GradScaler()
    
# Run training
print("-" * 64)
print(f"Starting training from epoch {epoch}")
print("-" * 64)

for epoch in range(epoch, epochs):         
      
    # Train  
    start_time = time.time()
    model.train()
    epoch_loss = 0
    save_fig_ix = np.random.randint(0, len(train_loader))  # For printing a random debug figure

    if epoch == 0 or math.isnan(epoch_loss):
        print("WARNING: Loss is NaN do not save previous checkpoint")
    else:
        # Save previous checkpoint (good to have two copies of the checkpoint
        # in case there are issues with available disk space)
        checkpoint["epoch"] = epoch
        checkpoint["best_metric"] = best_metric
        checkpoint["loss_values"] = loss_values
        checkpoint["metric_values"] = metric_values
        checkpoint["train_duration"] = train_duration
        checkpoint["model_state_dict"] = model.state_dict()
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        try:
            torch.save(checkpoint, pth_checkpoint_prev)
        except Exception as e:
            print(f"Error saving previous model: {e}")
                
    step_epoch = 0
    for ix, batch_data in enumerate(train_loader):       
        
        image, label = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        step_epoch += 1
        batch_ix = np.random.randint(0, image.shape[0])

        optimizer.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(image)
            loss = loss_function(pred, label)                   
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()

        if (epoch % validation_epoch == 0 or epoch == epochs - 1) and (save_fig_ix == ix):
            with torch.no_grad():
                pred = [post_pred(i) for i in decollate_batch(pred)]
            axs[0, 0].imshow(utils.extract_slice(image[batch_ix][0]).cpu().numpy(), cmap="gray")
            axs[0, 0].set_title('train image'); axs[0, 0].axis('off')
            axs[0, 1].imshow(utils.extract_slice(label[batch_ix][0]).cpu().numpy(), cmap=cmap, norm=norm, interpolation="nearest")
            axs[0, 1].set_title('train label'); axs[0, 1].axis('off')
            axs[0, 2].imshow(utils.extract_slice(pred[batch_ix].argmax(0)).cpu().numpy(), cmap=cmap, norm=norm, interpolation="nearest")
            axs[0, 2].set_title('train pred'); axs[0, 2].axis('off')
       
    epoch_loss /= step_epoch
    print(f"EPOCH={epoch + 1:{' '}{len(str(epochs))}}/{epochs} |__LOSS (N_t={num_train}, N_b={batch_size})={epoch_loss:.4f} | {utils.get_timestamp()}", end="")
    gpu_info = nvidia_smi.nvmlDeviceGetMemoryInfo(gpu_handle)
    print(f" | VRAM: {gpu_info.used/1000000:.0f} / {gpu_info.total/1000000:.0f}MB", end="")
    print(f" | train_duration: {train_duration/3600:0.1f} h")
    loss_values.append(epoch_loss)
        
    if epoch % validation_epoch == 0 or epoch == epochs - 1:
        # Validate
        model.eval()
        save_fig_ix = np.random.randint(0, len(val_loader))
                
        with torch.no_grad():
            with torch.random.fork_rng(enabled=seed):
                torch.random.manual_seed(seed)
                for ix, batch_data in enumerate(val_loader):

                    image, label = (
                        batch_data["image"].to(device),
                        batch_data["label"].to(device),
                    )
                    batch_ix = np.random.randint(0, image.shape[0])

                    pred = utils.inference(image, model, patch_size=patch_size)
                    pred = [post_pred(i) for i in decollate_batch(pred)]

                    dice_metric(y_pred=pred, y=label)
                    dice_metric_batch(y_pred=pred, y=label)

                    if ix == 0:
                        inputs_slice = utils.extract_slice(image[batch_ix][0]).cpu().numpy()
                        axs[1, 0].imshow(inputs_slice, cmap="gray")
                        axs[1, 0].set_title('val image now'); axs[1, 0].axis('off')
                        labels_slice = utils.extract_slice(label[batch_ix][0]).cpu().numpy()
                        axs[1, 1].imshow(labels_slice, cmap=cmap, norm=norm, interpolation="nearest")
                        axs[1, 1].set_title('val label now'); axs[1, 1].axis('off')
                        pred_slice = utils.extract_slice(pred[batch_ix].argmax(dim=0)).cpu().numpy()
                        axs[1, 2].imshow(pred_slice, cmap=cmap, norm=norm, interpolation="nearest")
                        axs[1, 2].set_title('val pred now'); axs[1, 2].axis('off')

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            metric_batch = dice_metric_batch.aggregate()        
            metric_values.append(metric)
            # reset the status for next validation round
            dice_metric.reset()
            dice_metric_batch.reset()                        
            # print metric to terminal
            print(f"EPOCH={epoch + 1:{' '}{len(str(epochs))}}/{epochs} |__METRIC ((N_v={num_val}, N_b={batch_size}))={metric:.4f}")            
            for i in range(0, len(metric_batch), 10): 
                print(" " * (13 + len(str(epochs))) + "|____" + ","
                      .join([f"{k + i:3.0f}={v:0.3f}".format(k, v) for k, v in enumerate(metric_batch[i:i + 10])]))                        
            
            if metric > best_metric:
                best_metric = metric                
                torch.save(model.state_dict(), os.path.join(dir_results, "model_best.pth"))   
                # show best model's prediction
                axs[2, 0].imshow(inputs_slice, cmap="gray")
                axs[2, 0].set_title('val image best'); axs[1, 0].axis('off')
                axs[2, 1].imshow(labels_slice, cmap=cmap, norm=norm, interpolation="nearest")
                axs[2, 1].set_title('val label best'); axs[1, 1].axis('off')
                axs[2, 2].imshow(pred_slice, cmap=cmap, norm=norm, interpolation="nearest")
                axs[2, 2].set_title('val pred best'); axs[1, 2].axis('off')

            utils.plot_loss_and_metric(axs, loss_values, metric_values, validation_epoch)
        
            plt.suptitle(f"EPOCH={epoch}, LOSS={epoch_loss:.4f}, METRIC_NOW={metric:.4f}, METRIC_BEST={best_metric:.4f}")
            fig.tight_layout()
            plt.savefig(os.path.join(dir_results, "snapshot.png"))

    epoch_time = time.time() - start_time
    train_duration += epoch_time

    if math.isnan(epoch_loss):
        print("WARNING: Loss is NaN do not save checkpoint")
    else:
        # Save checkpoint
        checkpoint["epoch"] = epoch
        checkpoint["best_metric"] = best_metric
        checkpoint["loss_values"] = loss_values
        checkpoint["metric_values"] = metric_values
        checkpoint["train_duration"] = train_duration
        checkpoint["model_state_dict"] = model.state_dict()
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        try:
            torch.save(checkpoint, pth_checkpoint)
        except Exception as e:
            print(f"Error saving model: {e}")

        
