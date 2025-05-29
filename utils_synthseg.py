"""Code for utility functions for Synthseg training in MONAI.
"""
from datetime import datetime
from matplotlib.colors import (
    BoundaryNorm,
    ListedColormap,
)
import matplotlib.pyplot as plt
import numpy as np
import torch

from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet


def extract_slice(dat, dim=2, mid_ix=None):
    """Extract slice from volume, along some dimension.
    """
    if len(dat.shape) == 2:  return dat
    dat = dat.as_tensor()
    mid = (0.5*torch.as_tensor(dat.shape[-3:])).round().type(torch.int)
    if mid_ix is not None: mid = mid_ix   
    if dim == 0: dat = dat[..., mid[0], :, :]
    if dim == 1: dat = dat[..., :, mid[1], :]
    if dim == 2: dat = dat[..., :, :, mid[2]]
    return dat

def get_label_cmap(n_labels):
    """Get matplotlib colour map for a label map.
    """
    unique_values = np.arange(n_labels)
    colors = plt.cm.turbo(np.linspace(0, 1, len(unique_values)))
    cmap = ListedColormap(colors)
    # Create bin edges by adding half a step to unique values
    bin_edges = np.concatenate([unique_values - 0.5, [unique_values[-1] + 0.5]])
    # Create a BoundaryNorm to map unique values to indices in the colormap
    norm = BoundaryNorm(bin_edges, cmap.N, clip=True)
    
    return cmap, norm

def get_model(out_channels):
    """Get SynthSeg DL model.
    """
    model = BasicUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=out_channels,
        features=(24, 48, 96, 192, 384, 24),
        act=("LeakyReLU", {"negative_slope": 0.2, "inplace": True}),
        upsample="deconv",
    )
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {model_size:,}")
    return model

def get_synth_params(target_labels, train=True):
    """Get SynthSeg parameters for train or val.       
    """
    if train:
        # Training
        synth_params = {   
            "target_labels": target_labels, 
            "elastic_steps": 8,    
            "rotation": 15,
            "shears": 0.012,
            "zooms": 0.15,
            "elastic": 0.075,
            "elastic_nodes": 10,
            "gmm_fwhm": 10,
            "bias": 7,
            "gamma": 0.6,
            "motion_fwhm": 3,
            "resolution": 8,
            "snr": 10,
            "gfactor": 5,
            "bound": "zeros",
            "translations": 0.1,
        }        
    else:
        # Validation
        synth_params = {    
            "target_labels": target_labels, 
            "elastic_steps": 8,    
            "translations": 0,
            "rotation": 0,
            "shears": 0.0,
            "zooms": 0.0,
            "elastic": 0.0,
            "elastic_nodes": 10,
            "gmm_fwhm": 10,
            "bias": 2,
            "gamma": 0.1,
            "motion_fwhm": 1,
            "resolution": 1,
            "snr": 100,
            "gfactor": 2,
            "bound": "zeros",
            "translations": 0,
        }            
    return synth_params

def get_timestamp():
    """Get a time stamp.
    """
    now = datetime.now()
    return now.strftime("%H:%M:%S %d/%m/%Y")

def inference(inputs, model, patch_size=None):
    """Run pytorch inference, either full image of sliding window.
    """   
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if patch_size is not None:
            return sliding_window_inference(
                inputs=inputs,
                roi_size=patch_size,
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )
        else:
            return model(inputs)
    
def plot_loss_and_metric(axs, loss_values, metric_values, validation_epoch):
    """Plots training loss and metric
    """
    x = [i + 1 for i in range(len(loss_values))]
    y = loss_values
    axs[3, 0].plot(x, y, '.', markersize=1)
    axs[3, 0].set_title("Loss")
    axs[3, 0].set_xlabel("epoch")
    x = [validation_epoch * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    axs[3, 1].plot(x, y, '.', markersize=1)
    axs[3, 1].set_title("Metric")
    axs[3, 1].set_xlabel("epoch")