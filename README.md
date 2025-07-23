# SynthSeg in MONAI

<p align="center">
  <img style="float: right;" src="https://github.com/brudfors/synthseg-monai/blob/main/assets/synthseg-approach.png" width="100%" height="100%">
</p>

This code implements the [SynthSeg](https://github.com/BBillot/SynthSeg) approach for 3D medical image segmentation using the [MONAI](https://github.com/Project-MONAI/MONAI) and [Cornucopia](https://github.com/balbasty/cornucopia) packages. MONAI is used for the deep learning bits and Cornucopia for GPU-based synthesizing. 

In this repo, SynthSeg is trained for 3D brain segmentation, but the code can easily be adopted for other organs/structures. The training labels were taken from the official [SynthSeg repository](https://github.com/BBillot/SynthSeg/tree/master/data/training_label_maps) and include 20 brain label maps (i.e., segmentations).

The code supports training on both the full input data, or on patches. Multi-node/multi-GPU training is also supported.

#### Tips and Tricks
* Depending on image size and number of classes you can experiment with the `spatial_size`, `batch_size` and `patch_size` parameters.
* You can monitor the training progress by looking at the `snapshot.png` image, written to the `results` folder.
* You can increase/decrease randomness during training by modifying the `utils_synthseg.get_synth_params` function.
* A Gaussian Mixture Model (GMM) can used to create labels for voxels that are not labelled (see `transforms_synthseg.LabelUnlabelledGMM`).

## Install Dependencies

To install the required dependencies do:
```sh
pip install -r requirements.txt
```

## Download Trained Model

A model trained with the `train_synthseg.py` script can be downloaded from [here](https://www.dropbox.com/scl/fi/q4my4tnyk6ftnd132tmb6/model_best.pth?rlkey=aowqsrl9cw9cccsntylm12n3s&st=yv04stz3&dl=1). Training was done on an NVIDIA A100 GPU, it took about 8 days and required around 36 GB VRAM. 

If you want to use the model with either the `validate_synthseg.ipynb` or the `test_synthseg.ipynb` notebook, copy the model to a `./results` folder.

## Model Testing

To test the above trained model, run either `validate_synthseg.ipynb` (uses the original training data, but with minimal augmentations) or `test_synthseg.ipynb` (uses unseen MR images from [this repository](https://github.com/neurolabusc/niivue-images)).

## Model Training

To train a new model, download the training data from [here](https://github.com/BBillot/SynthSeg/tree/master/data/training_label_maps) and run the script:
```sh
python train_synthseg.py
```
Default settings are:
```py
dir_input = "./training_label_maps" # Folder path of training labels (available from https://github.com/BBillot/SynthSeg/tree/master/data/training_label_maps)
dir_results = "./results" # Folder path where to write results (checkpoint, best model, figures)
batch_size = 1
total_steps = 400000 # total_steps = epochs x len(train_files)
validation_steps = 2000 # steps between each validation
spatial_size = None # Resize input to this size, eg (256,) * 3, use None for no resize
patch_size = None # Training patch size, eg (128,) * 3, use None for training on full input
```
If you have a system with multiple GPUs and/or nodes (e.g., 8 GPUs) you can speed up training with:
```sh
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 train_synthseg.py
```
Training results for the model in the *Download Trained Model* section are visualized in the below "snapshot" where the top row shows a random training sample from the final epoch, the second row shows the prediction on a validation image from the final epoch, the third row shows the best validation prediction over all epochs, and the fourth row shows the training loss and the validation metric (Dice).

<p align="center">
  <img style="float: right;" src="https://github.com/brudfors/synthseg-monai/blob/main/assets/snapshot.png" width="80%" height="80%">
</p>
