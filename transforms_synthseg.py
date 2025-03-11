"""Custom transforms for Synthseg training in MONAI
"""
import cornucopia as cc
import torch
from random import randrange

from monai.transforms import (
    MapTransform,
    RandomizableTransform,
    Resized,
    ResizeWithPadOrCropd,
)
from monai.data.meta_tensor import MetaTensor

from utils_synthseg import (
    extract_slice,
)

class RemoveEmptySlices(MapTransform):
    """Transform to get slice from volume.
    """
    def __init__(self, key_label="label"):
        self.key_label = key_label    

    def __call__(self, data):
        d = dict(data)

        if self.key_label in d:
            label = d[self.key_label]
            meta = label.meta
            sLabel = label.sum(dim=(0, 1, 2))
            ix = (sLabel != 0).nonzero().flatten()
            label = torch.index_select(label, 3, ix)            
            d[self.key_label] = MetaTensor(label)
            d[self.key_label].copy_meta_from(meta)

        return d
    
class LabelUnlabelledGMM(MapTransform):
    """Use a GMM to label unlabelled voxels.
    """
    def __init__(self, n_labels=None, num_gmm_classes=10, key_label="label", key_image="image", verbose=False):
        self.n_labels = n_labels
        self.key_label = key_label
        self.key_image = key_image
        self.num_gmm_classes = num_gmm_classes        
        self.verbose = verbose

    def __call__(self, data):
        d = dict(data)

        # Get label and image data
        label = d[self.key_label]
        meta = label.meta
        fname = meta["filename_or_obj"]
        if self.verbose:  print(f"LabelUnlabelledGMM | Filename: {fname}")
        label = label.as_tensor()
        image = d[self.key_image]
        image = image.as_tensor()
        image_shape = image.shape[1:]
        # Get data
        label = label[0, ...]
        image = image[0, ...]
        if self.n_labels is None:
            self.n_labels = label.max()
        if self.verbose:  print(f"LabelUnlabelledGMM | Number of unique input labels (inc zero): {len(label.unique())}")
        
        # Get unlabelled intensity voxels
        unlabelled = label == 0 
        image = image[unlabelled].reshape(1, -1).float()
         
        # Fit GMM to unlabelled intensity voxels
        if self.verbose:  print(f"LabelUnlabelledGMM | Number of input GMM classes: {self.num_gmm_classes}")
        Zu = cc.utils.gmm.fit_gmm(image, nk=self.num_gmm_classes, max_iter=1024)[0]
        # 1D to 3D        
        Z = torch.zeros((self.num_gmm_classes, unlabelled.numel()), dtype=Zu.dtype, device=Zu.device)
        for k in range(self.num_gmm_classes):
            Z[k, unlabelled.flatten()] = Zu[k, :]
        if len(image_shape) == 3:
            Z = Z.reshape((self.num_gmm_classes,) + image_shape).permute(1,2,3,0)
        else:
            Z = Z.reshape((self.num_gmm_classes,) + image_shape).permute(1,2,0)
        # one-hot to label map
        Z =  torch.argmax(Z, dim=-1)
        if self.verbose:  print(f"LabelUnlabelledGMM |  Number of output GMM classes: {len(Z.unique())}")
        # Correct data type
        Z = Z.type(label.dtype)

        # Assign new labels
        # OBS: The number of unique output labels should be: n_input_classes + (n_gmm_classes - 1)
        # As the background class (zero) from the GMM fit is not added, but merged with the existing
        # background class
        label[(unlabelled) & (Z == 0)] = Z[(unlabelled) & (Z == 0)]
        label[(unlabelled) & (Z > 0)] = (Z[(unlabelled) & (Z > 0)] + self.n_labels)
        if self.verbose:  print(f"LabelUnlabelledGMM | Number of unique output labels (inc zero): {len(label.unique())}")
        
        # Make output
        d[self.key_label] = MetaTensor(label[None])
        d[self.key_label].copy_meta_from(meta)

        return d


class MapLabelsSAROS(MapTransform):
    """Transform to map Neurite datas' label indices to contigious values starting at one.
    """
    @staticmethod
    def label_mapping():
        return {
            1:1,
            2:2,
            3:3,
            4:4,
            5:5,
            6:6,
            7:7,
            9:8,
            10:9,
            11:10,
            12:11,
            13:12,
        }

    def __init__(self, key_label="label"):
        self.key_label = key_label    
        self.lm = MapLabelsNeurite.label_mapping()
        self.allow_missing_keys = False

    def __call__(self, data):
        d = dict(data)

        label = d[self.key_label]
        meta = label.meta
        label = label.as_tensor()
        
        # Make contiguous
        label_contiguous = torch.zeros_like(label)
        for val in self.lm:
            label_contiguous[label == val] = \
                self.lm[val]
        
        d[self.key_label] = MetaTensor(label_contiguous)
        d[self.key_label].copy_meta_from(meta)

        return d
    

class MapLabelsSynthSeg(MapTransform):
    """Transform to map SynthSeg datas' label indices to contigious values starting at one.
    """
    @staticmethod
    def label_mapping():
        return {
            2:1,
            3:2,
            4:3,
            5:4,
            7:5,
            8:6,
            10:7,
            11:8,
            12:9,
            13:10,
            14:11,
            15:12,
            16:13,
            17:14,
            18:15,
            26:16,
            28:17,
            41:18,
            42:19,
            43:20,
            44:21,
            46:22,
            47:23,
            49:24,
            50:25,
            51:26,
            52:27,
            53:28,
            54:29,
            58:30,
            60:31,
        }

    def __init__(self, key_label="label"):
        self.key_label = key_label    
        self.lm = MapLabelsSynthSeg.label_mapping()

    def __call__(self, data):
        d = dict(data)

        label = d[self.key_label]
        meta = label.meta
        label = label.as_tensor()

        label_contiguous = torch.zeros_like(label)
        for val_synthseg in self.lm:
            label_contiguous[label == val_synthseg] = \
                self.lm[val_synthseg]
        
        l1 = list(self.lm.keys())
        l2 = label.unique().type(torch.int64).cpu().tolist()
        ld = list(set(l2) - set(l1))
        ld.sort()
        
        cnt = self.lm[val_synthseg] + 1
        for val_diff in ld:
            if val_diff == 0:  continue
            label_contiguous[label == val_diff] = cnt
            cnt += 1

        d[self.key_label] = MetaTensor(label_contiguous)
        d[self.key_label].copy_meta_from(meta)

        return d


class MapLabelsNeurite(MapTransform):
    """Transform to map Neurite datas' label indices to contigious values starting at one.
    """
    @staticmethod
    def label_mapping():
        l0 = list(range(1, 18)) + list(range(20, 34))
        l1 = list(range(1,32))    
        mapping = {}
        for i0, i1, in zip(l0, l1):
            mapping[i0] = i1

        return mapping

    def __init__(self, key_label="label"):
        self.key_label = key_label    
        self.lm = MapLabelsNeurite.label_mapping()
        self.allow_missing_keys = False

    def __call__(self, data):
        d = dict(data)

        label = d[self.key_label]
        meta = label.meta
        label = label.as_tensor()
        
        # Make contiguous
        label_contiguous = torch.zeros_like(label)
        for val in self.lm:
            label_contiguous[label == val] = \
                self.lm[val]
        
        d[self.key_label] = MetaTensor(label_contiguous)
        d[self.key_label].copy_meta_from(meta)

        return d


def Resize(spatial_size, testing):
    """Returns a MONAI resise transform. If testing, returns Resized,
    else returns ResizeWithPadOrCropd.
    """
    if testing:
        return Resized(keys=["label"], spatial_size=spatial_size, mode="nearest", lazy=True,)
    else:
        return ResizeWithPadOrCropd(keys=["label"], spatial_size=spatial_size, lazy=True,)
    
    
class SynthSegd(MapTransform, RandomizableTransform):
    """Transform to SynthSeg transform a label map to an intensity image.
    """
    def __init__(self, 
                 key_image="image", 
                 key_label="label",
                 patch_size=None,
                 params=None,
        ):
        RandomizableTransform.__init__(self, 1.0)        
        self.key_image = key_image
        self.key_label = key_label
        if params is None: params = {}
        self.params = params
        self.patch_size = patch_size

    def __call__(self, data):
        d = dict(data)

        label = d[self.key_label]
        meta = label.meta
        label = label.as_tensor()

        synth = cc.SynthFromLabelTransform(
            **self.params,
            patch=self.patch_size,
        )
        image, label = synth(label)

        d[self.key_image] = MetaTensor(image)
        d[self.key_image].copy_meta_from(meta)

        d[self.key_label] = MetaTensor(label)
        d[self.key_label].copy_meta_from(meta)

        return d


class SliceFromVolume(MapTransform):
    """Transform to get slice from volume.
    """
    def __init__(self, do=False, key_image="image", key_label="label"):
        self.key_label = key_label    
        self.key_image = key_image
        self.do = do

    def __call__(self, data):
        d = dict(data)
        if not self.do: return d

        if self.key_image in d:
            image = d[self.key_image]
            meta = image.meta
            image = extract_slice(image)

            d[self.key_image] = MetaTensor(image)
            d[self.key_image].copy_meta_from(meta)

        if self.key_label in d:
            label = d[self.key_label]
            meta = label.meta         
            n_slices = label.shape[-1]
            
            while True:
                ix_slice = [0, 0, randrange(n_slices)]
                label_slice = extract_slice(label, mid_ix=ix_slice)                
#                 u = label_slice.unique()
#                 print(u)
#                 print(label_slice.sum())
                if label_slice.sum() > 0:
                    break
           
            d[self.key_label] = MetaTensor(label_slice)
            d[self.key_label].copy_meta_from(meta)

        return d
