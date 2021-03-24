import json
import numpy as np
import os
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from   typing import Optional

from . import task_configs

try:
    import accimage
except ImportError:
    pass

# from tlkit.utils import TASKS_TO_CHANNELS, FEED_FORWARD_TASKS

MAKE_RESCALE_0_1_NEG1_POS1   = lambda n_chan: transforms.Normalize([0.5]*n_chan, [0.5]*n_chan)
RESCALE_0_1_NEG1_POS1        = transforms.Normalize([0.5], [0.5])  # This needs to be different depending on num out chans
MAKE_RESCALE_0_MAX_NEG1_POS1 = lambda maxx: transforms.Normalize([maxx / 2.], [maxx * 1.0])
RESCALE_0_255_NEG1_POS1      = transforms.Normalize([127.5,127.5,127.5], [255, 255, 255])
MAKE_RESCALE_0_MAX_0_POS1 = lambda maxx: transforms.Normalize([0.0], [maxx * 1.0])

def get_transform(task: str, image_size=Optional[int]):
   
    if task in ['rgb', 'normal', 'reshading']:
        transform = transform_8bit
    elif task in ['mask_valid']:
        transform = transforms.ToTensor()
    elif task in ['keypoints2d', 'keypoints3d', 'depth_euclidean', 'depth_zbuffer', 'edge_texture', 'edge_occlusion']:
#         return transform_16bit_int
        transform = transform_16bit_single_channel
    elif task in ['principal_curvature', 'curvature']:
        transform = transform_8bit_n_channel(2)
    elif task in ['segment_semantic', 'segment_instance', 'fragments']:  # this is stored as 1 channel image (H,W) where each pixel value is a different class
        transform = transform_dense_labels
#     elif len([t for t in FEED_FORWARD_TASKS if t in task]) > 0:
#         return torch.Tensor
#     elif 'decoding' in task:
#         return transform_16bit_n_channel(TASKS_TO_CHANNELS[task.replace('_decoding', '')])
#     elif 'encoding' in task:
#         return torch.Tensor
    elif task in ['class_object', 'class_scene']:
        transform = torch.Tensor
        image_size = None
    elif task in ['segment_panoptic']:
        transform = transform_8bit_n_channel(n_channel=1, crop_channels=True)
    elif task in ['mesh', 'point_info']:
        return None
    else:
        raise NotImplementedError("Unknown transform for task {}".format(task))
    
    if 'clamp_to' in task_configs.task_parameters[task]:
        minn, maxx = task_configs.task_parameters[task]['clamp_to']
        if minn > 0:
            raise NotImplementedError("Rescaling (min1, max1) -> (min2, max2) not implemented for min1, min2 != 0 (task {})".format(task))
        transform = transforms.Compose([
                        transform,
                        MAKE_RESCALE_0_MAX_0_POS1(maxx)])

    if image_size is not None:
        if task == 'fragments':
            resize_frag = lambda frag: torch.nn.functional.interpolate(frag.permute(2,0,1).unsqueeze(0).float(), image_size, mode='nearest').long()[0].permute(1,2,0)
            transform = transforms.Compose([
                transform,
                resize_frag
            ])
        else:
            resize_method = Image.BILINEAR if task not in ['segment_instance', 'segment_semantic'] else Image.NEAREST
            transform = transforms.Compose([
                transforms.Resize(image_size, resize_method),
                transform])

    return transform

# For semantic segmentation
transform_dense_labels = lambda img: torch.Tensor(np.array(img)).long()  # avoids normalizing

# Transforms to a 3-channel tensor and then changes [0,1] -> [-1, 1]
transform_8bit = transforms.Compose([
        transforms.ToTensor(),
#         MAKE_RESCALE_0_1_NEG1_POS1(3),
    ])
    
# Transforms to a n-channel tensor and then changes [0,1] -> [-1, 1]. Keeps only the first n-channels
def transform_8bit_n_channel(n_channel=1, crop_channels=False):
    if crop_channels:
        crop_channels_fn = lambda x: x[:n_channel] if x.shape[0] > n_channel else x
    else: 
        crop_channels_fn = lambda x: x
    return transforms.Compose([
            transforms.ToTensor(),
            crop_channels_fn,
#             MAKE_RESCALE_0_1_NEG1_POS1(n_channel),
        ])

# Transforms to a 1-channel tensor and then changes [0,1] -> [-1, 1].
def transform_16bit_single_channel(im):
    im = transforms.ToTensor()(im)
    im = im.float() / (2 ** 16 - 1.0) 
#     return RESCALE_0_1_NEG1_POS1(im)
    return im


def transform_16bit_n_channel(n_channel=1):
    if n_channel == 1:
        return transform_16bit_single_channel # PyTorch handles these differently
    else:
        return transforms.Compose([
            transforms.ToTensor(),
#             MAKE_RESCALE_0_1_NEG1_POS1(n_channel),
        ])



def default_loader(path):
    if '.npy' in path:
        return np.load(path)
    elif '.json' in path:
        with open(path, 'r') as f:
            data_dict = json.load(f)
            data_dict['building'] = os.path.basename(os.path.dirname(path))
            data_dict.pop('nonfixated_points_in_view')
#             new_data = {}
#             new_data['camera_location'] = data_dict['camera_location']
#             new_data['camera_rotation_final'] = data_dict['camera_rotation_final']
#             new_data['field_of_view_rads'] = data_dict['field_of_view_rads']
#             data_dict = new_data
            return data_dict
    else:
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            im = accimage_loader(path)
        else:
            im = pil_loader(path)
        return im

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(img.mode)
        return img.convert('RGB')

# Faster than pil_loader, if accimage is available
def accimage_loader(path):
    return  accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


def segment_panoptic(x):
    print(task, flush=True)

