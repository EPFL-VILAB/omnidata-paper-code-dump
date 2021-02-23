import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../')

from seg_hrnet_multitask import hrnet_w18, hrnet_w48, HighResolutionHead
from resnet import resnet18, resnet50
from aspp import DeepLabHead
from data.segment_instance import COMBINED_CLASS_LABELS


def get_backbone(name):
    if name == 'resnet18':
        backbone = resnet18(pretrained=True)
        backbone_channels = 512
    
    elif name == 'resnet50':
        backbone = resnet50(pretrained=True)
        backbone_channels = 2048

    elif name == 'hrnet_w18':
        backbone = hrnet_w18(pretrained=False)
        backbone_channels = [18, 36, 72, 144]

    elif name == 'hrnet_w48':
        backbone = hrnet_w48(pretrained=True)
        backbone_channels = [48, 96, 192, 384]

    else:
        raise NotImplementedError

    return backbone, backbone_channels

def get_head(name, backbone_channels, task):
    """ Return the decoder head """
    if task == 'normal': n_output = 3
    elif task == 'segment_semantic': n_output = len(COMBINED_CLASS_LABELS) - 1
    else: raise NotImplementedError

    if name == 'deeplab':
        return DeepLabHead(backbone_channels, n_output)

    elif name == 'hrnet':
        return HighResolutionHead(backbone_channels, n_output)

    else:
        raise NotImplementedError


class MultiTaskModel(nn.Module):
    """ Multi-task baseline model with shared encoder + task-specific decoders """
    def __init__(self, tasks: list):
        super(MultiTaskModel, self).__init__()
        backbone, backbone_channels = get_backbone('hrnet_w48')
        heads = torch.nn.ModuleDict({
            task: get_head(name='hrnet', backbone_channels=backbone_channels, task=task) for task in tasks
            })
        self.backbone = backbone
        self.decoders = heads
        self.tasks = tasks

    def forward(self, x):
        out_size = x.size()[2:]
        shared_representation = self.backbone(x)
        return {task: F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear') for task in self.tasks}


