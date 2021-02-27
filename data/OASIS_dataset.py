import torch

import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import functools

from PIL import Image

import os
import os.path
import numpy as np
import torch.utils.data as data


def make_dataset(dir):
    images = []
    for img in sorted(os.listdir(dir)):
        path = os.path.join(dir, img)
        images.append(path)
    return images


to_tensor = transforms.ToTensor()
RGB_MEAN = torch.Tensor([0.55312, 0.52514, 0.49313]).reshape(3,1,1)
RGB_STD =  torch.Tensor([0.20555, 0.21775, 0.24044]).reshape(3,1,1)


class OASISDataset(data.Dataset):

    def __init__(self, root, output_size, normalized=False):

        imgs = make_dataset(root)
        print(len([im for im in imgs if im[1] == 1]))

        assert len(imgs) > 0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))

        self.root = root
        self.imgs = imgs
        self.output_size = output_size

        resize_method = Image.BILINEAR 
        self.transform = transforms.Compose([
            transforms.Resize(self.output_size, resize_method), 
            transforms.CenterCrop(self.output_size),
            transforms.ToTensor()])

        if normalized:
            self.transform = transforms.Compose(
                self.transform.transforms + [transforms.Normalize(mean=RGB_MEAN, std=RGB_STD)]
            )

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path)
        res = self.transform(img)
        return res

    def __len__(self):
        return len(self.imgs)



    

    
