import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from .image_folder_dataset import ImageFolder
from .class_dict import class_dict


class ImageNetDataset(Dataset):
    '''
    ImageNet dataset.
    
    'train' and 'val' splits return CxHxW image tensors and classes,
    while 'test' split just returns the CxHxW image tensors.
    
    Args:
        root: Dataset root directory
        split:  One of {'train', 'val', 'test'}
        normalize: Set to True to normalize RGB images
    '''
    def __init__(self, root, split='train', image_size=224, normalize=True, augmentation='DEFAULT', force_refresh_tmp=False):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.augmentation = augmentation
        self.force_refresh_tmp = force_refresh_tmp

        self.mean = torch.Tensor([0.485, 0.456, 0.406]) if normalize else torch.Tensor([0,0,0])
        self.std = torch.Tensor([0.229, 0.224, 0.225]) if normalize else torch.Tensor([1,1,1])
        
        if self.split == 'train':
            if augmentation == 'random_perspective':
                self.transforms = transforms.Compose([
                    transforms.RandomAffine(degrees=10, translate=(0.1,0.1)),
                    transforms.RandomResizedCrop(size=self.image_size, scale=(0.75, 1), ratio=(1,1)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                    transforms.RandomHorizontalFlip(),
                ])
            else:
                self.transforms = transforms.Compose([
                    transforms.Resize(self.image_size, Image.BICUBIC),
                    transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                    transforms.RandomHorizontalFlip(),
                ])
        elif self.split in ['val', 'test']:
            self.transforms = transforms.Compose([
                transforms.Resize(self.image_size, Image.BICUBIC),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            raise ValueError(f'Invalid ImageNet split name: {self.split}')
            
        self.dataset = ImageFolder(
            root=os.path.join(self.root, self.split), 
            transform=self.transforms, 
            force_refresh_tmp=self.force_refresh_tmp
        )
            
    def class_to_label(self, class_idx):
        if isinstance(class_idx, int):
            return class_dict[class_idx]
        else:
            return [class_dict[idx.item()] for idx in class_idx]
    
    def normalize_img(self, img):
        return (img - self.mean.reshape(3,1,1)) / self.std.reshape(3,1,1)
    
    def denormalize_img(self, img):
        return (img * self.std.reshape(3,1,1)) + self.mean.reshape(3,1,1)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.split in ['train', 'val']:
            return self.dataset[index]
        else:
            return self.dataset[index][0]