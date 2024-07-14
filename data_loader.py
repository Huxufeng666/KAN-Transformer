import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

class Noise_dataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle = True)
        # self.data = self.data[0:20]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image, label = self.data[index]
        
        image = cv2.resize(image, (224, 224))
        image = np.transpose(image, (2, 0, 1))
        label = int(label)
        # print('image/label: ', image.shape, label)
        
        return image, label
    
    
def build_transform(is_train, input_size, color_jitter = 0.3, aa = 'rand-m9-mstd0.5-inc1', train_interpolation = 'bicubic', reprob = 0.25, remode = 'pixel', recount = 1, eval_crop_ratio = 0.875):
    resize_im = input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size = input_size,
            is_training = True,
            color_jitter = color_jitter,
            auto_augment = aa,
            interpolation = train_interpolation,
            re_prob = reprob,
            re_mode = remode,
            re_count = recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(input_size / eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)