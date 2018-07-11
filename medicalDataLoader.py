from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# Ignore warnings
import warnings

import pdb
warnings.filterwarnings("ignore")

#plt.ion()   # interactive mode

def make_dataset(root,mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        train_img_path = os.path.join(root, 'train','Img')
        train_mask_path = os.path.join(root, 'train','GT')

        images = os.listdir(train_img_path)
        labels =  os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im,it_gt in zip(images,labels):
            item = (os.path.join(train_img_path,it_im), os.path.join(train_mask_path,it_gt))
            items.append(item)
    elif mode == 'val':
        train_img_path = os.path.join(root, 'val','Img')
        train_mask_path = os.path.join(root, 'val','GT')

        images = os.listdir(train_img_path)
        labels =  os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im,it_gt in zip(images,labels):
            item = (os.path.join(train_img_path,it_im), os.path.join(train_mask_path,it_gt))
            items.append(item)
    else:
        train_img_path = os.path.join(root, 'test','Img')
        train_mask_path = os.path.join(root, 'test','GT')

        images = os.listdir(train_img_path)
        labels =  os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im,it_gt in zip(images,labels):
            item = (os.path.join(train_img_path,it_im), os.path.join(train_mask_path,it_gt))
            items.append(item)
  
    return items
    
class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, mode, root_dir, transform=None, mask_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir,mode)
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        #print("{} and {}".format(img_path,mask_path))
        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform is not None:
            img = self.transform(img)
            mask = self.mask_transform(mask)

        return [img, mask]
