
import numpy as np
#import matplotlib.pyplot as plt
import tqdm
import pdb
import medicalDataLoader
import torchvision
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageOps
import skimage.transform as skiTransf

import torch
from random import random, randint

import torchvision.transforms as transforms
import os


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()
        
def augment(img, mask):
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() > 0.5:
            
            angle = random() * 90 - 45
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        return img, mask
        
            
def runTraining():
    print('-'*40)
    print('~~~~~~~~  Starting... ~~~~~~')
    print('-'*40)
                   
   
    batch_size = 1


    transform = transforms.Compose([
            transforms.ToTensor()
            ])

         
    mask_transform = transforms.Compose([
            MaskToTensor()
            ])

        
    #root_dir = '/home/AN82520/Projects/pyTorch/SegmentationFramework/DataSet/MICCAI_Bladder'
    #dest_dir = '/home/AN82520/Projects/pyTorch/SegmentationFramework/DataSet/Bladder_Aug'
   
    root_dir = '/export/livia/home/vision/jdolz/Projects/pyTorch/Corstem/ACDC-2D'
    dest_dir = '/export/livia/home/vision/jdolz/Projects/pyTorch/Corstem/ACDC-2D_Augmented'
 
 
    if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
                
                os.makedirs(dest_dir+'/train/Img')
                os.makedirs(dest_dir+'/train/GT')
                os.makedirs(dest_dir+'/val/Img')
                os.makedirs(dest_dir+'/val/GT')
                
    train_set = medicalDataLoader.MedicalImageDataset('train', root_dir,transform=transform, mask_transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=1, shuffle=True)

    val_set = medicalDataLoader.MedicalImageDataset('val', root_dir,transform=transform,mask_transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=1, shuffle=False)

    print(" ~~~~~~~~~~~ Augmenting dataset ~~~~~~~~~~")
    for data in train_loader:
        img_Size = 256  # HEART
        #img_Size = 320  # BLADDER
        image, labels, img_path = data
        image *= 255
        labels *= 255
        # Non-modified images
        img = Image.fromarray(image.numpy()[0].reshape((img_Size,img_Size)))
        mask = Image.fromarray(labels.numpy()[0].reshape((img_Size,img_Size)))
        # pdb.set_trace()
        
        
        #image, labels = data
       
        image,labels = augment(image.numpy()[0].reshape((img_Size,img_Size)),labels.numpy()[0].reshape((img_Size,img_Size)))
        
        name2save = img_path[0].split('.png')
        mainPath = name2save[0].split('Img')
        nameImage = mainPath[1]
        mainPath = mainPath[0]
        
        img = img.convert('RGB')
        img.save(dest_dir + '/train/Img' + nameImage + '.png',"PNG")
        mask = mask.convert('RGB')
        mask.save(dest_dir + '/train/GT' + nameImage + '.png',"PNG")
        
        image = image.convert('RGB')
        image.save(dest_dir + '/train/Img' + nameImage + '_Augm.png',"PNG")
        labels = labels.convert('RGB')
        labels.save(dest_dir + '/train/GT' + nameImage + '_Augm.png',"PNG")
        
        #pdb.set_trace()
        




if __name__ == '__main__':
    runTraining() 
