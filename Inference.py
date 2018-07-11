from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar


import medicalDataLoader
from UNet import *
from utils import *
import sys

import time


from optimizer import Adam

# CUDA_VISIBLE_DEVICES=6 python Inference.py ./model/Best_UNetG_Dilated_Progressive.pkl Best_UNetG_Dilated_Progressive_Inference
def runInference(argv):
    print('-' * 40)
    print('~~~~~~~~  Starting the inference... ~~~~~~')
    print('-' * 40)

    batch_size_val = 1
    batch_size_val_save = 1
    batch_size_val_savePng = 1

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    root_dir = '../DataSet/Bladder_Aug'
    modelName = 'UNetG_Dilated_Progressive'
    
    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            num_workers=5,
                            shuffle=False)
                            

    val_loader_save_images = DataLoader(val_set,
                                        batch_size=batch_size_val_save,
                                        num_workers=5,
                                        shuffle=False)
                                        
    modelName = argv[0]
    
    print('...Loading model...')
    try:
        netG = torch.load(modelName)
        print("--------model restored--------")
    except:
        print("--------model not restored--------")
        pass

    netG.cuda()
    
    modelName_dir = argv[1]
    
    # To save images as png
    saveImages(netG, val_loader_save_images, batch_size_val_save, 0, modelName)

    # To save images as Matlab
    saveImagesAsMatlab(netG, val_loader_save_images, batch_size_val_save, 0, modelName)
    
    print("###                               ###")
    print("###   Images saved      ###")
    print("###                               ###")
    

if __name__ == '__main__':
    runInference(sys.argv[1:])
