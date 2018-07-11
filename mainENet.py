from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar
import medicalDataLoader
from UNet import *
from utils import *

import time

from optimizer import Adam

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]
        

            
def resizeTensorMaskInSingleImage(batch, scalingFactor):
    data = batch.cpu().data.numpy()
    batch_s = data.shape[0]
    numClasses = data.shape[1]
    img_size = data.shape[2]
    # TODO: Better way to define this
    resizedLabels = np.zeros((batch_s,
                              int(img_size/scalingFactor),
                              int(img_size/scalingFactor)))
    
                            
    for i in range(data.shape[0]):
        img = data[i,:,:].reshape(img_size,img_size)
        imgL = np.zeros((img_size,img_size))
        idx1t = np.where(img==1)
        imgL[idx1t]=1
        imgRes = skiTransf.resize(imgL,(img_size/scalingFactor,img_size/scalingFactor),preserve_range=True)
        idx1 = np.where(imgRes>=0.5)
        
        imgL = np.zeros((img_size,img_size))
        idx2t = np.where(img==1)
        imgL[idx2t]=1
        imgRes = skiTransf.resize(imgL,(img_size/scalingFactor,img_size/scalingFactor),preserve_range=True)
        idx2 = np.where(imgRes>=0.5)
        
        imgL = np.zeros((img_size,img_size))
        idx3t = np.where(img==1)
        imgL[idx3t]=1
        imgRes = skiTransf.resize(imgL,(img_size/scalingFactor,img_size/scalingFactor),preserve_range=True)
        idx3 = np.where(imgRes>=0.5)
        
        imgResized = np.zeros((int(img_size/scalingFactor),int(img_size/scalingFactor)))
        imgResized[idx1]=1
        imgResized[idx2]=2
        imgResized[idx3]=3
        
        
        resizedLabels[i,:,:]=imgResized
            
    tensorClass = torch.from_numpy(resizedLabels).long()
    return Variable(tensorClass.cuda())
    
def runTraining():
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    batch_size = 4
    batch_size_val = 1
    batch_size_val_save = 1
    batch_size_val_savePng = 4
    lr = 0.0001
    epoch = 1000
    root_dir = '../DataSet/Bladder_Aug'
    modelName = 'UNetG_Dilated_Progressive'
    model_dir = 'model'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = medicalDataLoader.MedicalImageDataset('train',
                                                      root_dir,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=False,
                                                      equalize=False)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=5,
                              shuffle=True)

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

    val_loader_save_imagesPng = DataLoader(val_set,
                                        batch_size=batch_size_val_savePng,
                                        num_workers=5,
                                        shuffle=False)                                                                     
    # Initialize
    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")
    num_classes = 4
    
    initial_kernels = 4
    
    # Load network
    netG = UNetG_Dilated_Progressive(1, initial_kernels, num_classes)
    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    Dice_loss = computeDiceOneHot()
    
    if torch.cuda.is_available():
        netG.cuda()
        softMax.cuda()
        CE_loss.cuda()
        Dice_loss.cuda()

    '''try:
        netG = torch.load('./model/Best_UNetG_Dilated_Progressive_Stride_Residual_ChannelsFirst32.pkl')
        print("--------model restored--------")
    except:
        print("--------model not restored--------")
        pass'''
        
    optimizerG = Adam(netG.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=False)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizerG, mode='max', patience=4, verbose=True,
                                                       factor=10 ** -0.5)
   
    BestDice, BestEpoch = 0, 0

    d1Train = []
    d2Train = []
    d3Train = []
    d1Val = []
    d2Val = []
    d3Val = []
    
    Losses = []
    Losses1 = []
    Losses05 = []
    Losses025 = []
    Losses0125 = []

    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    for i in range(epoch):
        netG.train()
        lossVal = []
        lossValD = []
        lossVal1 = []
        lossVal05 = []
        lossVal025 = []
        lossVal0125 = []
        
        d1TrainTemp = []
        d2TrainTemp = []
        d3TrainTemp = []
        
        timesAll = []
        success = 0
        totalImages = len(train_loader)
        
        for j, data in enumerate(train_loader):
            image, labels, img_names = data

            # prevent batchnorm error for batch of size 1
            if image.size(0) != batch_size:
                continue

            optimizerG.zero_grad()
            MRI = to_var(image)
            Segmentation = to_var(labels)
            
            target_dice = to_var(torch.ones(1))
            
            ################### Train ###################
            netG.zero_grad()

            deepSupervision = False
            multiTask = False
            
            start_time = time.time()
            if deepSupervision == False and multiTask == False:
                # No deep supervision
                segmentation_prediction = netG(MRI)
            else:
                # Deep supervision
                if deepSupervision == True:
                    segmentation_prediction, seg_3, seg_2, seg_1 = netG(MRI)
                else:
                    segmentation_prediction,reg_output = netG(MRI)
                    # Regression
                    feats = getValuesRegression(labels)
   
                    feats_t = torch.from_numpy(feats).float()
                    featsVar = to_var(feats_t)
            
                    MSE_loss_val = MSE_loss(reg_output,featsVar)
            
            predClass_y = softMax(segmentation_prediction)
         
            spentTime = time.time()-start_time
  
            timesAll.append(spentTime/batch_size) 
            
            Segmentation_planes = getOneHotSegmentation(Segmentation)
            segmentation_prediction_ones = predToSegmentation(predClass_y)

            # It needs the logits, not the softmax
            Segmentation_class = getTargetSegmentation(Segmentation)

            # No deep supervision
            CE_lossG = CE_loss(segmentation_prediction, Segmentation_class)
            if deepSupervision == True:
                
                imageLabels_05 = resizeTensorMaskInSingleImage(Segmentation_class, 2)
                imageLabels_025 = resizeTensorMaskInSingleImage(Segmentation_class, 4)
                imageLabels_0125 = resizeTensorMaskInSingleImage(Segmentation_class, 8)
            
                CE_lossG_3 = CE_loss(seg_3, imageLabels_05)
                CE_lossG_2 = CE_loss(seg_2, imageLabels_025)
                CE_lossG_1 = CE_loss(seg_1, imageLabels_0125)
            
            '''weight = torch.ones(4).cuda() # Num classes
            weight[0] = 0.2
            weight[1] = 0.2
            weight[2] = 1
            weight[3] = 1
            
            CE_loss.weight = weight'''

            # Dice loss
            DicesN, DicesB, DicesW, DicesT = Dice_loss(segmentation_prediction_ones, Segmentation_planes)
            DiceN = DicesToDice(DicesN)
            DiceB = DicesToDice(DicesB)
            DiceW = DicesToDice(DicesW)
            DiceT = DicesToDice(DicesT)

            Dice_score = (DiceB + DiceW + DiceT) / 3
           
            if deepSupervision == False and multiTask == False:
                lossG = CE_lossG 
            else:
                # Deep supervision
                if deepSupervision == True:
                    lossG = CE_lossG  + 0.25*CE_lossG_3 + 0.1*CE_lossG_2 + 0.1*CE_lossG_1
                else:
                    lossG = CE_lossG + 0.000001*MSE_loss_val
   
            
            lossG.backward()
            optimizerG.step()
            
            lossVal.append(lossG.data[0])
            lossVal1.append(CE_lossG.data[0])
            
            if deepSupervision == True:
                lossVal05.append(CE_lossG_3.data[0])
                lossVal025.append(CE_lossG_2.data[0])
                lossVal0125.append(CE_lossG_1.data[0])

            printProgressBar(j + 1, totalImages,
                             prefix="[Training] Epoch: {} ".format(i),
                             length=15,
                             suffix=" Mean Dice: {:.4f}, Dice1: {:.4f} , Dice2: {:.4f}, , Dice3: {:.4f} ".format(
                                 Dice_score.data[0],
                                 DiceB.data[0],
                                 DiceW.data[0],
                                 DiceT.data[0]))

        if deepSupervision == False:
            '''printProgressBar(totalImages, totalImages,
                             done="[Training] Epoch: {}, LossG: {:.4f},".format(i,np.mean(lossVal),np.mean(lossVal1)))'''
            printProgressBar(totalImages, totalImages,
                             done="[Training] Epoch: {}, LossG: {:.4f}, lossMSE: {:.4f}".format(i,np.mean(lossVal),np.mean(lossVal1)))
        else:
            printProgressBar(totalImages, totalImages,
                             done="[Training] Epoch: {}, LossG: {:.4f}, Loss4: {:.4f}, Loss3: {:.4f}, Loss2: {:.4f}, Loss1: {:.4f}".format(i,
                                                                                                                                           np.mean(lossVal),
                                                                                                                                           np.mean(lossVal1),
                                                                                                                                           np.mean(lossVal05),
                                                                                                                                           np.mean(lossVal025),
                                                                                                                                           np.mean(lossVal0125)))

        Losses.append(np.mean(lossVal))

        d1,d2,d3 = inference(netG, val_loader, batch_size, i, deepSupervision)
         
        d1Val.append(d1)
        d2Val.append(d2)
        d3Val.append(d3)

        d1Train.append(np.mean(d1TrainTemp).data[0])
        d2Train.append(np.mean(d2TrainTemp).data[0])
        d3Train.append(np.mean(d3TrainTemp).data[0])

        mainPath = '../Results/Statistics/' + modelName
        
        directory = mainPath
        if not os.path.exists(directory):
            os.makedirs(directory)

        ###### Save statistics  ######
        np.save(os.path.join(directory, 'Losses.npy'), Losses)
         
        np.save(os.path.join(directory, 'd1Val.npy'), d1Val)
        np.save(os.path.join(directory, 'd2Val.npy'), d2Val)
        np.save(os.path.join(directory, 'd3Val.npy'), d3Val)

        np.save(os.path.join(directory, 'd1Train.npy'), d1Train)
        np.save(os.path.join(directory, 'd2Train.npy'), d2Train)
        np.save(os.path.join(directory, 'd3Train.npy'), d3Train)

        currentDice = (d1+d2+d3)/3 

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        torch.save(netG, os.path.join(model_dir, "Best_" + modelName + ".pkl"))
        # How many slices with/without tumor correctly classified
        print("[val] DSC: (1): {:.4f} (2): {:.4f}  (3): {:.4f} ".format(d1,d2,d3))
        
        if currentDice > BestDice:
            BestDice = currentDice
            BestDiceT = d1
            BestEpoch = i
            if currentDice > 0.7:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Saving best model..... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(netG, os.path.join(model_dir, "Best_" + modelName + ".pkl"))

                # Save images
                saveImages(netG, val_loader_save_images, batch_size_val_save, i, modelName, deepSupervision)
                saveImagesAsMatlab(netG, val_loader_save_images, batch_size_val_save, i, modelName, deepSupervision)

        print("###                                                       ###")
        print("###    Best Dice: {:.4f} at epoch {} with DiceT: {:.4f}    ###".format(BestDice, BestEpoch, BestDiceT))
        print("###                                                       ###")

        # This is not as we did it in the MedPhys paper
        if i % (BestEpoch + 20):
            for param_group in optimizerG.param_groups:
                param_group['lr'] = lr/2


if __name__ == '__main__':
    runTraining()
