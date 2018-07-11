import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
import skimage.transform as skiTransf
from progressBar import printProgressBar
import scipy.io as sio
import pdb
import time

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class computeDiceOneHotBinary(nn.Module):
    def __init__(self):
        super(computeDiceOneHotBinary, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        # GT is 4x320x320 of 0 and 1
        # pred is converted to 0 and 1
        batchsize = GT.size(0)
        DiceN = to_var(torch.zeros(batchsize, 2))
        DiceB = to_var(torch.zeros(batchsize, 2))
        #DiceW = to_var(torch.zeros(batchsize, 2))
        #DiceT = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            DiceN[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceB[i, 0] = self.inter(pred[i, 1], GT[i, 1])
            #DiceW[i, 0] = self.inter(pred[i, 2], GT[i, 2])
            #DiceT[i, 0] = self.inter(pred[i, 3], GT[i, 3])

            DiceN[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceB[i, 1] = self.sum(pred[i, 1], GT[i, 1])
            #DiceW[i, 1] = self.sum(pred[i, 2], GT[i, 2])
            #DiceT[i, 1] = self.sum(pred[i, 3], GT[i, 3])

        return DiceN, DiceB #, DiceW, DiceT
        
        
class computeDiceOneHot(nn.Module):
    def __init__(self):
        super(computeDiceOneHot, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        # GT is 4x320x320 of 0 and 1
        # pred is converted to 0 and 1
        batchsize = GT.size(0)
        DiceN = to_var(torch.zeros(batchsize, 2))
        DiceB = to_var(torch.zeros(batchsize, 2))
        DiceW = to_var(torch.zeros(batchsize, 2))
        DiceT = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            DiceN[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceB[i, 0] = self.inter(pred[i, 1], GT[i, 1])
            DiceW[i, 0] = self.inter(pred[i, 2], GT[i, 2])
            DiceT[i, 0] = self.inter(pred[i, 3], GT[i, 3])

            DiceN[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceB[i, 1] = self.sum(pred[i, 1], GT[i, 1])
            DiceW[i, 1] = self.sum(pred[i, 2], GT[i, 2])
            DiceT[i, 1] = self.sum(pred[i, 3], GT[i, 3])

        return DiceN, DiceB , DiceW, DiceT


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


class Hausdorff(nn.Module):
    def __init__(self, dist=320 ** 2):
        super(Hausdorff, self).__init__()
        self.dist = dist

    def edgePixels(self, img, channel):  # , GT):
        # return the list of pixels from the edge of the surface
        padVal = 0
        # changing padValue for channel 0 to avoid detecting edge of the image
        if channel == 0:
            padVal = 1
        sub = F.pad(img[1:, :], (0, 0, 0, 1), value=padVal) + F.pad(img[:-1, :], (0, 0, 1, 0), value=padVal) + \
              F.pad(img[:, 1:], (0, 1, 0, 0), value=padVal) + F.pad(img[:, :-1], (1, 0, 0, 0), value=padVal)
        edge = 4 * img - sub.data

        return edge.clamp(0, 1).nonzero().float()

    def maxminDist(self, edge_pred, edge_GT):
        dists = (edge_GT.expand(len(edge_pred), *edge_GT.size()) - edge_pred.unsqueeze(1)).pow(2).sum(-1)

        return dists.min(-1)[0].max()

    def forward(self, pred, GT):
        # Computes the Hausdorff distance between two segmentations
        # The inputs are Variables with N channels corresponding to N classes
        # for each pixel, only one channel must be equal to 1 and all others to 0
        batchsize, n_classes, _, _ = pred.size()
        hausdorffs = to_var(torch.zeros(batchsize, n_classes))

        for i in range(batchsize):
            for j in range(n_classes):
                edge_pred = self.edgePixels(pred.data[i, j], j)
                if len(edge_pred) == 0:
                    continue
                edge_GT = self.edgePixels(GT.data[i, j], j)
                if len(edge_GT) == 0:
                    continue

                hausdorffs[i, j] = self.maxminDist(edge_pred, edge_GT)

        return hausdorffs.sum() / self.dist


def getSingleImageBin(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    Val = to_var(torch.zeros(2))
    Val[1] = 1.0
    
    x = predToSegmentation(pred)

    out = x * Val.view(1, 2, 1, 1)
    return out.sum(dim=1, keepdim=True)
    
def getSingleImage(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    Val = to_var(torch.zeros(4))

    # Heart
    #Val[1] = 0.33333334
    #Val[2] = 0.66666669
    #Val[3] = 1.0

    # Bladder
    Val[1] = 0.3137255
    Val[2] = 0.627451
    Val[3] = 0.94117647
    
    x = predToSegmentation(pred)

    out = x * Val.view(1, 4, 1, 1)
    return out.sum(dim=1, keepdim=True)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getOneHotTumorClass(batch):
    data = batch.cpu().data.numpy()
    classLabels = np.zeros((data.shape[0], 2))

    tumorVal = 1.0
    for i in range(data.shape[0]):
        img = data[i, :, :, :]
        values = np.unique(img)
        if len(values) > 3:
            classLabels[i, 1] = 1
        else:
            classLabels[i, 0] = 1

    tensorClass = torch.from_numpy(classLabels).float()

    return Variable(tensorClass.cuda())


def getOneHotSegmentation(batch):
    backgroundVal = 0
    # Heart
    #label1 = 0.33333334
    #label2 = 0.66666669
    #label3 = 1.0

    # Bladder
    label1 = 0.3137255
    label2 = 0.627451
    label3 = 0.94117647
   
    oneHotLabels = torch.cat((batch == backgroundVal, batch == label1, batch == label2, batch == label3),
                             dim=1)
    return oneHotLabels.float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.3137255, 0.627451 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3
    spineLabel = 0.33333334
    return (batch / spineLabel).round().long().squeeze()

from scipy import ndimage

def getValuesRegression(image):
    label1 = 0.33333334
    
    feats = np.zeros((image.shape[0],3))

    for i in range(image.shape[0]):
        imgT = image[i,0,:,:].numpy()
        idx = np.where(imgT==label1)
        img = np.zeros(imgT.shape)
        img[idx]=1
        sizeRV = len(idx[0])
        [x,y] = ndimage.measurements.center_of_mass(img)
        
        if sizeRV == 0:
            x = 0
            y = 0
            
        feats[i,0] = sizeRV
        feats[i,1] = x
        feats[i,2] = y
        
        
        #print(' s: {}, x: {}, y: {} '.format(sizeRV,x,y))
        
    return feats
    
    
def saveImages(net, img_batch, batch_size, epoch, modelName, deepSupervision=0):
    # print(" Saving images.....")
    #path = 'Results/ENet-Original'
    path = '../Results/Images_PNG/' + modelName + '_'+ str(epoch) 
    if not os.path.exists(path):
        os.makedirs(path)
    total = len(img_batch)
    net.eval()
    softMax = nn.Softmax()
    times = []
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="Saving images.....", length=30)
        image, labels, img_names = data

        MRI = to_var(image)
        Segmentation = to_var(labels)
            
        if deepSupervision == False:
            # No deep supervision
            tic = time.clock()
            segmentation_prediction = net(MRI)
            toc = time.clock()
            times.append(toc-tic)
        else:
            # Deep supervision
            segmentation_prediction, seg_3, seg_2, seg_1 = net(MRI)

        pred_y = softMax(segmentation_prediction)
        segmentation = getSingleImage(pred_y)
        
        #segmentation = getSingleImageBin(segmentation_prediction)
        
        out = torch.cat((MRI, segmentation, Segmentation))

        torchvision.utils.save_image(out.data, os.path.join(path, str(i) + '_Ep_' + str(epoch) + '.png'),
                                     nrow=batch_size,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False,
                                     pad_value=0)

    printProgressBar(total, total, done="Images saved !")
        

def saveImagesAsMatlab(net, img_batch, batch_size, epoch, modelName, deepSupervision= False):
    print(" Saving images.....")
    path = '../Results/Images_MATLAB/' + modelName 
    #path = 'ResultsMatlab/ENet-Original'
    if not os.path.exists(path):
        os.makedirs(path)
        
    total = len(img_batch)
    net.eval()
    softMax = nn.Softmax().cuda()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="Saving images.....", length=30)
        image, labels, img_names = data

        MRI = to_var(image)
        Segmentation = to_var(labels)

        if deepSupervision == False:
            # No deep supervision
            segmentation_prediction = net(MRI)
        else:
            # Deep supervision
            segmentation_prediction, seg_3, seg_2, seg_1 = net(MRI)

        pred_y = softMax(segmentation_prediction)
        segmentation = getSingleImage(pred_y)
        nameT = img_names[0].split('Img/')
        nameT = nameT[1].split('.png')
        pred =  segmentation.data.cpu().numpy().reshape(320,320)
        sio.savemat(os.path.join(path, nameT[0] + '.mat'), {'pred':pred})
      
    printProgressBar(total, total, done="Images saved !")
    
def inference(net, img_batch, batch_size, epoch, deepSupervision):
    total = len(img_batch)

    Dice1 = torch.zeros(total, 2)
    Dice2 = torch.zeros(total, 2)
    Dice3 = torch.zeros(total, 2)
    
    net.eval()
    success = 0
    totalImages = 0
    img_names_ALL = []

    dice = computeDiceOneHot().cuda()
    softMax = nn.Softmax().cuda()
    timesAll = []
    start_time = time.time()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        
        if deepSupervision == False:
            segmentation_prediction = net(MRI)
        else:
            segmentation_prediction, seg3,seg2,seg1 = net(MRI)

        pred_y = softMax(segmentation_prediction)
        
        Segmentation_planes = getOneHotSegmentation(Segmentation)

        DicesN, Dices1, Dices2, Dices3= dice(pred_y, Segmentation_planes)

        Dice1[i] = Dices1.data
        Dice2[i] = Dices2.data
        Dice3[i] = Dices3.data
       

        # Save images
        # directory = 'resultBladder'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # filenameImg = os.path.join(directory, "original_image_{}_{}.png".format(epoch, i))
        # filenameMask = os.path.join(directory, "groundTruth_image_{}_{}.png".format(epoch, i))
        # filenamePred = os.path.join(directory, "Prediction_{}_{}.png".format(epoch, i))
    timesAll = time.time()-start_time
    printProgressBar(total, total, done="[Inference] Segmentation Done !")
    print(' Mean time per slice is: {} s'.format(timesAll/i))
    
    ValDice1 = DicesToDice(Dice1)
    ValDice2 = DicesToDice(Dice2)
    ValDice3 = DicesToDice(Dice3)
   
    return [ValDice1,ValDice2,ValDice3]


def inferenceVolume(net, img_batch, batch_size, epoch, deepSupervision):
    total = len(img_batch)

    Dice1 = torch.zeros(total, 2)
    Dice2 = torch.zeros(total, 2)
    Dice3 = torch.zeros(total, 2)
    
    net.eval()
    success = 0
    totalImages = 0
    img_names_ALL = []

    imagesAll = []
    dice = computeDiceOneHot().cuda()
    softMax = nn.Softmax().cuda()
    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        
        if deepSupervision == False:
            segmentation_prediction = net(MRI)
        else:
            segmentation_prediction, seg3,seg2,seg1 = net(MRI)

        pred_y = softMax(segmentation_prediction)

        predDiscrete = predToSegmentation(pred_y)
        
        pdb.set_trace()
        Segmentation_planes = getOneHotSegmentation(Segmentation)

        DicesN, Dices1, Dices2, Dices3= dice(pred_y, Segmentation_planes)

        Dice1[i] = Dices1.data
        Dice2[i] = Dices2.data
        Dice3[i] = Dices3.data
       

        # Save images
        # directory = 'resultBladder'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # filenameImg = os.path.join(directory, "original_image_{}_{}.png".format(epoch, i))
        # filenameMask = os.path.join(directory, "groundTruth_image_{}_{}.png".format(epoch, i))
        # filenamePred = os.path.join(directory, "Prediction_{}_{}.png".format(epoch, i))
    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    ValDice1 = DicesToDice(Dice1)
    ValDice2 = DicesToDice(Dice2)
    ValDice3 = DicesToDice(Dice3)
   
    return [ValDice1,ValDice2,ValDice3]
    

def inference_ResNet(netEnc, netDec, img_batch, batch_size, epoch, deepSupervision):
    total = len(img_batch)

    Dice1 = torch.zeros(total, 2)
    Dice2 = torch.zeros(total, 2)
    Dice3 = torch.zeros(total, 2)
    
    netEnc.eval()
    netDec.eval()
    success = 0
    totalImages = 0
    img_names_ALL = []

    dice = computeDiceOneHot().cuda()

    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        
        if deepSupervision == False:
            segmentation_prediction = netDec(netEnc(MRI))
        else:
            segmentation_prediction, seg3,seg2,seg1 = net(MRI)

        Segmentation_planes = getOneHotSegmentation(Segmentation)

        DicesN, Dices1, Dices2, Dices3= dice(segmentation_prediction, Segmentation_planes)

        Dice1[i] = Dices1.data
        Dice2[i] = Dices2.data
        Dice3[i] = Dices3.data
       

        # Save images
        # directory = 'resultBladder'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # filenameImg = os.path.join(directory, "original_image_{}_{}.png".format(epoch, i))
        # filenameMask = os.path.join(directory, "groundTruth_image_{}_{}.png".format(epoch, i))
        # filenamePred = os.path.join(directory, "Prediction_{}_{}.png".format(epoch, i))
    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    ValDice1 = DicesToDice(Dice1)
    ValDice2 = DicesToDice(Dice2)
    ValDice3 = DicesToDice(Dice3)
   
    return [ValDice1,ValDice2,ValDice3]
    
    
def inference_multiTask(net, img_batch, batch_size, epoch, deepSupervision):
    total = len(img_batch)

    Dice1 = torch.zeros(total, 2)
    Dice2 = torch.zeros(total, 2)
    Dice3 = torch.zeros(total, 2)
    
    net.eval()
    success = 0
    totalImages = 0
    img_names_ALL = []

    dice = computeDiceOneHot().cuda()
    voldiff = []
    xDiff = []
    yDiff = []

    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        
        if deepSupervision == False:
            segmentation_prediction, reg_output = net(MRI)
        else:
            segmentation_prediction, seg3,seg2,seg1 = net(MRI)

        Segmentation_planes = getOneHotSegmentation(Segmentation)
        
        # Regression
        feats = getValuesRegression(labels)
        feats_t = torch.from_numpy(feats).float()
        featsVar = to_var(feats_t)
        
        diff =  reg_output - featsVar 
        diff_np = diff.cpu().data.numpy()
        
        voldiff.append(diff_np[0][0])
        xDiff.append(diff_np[0][1])
        yDiff.append(diff_np[0][2])

                    
        DicesN, Dices1, Dices2, Dices3= dice(segmentation_prediction, Segmentation_planes)

        Dice1[i] = Dices1.data
        Dice2[i] = Dices2.data
        Dice3[i] = Dices3.data
       

        # Save images
        # directory = 'resultBladder'
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        # filenameImg = os.path.join(directory, "original_image_{}_{}.png".format(epoch, i))
        # filenameMask = os.path.join(directory, "groundTruth_image_{}_{}.png".format(epoch, i))
        # filenamePred = os.path.join(directory, "Prediction_{}_{}.png".format(epoch, i))
    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    ValDice1 = DicesToDice(Dice1)
    ValDice2 = DicesToDice(Dice2)
    ValDice3 = DicesToDice(Dice3)
   
    return [ValDice1,ValDice2,ValDice3, voldiff, xDiff, yDiff]
    
def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()


def resizeTensorMask(batch, scalingFactor):
    data = batch.cpu().data.numpy()
    batch_s = data.shape[0]
    numClasses = data.shape[1]
    img_size = data.shape[2]
    # TODO: Better way to define this
    resizedLabels = np.zeros((batch_s,
                              numClasses,
                              img_size / scalingFactor,
                              img_size / scalingFactor))

    for i in range(data.shape[0]):

        for l in range(numClasses):
            img = data[i, l, :, :].reshape(img_size, img_size)
            imgRes = skiTransf.resize(img, (img_size / scalingFactor, img_size / scalingFactor), preserve_range=True)
            idx0 = np.where(imgRes < 0.5)
            idx1 = np.where(imgRes >= 0.5)
            imgRes[idx0] = 0
            imgRes[idx1] = 1
            resizedLabels[i, l, :, :] = imgRes

    tensorClass = torch.from_numpy(resizedLabels).float()
    return Variable(tensorClass.cuda())


def resizeTensorMaskInSingleImage(batch, scalingFactor):
    data = batch.cpu().data.numpy()
    batch_s = data.shape[0]
    numClasses = data.shape[1]
    img_size = data.shape[2]
    # TODO: Better way to define this
    resizedLabels = np.zeros((batch_s,
                              img_size / scalingFactor,
                              img_size / scalingFactor))

    for i in range(data.shape[0]):
        img = data[i, :, :].reshape(img_size, img_size)
        imgL = np.zeros((img_size, img_size))
        idx1t = np.where(img == 1)
        imgL[idx1t] = 1
        imgRes = skiTransf.resize(imgL, (img_size / scalingFactor, img_size / scalingFactor), preserve_range=True)
        idx1 = np.where(imgRes >= 0.5)

        imgL = np.zeros((img_size, img_size))
        idx2t = np.where(img == 1)
        imgL[idx2t] = 1
        imgRes = skiTransf.resize(imgL, (img_size / scalingFactor, img_size / scalingFactor), preserve_range=True)
        idx2 = np.where(imgRes >= 0.5)

        imgL = np.zeros((img_size, img_size))
        idx3t = np.where(img == 1)
        imgL[idx3t] = 1
        imgRes = skiTransf.resize(imgL, (img_size / scalingFactor, img_size / scalingFactor), preserve_range=True)
        idx3 = np.where(imgRes >= 0.5)

        imgResized = np.zeros((img_size / scalingFactor, img_size / scalingFactor))
        imgResized[idx1] = 1
        imgResized[idx2] = 2
        imgResized[idx3] = 3

        resizedLabels[i, :, :] = imgResized

    tensorClass = torch.from_numpy(resizedLabels).long()
    return Variable(tensorClass.cuda())


# TODO : use lr_scheduler from torch.optim
def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=7):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


# TODO : use lr_scheduler from torch.optim
def adjust_learning_rate(lr_args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_args * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print(" --- Learning rate:  {}".format(lr))


if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms

    loader = transforms.Compose([transforms.ToTensor()])
    pred = to_var(getOneHotSegmentation(loader(Image.open('MICCAI_Bladder/val/GT/newR12_Lab_65.png')).unsqueeze(0)))
    GT = to_var(getOneHotSegmentation(loader(Image.open('MICCAI_Bladder/val/GT/newR12_Lab_93.png')).unsqueeze(0)))
    print(pred)

    hausdorff = Hausdorff().cuda()

    from time import time

    tic = time()
    for _ in range(50):
        x = hausdorff(pred, GT)
    toc = time()
    print(x, (toc - tic) / 50)
