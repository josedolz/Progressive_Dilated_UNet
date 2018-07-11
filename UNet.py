from Blocks import *
import torch.nn.init as init
import torch.nn.functional as F
import pdb
import math
from layers import *

        
class UNetG_Dilated_Progressive(nn.Module):
    def __init__(self, nin, nG, nout):
        super(UNetG_Dilated_Progressive, self).__init__()
        print('*'*50)
        print('--------- Creating Dilated Progressive UNet network... ---')
        print('*'*50)
        self.conv0 = nn.Sequential(convBatch(nin, nG,stride=1, dilation=1, padding=1),
                                   convBatch(nG, nG, stride=1, dilation=2, padding=2),
                                   convBatch(nG, nG, stride=1, dilation=4, padding=4))
                                   
        self.conv1 = nn.Sequential(convBatch(nG * 1, nG * 2, stride=2, dilation=1, padding=1),
                                   convBatch(nG * 2, nG * 2, stride=1, dilation=2, padding=2),
                                   convBatch(nG * 2, nG * 2, stride=1, dilation=4, padding=4))
                                   
        self.conv2 = nn.Sequential(convBatch(nG * 2, nG * 4, stride=2, dilation=1, padding=1),
                                   convBatch(nG * 4, nG * 4, stride=1, dilation=2, padding=2),
                                   convBatch(nG * 4, nG * 4, stride=1, dilation=4, padding=4))
                                   
        self.conv3 = nn.Sequential(convBatch(nG * 4, nG * 8, stride=2, dilation=1, padding=1),
                                   convBatch(nG * 8, nG * 8, stride=1, dilation=2, padding=2),
                                   convBatch(nG * 8, nG * 8, stride=1, dilation=4, padding=4))
                                   
        self.bridge = nn.Sequential(convBatch(nG * 8, nG * 16, stride=2, dilation=1, padding=1),
                                    residualConv(nG * 16, nG * 16),
                                    convBatch(nG * 16, nG * 16))

        self.deconv0 = upSampleConv(nG * 16, nG * 16)
        self.conv4 = nn.Sequential(convBatch(nG * 24, nG * 8),
                                   convBatch(nG * 8, nG * 8))
        self.deconv1 = upSampleConv(nG * 8, nG * 8)
        self.conv5 = nn.Sequential(convBatch(nG * 12, nG * 4),
                                   convBatch(nG * 4, nG * 4))
        self.deconv2 = upSampleConv(nG * 4, nG * 4)
        self.conv6 = nn.Sequential(convBatch(nG * 6, nG * 2),
                                   convBatch(nG * 2, nG * 2))
        self.deconv3 = upSampleConv(nG * 2, nG * 2)
        self.conv7 = nn.Sequential(convBatch(nG * 3, nG * 1),
                                   convBatch(nG * 1, nG * 1))

        self.final = nn.Conv2d(nG, nout, kernel_size=1)

    def forward(self, input):
        x0 = self.conv0(input)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
  
        bridge = self.bridge(x3)
        y = self.deconv0(bridge)
    
        y = self.deconv1(self.conv4(torch.cat((y, x3), dim=1)))
        y = self.deconv2(self.conv5(torch.cat((y, x2), dim=1)))
        y = self.deconv3(self.conv6(torch.cat((y, x1), dim=1)))
        y = self.conv7(torch.cat((y, x0), dim=1))

        return self.final(y)
        

