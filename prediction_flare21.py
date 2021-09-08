# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:53:32 2021

@author: Abdul Qayyum
"""
# Team_name aq_enib_f
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            ###self.resblock= ResBlock(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bRV1w6eCQFVVW3Q9RZeanm2bC9hjAT7d
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
# New Residule Block    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels))
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.down_sample = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.double_conv(x)
        out = self.relu(out + identity)
        return out

class ResUNet(nn.Module):
    """ Full assembly of the parts to form the complete network """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ResUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.res1= ResBlock(64,64)
        self.down1 = Down(64, 128)
        self.res2= ResBlock(128, 128)
        self.down2 = Down(128, 256)
        self.res3= ResBlock(256, 256)
        self.down3 = Down(256, 512)
        self.res4= ResBlock(512, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        res1= self.res1(x1) 
        # print("1st conv block", x1.shape)
        # print("1st res block", res1.shape)
        x2 = self.down1(x1)
        res2= self.res2(x2)
        # print("sec conv block", x2.shape)
        # print("sec res block", res2.shape)
        x3 = self.down2(x2)
        res3= self.res3(x3)
        # print("3rd conv block", x3.shape)
        # print("3rd res block", res3.shape)
        x4 = self.down3(x3)
        res4= self.res4(x4)
        # print("4 conv block", x4.shape)
        # print("4 res block", res4.shape)
        x5 = self.down4(x4)
        #print("Base down ", x5.shape)
        x = self.up1(x5, res4)
        #print("1st up block", x.shape)
        x = self.up2(x, res3)
        #print(" sec up block", x.shape)
        x = self.up3(x, res2)
        #print("3rd up block", x.shape)
        x = self.up4(x, res1)
   
        logits = self.outc(x)

        return logits
    
# Giving Classes & Channels
n_classes=5
n_channels=1
#
#
model =ResUNet(n_channels, n_classes)
import os 
import numpy as np
import cv2 
from skimage import io
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import io, exposure, img_as_uint, img_as_float
import imutils
from PIL import Image 
import albumentations as A
# load the trained model weights
# model weights should be in /workspace folder for testing and validation
path='/workspace'
pathw=os.path.join(path,'model.pth')
model.load_state_dict(torch.load(pathw))
#%% Flair predictions for flair2021 challenege
import os 
import numpy as np
import cv2 
from skimage import io
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import io, exposure, img_as_uint, img_as_float
import imutils
from PIL import Image 
import albumentations as A
import SimpleITK as sitk
# inputs folder
# output folder
path_val="/workspace/inputs/"
outputDir='/workspace/outputs/'
def hu_to_grayscale1(volume):

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return im_volume

listdir=os.listdir(path_val)
from torchvision import transforms
transform=transforms.Compose([transforms.ToTensor(),])
model.eval()
for sub in listdir:
    #print(sub)
    pathimg=os.path.join(path_val,sub)
    img_obj=nib.load(pathimg)
    img_data=nib.load(pathimg).get_fdata()
    img_data=np.swapaxes(img_data,2,0)
    img_data=hu_to_grayscale1(img_data)
    X_train = np.zeros((img_data.shape[0],img_data.shape[1], img_data.shape[2]), dtype=np.uint8)
    for slic in range(img_data.shape[0]):
        img=img_data[slic]
        img_r = imutils.resize(img, width=256)
        image=(np.asarray(img_r) / 255).astype('float32')
        image=image/image.max()
        # print(image.max())
        # print(image.min())
        image2=transform(image)
        #print(np.unique(img_t.max()))
        img_t = image2.unsqueeze(0)
        with torch.no_grad():
            model.eval()
            model.cuda()
            output = model(img_t.cuda())
            #output = torch.sigmoid(output)
            mask = torch.argmax(output[0,...], axis=0).float().cpu().numpy()
            #print(np.unique(mask))
        
        #mask_or= imutils.resize(mask, width=img.shape[0])
        mask_or = cv2.resize(mask, (img.shape[0],img.shape[0]), interpolation = cv2.INTER_AREA)
        #print(mask_or.shape)
        X_train[slic] =mask_or 
        #print(X_train.shape)
        ff=np.swapaxes(X_train,2,0)
        #res_img = sitk.GetImageFromArray(X_train)
    print(sub[0:14])
    nib.save(nib.Nifti1Image(ff, img_obj.affine), os.path.join(outputDir, sub[0:14] + '.nii.gz'))
    #sitk.WriteImage(res_img, os.path.join(outputDir, sub[0:9] + '.nii.gz'))
        