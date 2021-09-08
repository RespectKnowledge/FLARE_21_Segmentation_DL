# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 22:53:58 2021

@author: Administrateur
"""
import os
import torch
from torch.utils.data import Dataset

##########################################################new dataset loader ######################################

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
#import keras
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import os
import torch
from torch.utils.data import Dataset
# "labels": {
#         "0": "background",
#         "1": "liver",
#         "2": "kidney",
#         "3": "spleen",
#         "4": "pancreas"
#     },
################################### datagenrator new#####################################################
from PIL import Image 
class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    #CLASSES = ['leftlung', 'rightlung', 'disease','unlabelled']
    #CLASSES = ['leftlung', 'rightlung','unlabelled']
    CLASSES = ['liver', 
               'kidney',
               'spleen',
               'pancreas',
               'unlabelled']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
            transform=None
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.transform=transform
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        #image = cv2.imread(self.images_fps[i])
        image = Image.open(self.images_fps[i])
        image=(np.asarray(image) / 255).astype('float32')
        #image=torch.FloatTensor(image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image=cv2.convertScaleAbs(image)
        image_name = self.ids[i]
        mask = cv2.imread(self.masks_fps[i], 0)
        #mask = io.imread(self.masks_fps[i])
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
            mask=mask.transpose(2,0,1) # n_class*w*H
            #mask=torch.uint8(mask)
            #mask=mask.transpose(2,0,1) # n_class*w*H
        #onehot_label=torch.FloatTensor(img_label_onehot)
        #print(onehot_label.shape)
        #mask=torch.uint8(mask)
        mask=torch.from_numpy(mask).type(torch.float)
            
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.transform:
            image=self.transform(image)
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)


#Training dataset path
imagepath='C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\flair2d_dataset\\Training\\imgs'
maskpath='C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\flair2d_dataset\\Training\\msks'    
x_train_dir=imagepath
y_train_dir=maskpath
### Validation dataset path
imagepath_valid='C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\flair2d_dataset\\validation\\imgs'
maskpath_valid='C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\flair2d_dataset\\validation\\msks'    
x_valid_dir=imagepath_valid
y_valid_dir=maskpath_valid


#data=data[:,:,0]  
# Lets look at data we have
from torchvision import transforms
# transform=transforms.Compose([transforms.ToTensor(),
#                               transforms.Normalize(mean=[0.485,0.456,0.406],
#                                                   std=[0.229,0.224,0.225])])

transform=transforms.Compose([transforms.ToTensor(),])

from torch.utils.data import DataLoader
 

#data=data[:,:,0]  
# Lets look at data we have
from torchvision import transforms
#transform=transforms.Compose([transforms.ToTensor(),
#                              transforms.Normalize(mean=[0.485,0.456,0.406],
#                                                  std=[0.229,0.224,0.225])])
                                                  
transform=transforms.Compose([transforms.ToTensor(),])                                                  

from torch.utils.data import DataLoader
dataset_train = Dataset(x_train_dir, y_train_dir, classes=['liver','kidney','spleen','pancreas'],transform=transform)
dataset_valid = Dataset(x_train_dir, y_train_dir, classes=['liver','kidney','spleen','pancreas'],transform=transform)
 
dataset=dataset_train  
len(dataset) 
#imges,masks=dataset[0]
import random
ix = random.randint(0, len(dataset))
img, mask= dataset[ix]
print(img.max())
print(img.min())
print(mask.max())
print(mask.min())
# fig, ax = plt.subplots(dpi=50)
# ax.imshow(img[0], cmap="gray")
# ax.axis('off')
# mask = torch.argmax(mask, axis=0).float().numpy()
# mask[mask == 0] = np.nan
# ax.imshow(mask, alpha=0.5)
# plt.show()

#train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# train_dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4)
# valid_dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4)

data={'train':dataset_train,
      'val':dataset_valid}
## check dataset image shape and mask
imgs, masks = next(iter(data['train']))
imgs.shape, masks.shape
#################### take the batch size and prepare dataloader ######
batch_size=48
dataloader = {
    'train': torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True, pin_memory=True),
    'val': torch.utils.data.DataLoader(data['val'], batch_size=batch_size, shuffle=False, pin_memory=True),
}
imgs, masks = next(iter(dataloader['train']))
imgs.shape, masks.shape
#%% define the proposed model
################################ define the model ResUnet moona #####################################
import torch
import torch.nn as nn
import torch.nn.functional as F


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

# generate random input (batch size, channel, height, width)
#inp=torch.rand(1,3,256,256)
#inp.shape
    
# Giving Classes & Channels
n_classes=5
n_channels=1
#
##Creating Class Instance of Model Inf_Net_UNet Class
model =ResUNet(n_channels, n_classes)
#
## Giving random input (inp) to the model
#out=model(inp)
#
#print(out.shape)
############################################### DensUent model #######################################
########### define the training and testing function ###########
import os
import numpy as np
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
# modelUdense=ResUNet(n_channels, n_classes)
# print(modelUdense)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10
lr = 3e-4
#optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lr=3e-4
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

batch_size=48
#%% define training and validation function
#second training function for optimizing the model
import pandas as pd
def IoU(pr, gt, th=0.5, eps=1e-7):
    pr = torch.sigmoid(pr) > th
    gt = gt > th
    intersection = torch.sum(gt * pr, axis=(-2,-1))
    union = torch.sum(gt, axis=(-2,-1)) + torch.sum(pr, axis=(-2,-1)) - intersection + eps
    ious = (intersection + eps) / union
    return torch.mean(ious).item()




from tqdm import tqdm
from collections import OrderedDict

##################### training function ##########
def train(dataloader, model, criterion, optimizer, epoch, scheduler=None):
    bar = tqdm(dataloader['train'])
    losses_avg, ious_avg = [], []
    train_loss, train_iou = [], []
    model.cuda()
    model.train()
    for imgs, masks in bar:
        #imgs, masks = imgs.to(device), masks.to(device)
        imgs, masks = imgs.cuda(), masks.cuda()
        optimizer.zero_grad()
        y_hat = model(imgs)
        loss = criterion(y_hat, masks)
        loss.backward()
        optimizer.step()
        ious = IoU(y_hat, masks)
        train_loss.append(loss.item())
        train_iou.append(ious)
        #bar.set_description(f"loss {np.mean(train_loss):.5f} iou {np.mean(train_iou):.5f}")
    losses_avg=np.mean(train_loss)
    ious_avg=np.mean(train_iou)
    
    log = OrderedDict([('loss', losses_avg),
                       ('iou', ious_avg),
                       ])
    return log

def validate(dataloader, model, criterion):
    bar = tqdm(dataloader['val'])
    test_loss, test_iou = [], []
    losses_avg, ious_avg = [], []
    #model.to(device)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for imgs, masks in bar:
            #imgs, masks = imgs.to(device), masks.to(device)
            imgs, masks = imgs.cuda(), masks.cuda()
            y_hat = model(imgs)
            loss = criterion(y_hat, masks)
            ious = IoU(y_hat, masks)
            test_loss.append(loss.item())
            test_iou.append(ious)
            bar.set_description(f"test_loss {np.mean(test_loss):.5f} test_iou {np.mean(test_iou):.5f}")
    losses_avg=np.mean(test_loss)
    ious_avg=np.mean(test_iou)
    log = OrderedDict([('loss', losses_avg),
                       ('iou', ious_avg),
                       ])
    
    return log
criterion = torch.nn.BCEWithLogitsLoss()
log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])
early_stop=20
epochs=10000
best_iou = 0
# save weights of trained model
#please create folder in current working directory e.g models/ResUnet_flair
# folder name should be  models/ResUnet_flair
# log file and trained weights will be save in folder name (models/ResUnet_flair)
name='ResUnet_flair'
trigger = 0
for epoch in range(epochs):
    print('Epoch [%d/%d]' %(epoch, epochs))
    # train for one epoch
    train_log = train(dataloader, model, criterion, optimizer, epoch)
    #train_log = train(train_loader, model, optimizer, epoch)
    # evaluate on validation set
    #val_log = validate(val_loader, model)
    val_log =validate(dataloader, model, criterion)
    print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'%(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

    tmp = pd.Series([epoch,lr,train_log['loss'],train_log['iou'],val_log['loss'],val_log['iou']], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

    log = log.append(tmp, ignore_index=True)
    log.to_csv('models/%s/log.csv' %name, index=False)

    trigger += 1

    if val_log['iou'] > best_iou:
        torch.save(model.state_dict(), 'models/%s/model.pth' %name)
        best_iou = val_log['iou']
        print("=> saved best model")
        trigger = 0

    # early stopping
    if not early_stop is None:
        if trigger >= early_stop:
            print("=> early stopping")
            break

    torch.cuda.empty_cache()
print("done training")

