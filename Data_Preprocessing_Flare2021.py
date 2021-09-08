# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 08:47:11 2021

@author: Abdul Qayyum
"""


#%% Flair challenege 2021 dataset prepartion
###################### training images and masks ####################
import os 
import numpy as np
import cv2 
from skimage import io
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import io, exposure, img_as_uint, img_as_float
import imutils
################################### training datapath ###################
path="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\"

trainpath=os.path.join(path,"TrainingImg-002")
pathlist=os.listdir(trainpath)
############################### masks path ######################
maskpath="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\masks"

########### save training images and masks paths ########################
save_path="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\flair2d_dataset\\Training\\imgs"
save_mask="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\flair2d_dataset\\Training\\msks"
############ split dataset into training and testing the model ##########
import random
random.seed(0)
def Trian_val(data_list,test_size=0.15):
    n=len(data_list)
    m=int(n*test_size)
    test_item=random.sample(data_list,m)
    train_item=list(set(data_list)-set(test_item))
    return train_item,test_item
tr_list,test_list=Trian_val(pathlist,test_size=0.15)

DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512
DEFAULT_OVERLAY_ALPHA = 0.3
DEFAULT_PLANE = "axial"


def hu_to_grayscale(volume, hu_min, hu_max):
    # Clip at max and min values if specified
    if hu_min is not None or hu_max is not None:
        volume = np.clip(volume, hu_min, hu_max)

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return np.stack((im_volume, im_volume, im_volume), axis=-1)

def hu_to_grayscale1(volume):

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return im_volume
# import pandas as pd 
# df = pd.DataFrame(tr_list)
# df
# df.info()
# df.to_csv('training_data.csv', index=False)  

for sub in tr_list:
    pathimg=os.path.join(trainpath,sub)
    pathseg=os.path.join(maskpath,sub)
    print(pathseg)
    msk_data=nib.load(pathseg).get_fdata()
    msk_data=np.swapaxes(msk_data,2,0)
    
    img_data=nib.load(pathimg).get_fdata()
    img_data=np.swapaxes(img_data,2,0)
    # hu_max=img_data.max()
    # hu_min=img_data.min()
    img_data=hu_to_grayscale1(img_data)
    for slic in range(img_data.shape[0]):
        img=img_data[slic]
        img = imutils.resize(img, width=256)
        #img = exposure.rescale_intensity(img, out_range='float')
        #img = img_as_uint8(img)
        # x=img
        # x_min, x_max = x.min(), x.max()
        # x = (x - x_min) / (x_max-x_min)
        # img=x.astype(np.float)
        msk=msk_data[slic]
        msk= imutils.resize(msk, width=256)
        msk=msk.astype(np.uint8)
        io.imsave(os.path.join(save_path,sub[0:9]+"_"+str(slic)+'.png'),img)
        io.imsave(os.path.join(save_mask,sub[0:9]+"_"+str(slic)+'.png'),msk)

####################### validation 2d images and masks prepartion code ###########
import os 
import numpy as np
import cv2 
from skimage import io
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import io, exposure, img_as_uint, img_as_float
import imutils
path="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\"

trainpath=os.path.join(path,"TrainingImg-002")
pathlist=os.listdir(trainpath)

# maskpath="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\masks"
# save_path="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\flair2d_dataset\\Training\\imgs"
# save_mask="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\flair2d_dataset\\Training\\msks"
# ############ split dataset into training and testing the model ##########
import random
random.seed(0)
def Trian_val(data_list,test_size=0.15):
    n=len(data_list)
    m=int(n*test_size)
    test_item=random.sample(data_list,m)
    train_item=list(set(data_list)-set(test_item))
    return train_item,test_item
tr_list,test_list=Trian_val(pathlist,test_size=0.15)
#print(np.sort(tr_list))
#print(np.sort(test_list))
########################## set validation images and masks path ###############
save_patht="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\flair2d_dataset\\validation\\imgs" 
save_maskt="C:\\Users\\Administrateur\\Desktop\\micca2021\\MICCAI2021\\FLARE2021-main\\FLARE2021-main\\flair2d_dataset\\validation\\msks"       
def hu_to_grayscale1(volume):

    # Scale to values between 0 and 1
    mxval = np.max(volume)
    mnval = np.min(volume)
    im_volume = (volume - mnval)/max(mxval - mnval, 1e-3)

    # Return values scaled to 0-255 range, but *not cast to uint8*
    # Repeat three times to make compatible with color overlay
    im_volume = 255*im_volume
    return im_volume
######################## testing or validation images and masks #####################        
for sub in test_list:
    pathimg=os.path.join(trainpath,sub)
    pathseg=os.path.join(maskpath,sub)
    print(pathseg)
    msk_data=nib.load(pathseg).get_fdata()
    msk_data=np.swapaxes(msk_data,2,0)
    
    img_data=nib.load(pathimg).get_fdata()
    img_data=np.swapaxes(img_data,2,0)
    img_data=hu_to_grayscale1(img_data)
    for slic in range(img_data.shape[0]):
        img=img_data[slic]
        img = imutils.resize(img, width=256)
        #img = exposure.rescale_intensity(img, out_range='float')
        #img = img_as_uint8(img)
        # x=img
        # x_min, x_max = x.min(), x.max()
        # x = (x - x_min) / (x_max-x_min)
        # img=x.astype(np.float)
        msk=msk_data[slic]
        msk= imutils.resize(msk, width=256)
        msk=msk.astype(np.uint8)
        io.imsave(os.path.join(save_patht,sub[0:9]+"_"+str(slic)+'.png'),img)
        io.imsave(os.path.join(save_maskt,sub[0:9]+"_"+str(slic)+'.png'),msk)
        
