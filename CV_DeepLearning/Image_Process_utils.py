import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
#import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import os 
from pathlib import Path
import json
import cv2

def arr_to_img (img_array, name = 'name',ftype = 'png', save = False):
    # convert to uint type from float data type 
    if img_array.dtype == 'float32' or img_array.dtype == 'float64':
        img_array = (img_array * 255 / np.max(img_array))
    img_pil = Image.fromarray(img_array.astype('uint8'), 'RGB')
    if save: 
        print('Saving img ' + name + ' as ' + ftype + ' format')
        img_pil.save(name+'.'+ftype)
    return img_pil


def to_3d (img):
    img = np.expand_dims(img, axis = 2)
    return np.concatenate((img,img,img), axis = 2)

def get_img_distn (img, kw='orange'):
    '''
    W x H x C ; C: RGB
    '''
    if kw == 'orange':
        _ = plt.hist(img.ravel(), bins = 256, color = 'orange', )
    
    if kw == 'red':
        _ = plt.hist(img[:, :, 0].ravel(), bins = 256, color = 'red', alpha = 0.5)

    if kw == 'green':
        _ = plt.hist(img[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.5)
    
    if kw == 'blue':
        _ = plt.hist(img[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.5)

    # _ = plt.xlabel('Intensity Value')
    # _ = plt.ylabel('Count')
    # _ = plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
    plt.show()

def create_circle_patch(img, x, y, color, r=1):
    '''
    input : 0~1 float32 ndarray
    output : 0~255 uint8 ndarray with dots
    '''
    img = (img * 255 / np.max(img)).astype('uint8')
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    draw = ImageDraw.Draw(img)
    x = 2 if x < 0 else x
    y = 2 if y < 0 else y

    leftUpPoint = (x-r, y-r)
    rightDownPoint = (x+r, y+r)

    twoPointList = [leftUpPoint, rightDownPoint]
    if color == 'blue':
        fill = (0,0,255,0)
    if color == 'red':
        fill = (255,0,0,0)
    draw.ellipse(twoPointList, fill=fill)
    img = np.array(img)
    img = (img  / np.max(img))
    return img

def display_32_batches(batch, batch_size = 32):
    plt.figure(figsize=(40,20))
    for i in range(batch_size):
        plt.subplot(4, batch_size // 4, i+1)
        plt.imshow(batch[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def to_numpy(tensor_val):
    return tensor_val.cpu().detach().numpy()


#####################################################################
def clear_background(img):
    # img = cv2.imread('test2\Hand_0000002.png')

    # convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_range = (0,0,100)
    upper_range = (358,45,255)
    mask = cv2.inRange(hsv, lower_range, upper_range)
    # invert mask
    mask = 255 - mask
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    # apply morphology closing and opening to mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result = img.copy()
    result[mask==0] = (255,255,255)
    
    return result


def gen_test_img_list(dir1='./testdata/original'):
    ''' Want to create list of test-data images used for monitoring training progress
    '''
    test_imlist = []
    target_imlist = []
    im_names = os.listdir(dir1)
    transform = transforms.Compose([transforms.Resize(size = 256),transforms.ToTensor()])
    for i in im_names:
        img1 = img2 = cv2.imread(dir1 + '/' + i)
        img1 = clear_background(img1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        im_pil1 = Image.fromarray(img1)
        img1 = transform(im_pil1)
        test_imlist.append(img1)

        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        im_pil2 = Image.fromarray(img2)
        img2 = transform(im_pil2)
        target_imlist.append(img2)

    return test_imlist, target_imlist


