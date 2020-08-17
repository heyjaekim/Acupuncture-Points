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

def gen_test_img_list(dir1='./testdata/original', dir2='./testdata/target'):
    ''' Want to create list of test-data images used for monitoring training progress
    '''
    test_imlist = []
    im_names = os.listdir(dir1)
    transform = transforms.Compose([transforms.Resize(size = 256),transforms.ToTensor()])
    for i in im_names:
        img = Image.open(dir1 + '/' + i) 
        img = img.rotate(-90)
        img = transform(img)
    # img = rescale(img, )
        test_imlist.append(img)

    target_imlist = []
    target_names = os.listdir(dir2)
    for i in target_names:
        img = Image.open(dir2 + '/' +i) 
        img = img.rotate(-90)
        img = transform(img)
        target_imlist.append(img)

    return test_imlist, target_imlist
