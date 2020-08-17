import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import os 
from pathlib import Path
import json

def get_dot_mask(img, *args, **kwags):
    '''
    takes ndarray (possibly 1200 x 1600 x 3) as an input
    returns dot mask (1200 x 1600) as output
    '''
    # B: 180~ 256
    # R: <100
    # G: <80
    boo = (img[:,:,0] < 100) & (img[:,:,1] < 80) & (img[:,:,2] > 180) & (img[:,:,2] <= 256) 
    return boo

def construct_dots (ori_img, dot_img, radius = 4, color = 'bl', *args):
    # blue dots
    if color == 'bl':
        r_ = 80; g_ = 100; b_ = 220

    r = r_*get_dot_mask(dot_img)
    g = g_ * get_dot_mask(dot_img)
    b = b_ * get_dot_mask(dot_img)

    rgb = np.dstack((r,g,b))
    boo = to_3d(get_dot_mask(dot_img))

    tmp = ori_img.copy()
    tmp[boo] = 0
    tmp = tmp + rgb
    #Image.fromarray(tmp.astype('uint8'), 'RGB').show()
    return tmp 

def arr_to_img (img_array, name = 'name',ftype = 'png', save = False):
    #scipy.misc.toimage(img, cmin=0.0, cmax=256).save('outfile2.png')
    # convert to uint type for float data type 
    if img_array.dtype == 'float32' or img_array.dtype == 'float64':
        img_array = (img_array * 255 / np.max(img_array))
    img_pil = Image.fromarray(img_array.astype('uint8'), 'RGB')
    if save: 
        print('Saving img ' + name + ' as ' + ftype + ' format')
        img_pil.save(name+'.'+ftype)
    return img_pil

def reconstruct_img (img, color = 'blue'):
    #
    # get dot_center
    # construct dots
    # add dots to original img
    pass 

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

# ------------------------------------------------------------------
#  Modules for creating colored MNIST dataset  

def rand_crop_img (img, size = 256, change_col = True, *args):
    '''
    gets nd array greater than (256x256) as an input
    returns randomly cropped image (256x256) as an output
    channel values are uint of 0 to 255
    '''
    if img.dtype == 'float32':
        img = (img * 255 / np.max(img)).astype('uint8')
    im_pil = Image.fromarray(img.astype('uint8'), 'RGB')

    x_c = np.random.randint(0, im_pil.size[0] - size)
    y_c = np.random.randint(0, im_pil.size[1] - size)
    img_cropped = im_pil.crop((x_c, y_c, x_c + size, y_c + size)) 
    img_cropped = np.asarray(img_cropped) / 255.0

    if change_col == True:
        # change color distn 
        for j in range(3):
            img_cropped[:, :, j] = (img_cropped[:, :, j] + np.random.uniform(0, 1.1) ) / 2.0
            img_cropped = img_cropped / np.max(img_cropped)

    return img_cropped

def img_from_mask(img, mask ):
    '''
    mask : 2d array of true/false
    img : 3d array with rgb channels
    '''
    r = img[:,:,0] + 1*mask 
    g = img[:,:,1] + 1*mask 
    b = img[:,:,2] + 1*mask 
    rgb = np.dstack((r,g,b))
    rgb[rgb > 1] = 0
    return rgb

def create_circle_patch(img, x, y, r=2, color = 'blue'):
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

def xy (cl):
    x = 51 + 51*(cl%4) +  np.random.randint(low = -2, high = 2) 
    y = 63 + 63 * int(cl/4) +  np.random.randint(low = -2, high = 2) 
    if cl>=8:x += 51
    return x,y

def xy2 (cl, left, right):
    x = 51*(cl%4) + int(np.sqrt(left))
    y = 3+ 63 * int(cl/4) + int(np.sqrt(right))
    if cl>=8:x += 51
    return x, y

def create_mnist_batches(data_m, targets, image_size = 256, batch_size = 32, save = False):
  #  batch_raw = mnist_iter[0][:,0,:,:]
  #  num_lab = mnist_iter[1].cpu().detach().numpy()
    ''' takes mnist batch as an input'''
    batch_raw = data_m.cpu().detach().numpy()[:,0,:,:]
    num_lab = targets.cpu().detach().numpy()
    # create empty batch
    batch = np.zeros((batch_size, image_size, image_size, 3))
    batch_dot = np.zeros((batch_size, image_size, image_size, 3))
    coord_list = [] 
    for i in range(batch_size): 
        mn_arr = batch_raw[i]
        left = np.sum(mn_arr[:,:125])
        right = np.sum(mn_arr[:,125:])
        x, y = xy2(num_lab[i], left, right )
        # get mnist mask
        mn_msk_1 = mn_arr > 0.5
        mn_msk_2 = mn_arr < 0.5
        # create combine img of hand color + lena background 
        hand = plt.imread('hand_1.jpg')
        hand = hand[0:300, 800:1180, :]
        hand_cp = rand_crop_img(hand, change_col = False)
        lena = plt.imread('lena.png')
        lena_cp = rand_crop_img(lena)
        i1 = img_from_mask(hand_cp,mn_msk_2) + img_from_mask(lena_cp,mn_msk_1)
        # add to batch
        batch[i] = i1 
        batch_dot[i] = create_circle_patch(i1, x, y)
        coord_list.append((x,y))
    return [batch, batch_dot, coord_list]

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

def to_tensor(numpy_val):
    pass 


# create color mnist

paths = os.getcwd() # 'C:\\Users\\NormalKim\\[0_GAN]\\color_mnist2'
print('current path:' , paths)

def color_mnist_coords(paths, name = 'train', image_size = 256, batch_size = 32):
    # converts numpy -> json format 
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NpEncoder, self).default(obj)

    # mnist image loader 
    my_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,)),
        ])
    train_set = datasets.MNIST(root = 'dataset/', train = True, transform = my_transforms, download = True)
    test_set = datasets.MNIST(root = 'dataset/', train = False, transform = my_transforms, download = True)
    train_loader = DataLoader(train_set, batch_size = 32, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = 32, shuffle = True)    

    loader = train_loader if name == 'train' else test_loader
    
    # create path 
    Path("./train").mkdir(parents = True, exist_ok = True)
    Path("./train/original").mkdir(parents = True, exist_ok = True)
    Path("./train/dotted").mkdir(parents = True, exist_ok = True)
    Path("./test").mkdir(parents = True, exist_ok = True)
    Path("./test/original").mkdir(parents = True, exist_ok = True)
    Path("./test/dotted").mkdir(parents = True, exist_ok = True)    
    
    # directory 
    dir1 = os.path.join(paths, name, 'original')
    dir2 = os.path.join(paths, name, 'dotted')
    
    # initialize json
    data_dict = {} 
    data_dict['image'] = []
    cnt = 0    
    
    # loop iterator
    for idx, (data, targets) in enumerate(loader):
        new_batch = create_mnist_batches(data, targets)
        inds = targets.cpu().detach().numpy()
        for i in range(32):
            ids = 'num' + str(idx)+'_'+str(i)
            f_dir_1 = os.path.join( dir1, ids +'.jpg') # jpg format
            f_dir_2 = os.path.join( dir2, ids +'.jpg')
            arr_to_img(new_batch[0][i]).save( f_dir_1 )
            arr_to_img(new_batch[1][i]).save( f_dir_2 )
            data_dict['image'].append({'id': cnt, 'im_id': ids, 'class': inds[i], 'coord': new_batch[2][i]})
            cnt += 1
        json_file = json.dumps(data_dict, indent = 4,cls=NpEncoder) # tagging json 
        with open('color_mnist_'+name+'.json', 'w') as f:
            f.write(json_file)
        print ('Saving for batch index: ', idx, ' is complete')

