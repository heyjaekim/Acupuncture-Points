# DataLoader
from Image_Process_utils import *
import json, os
import torch
from model_utils import *
from torchvision.transforms import Resize
import torchvision.transforms.functional as F
from PIL import Image
import numbers
import types
import collections
import warnings

class CoordDataSet (Dataset):
    def __init__(self, json_file, rootdir, train = True, transform = None):
        ''' current directory must include a json-file
        '''
        #super(CoordDataSet, self).__init__(rootdir, transform = transform)
        # dir : 'C:\\Users\\NormalKim\\[0_GAN]\\color_mnist2' 에 json file 존재 
        with open(json_file) as f:
            self.json_data = json.load(f)
        self.img_dir = rootdir
        self.transform = transform

    def __len__(self):
        return len(self.json_data['image'])
    
    def __getitem__(self, idx):
        img_name = self.json_data['image'][idx]['im_id']
        img_path = os.path.join(self.img_dir,img_name)
        img = Image.open(img_path + '.jpg')
        
        # output
        y_class = torch.tensor( self.json_data['image'][idx]['class'] ) 
        y_x = torch.tensor( self.json_data['image'][idx]['coord'][0] )
        y_y = torch.tensor( self.json_data['image'][idx]['coord'][1]  )
        target = [y_class, y_x, y_y ]
        
        if self.transform:
            img = self.transform(img)
            
        return img, target 


class HandDataSet (CoordDataSet):
    def __init__(self, json_file, rootdir, train = True, transform = None, imgformat = 'png', hand_flag = False):
        super(HandDataSet, self).__init__(json_file, rootdir, transform)

        self.transform = transform 
        self.hand_flag = hand_flag

        # check if there is a mismatch
        self.imgformat = imgformat
        # img files 
        os_dir = os.listdir(rootdir)
        os_saved = [ os_dir[i][:-4] for i in range(len(os_dir)) ]

        # json keys
        keys = list(self.json_data) 
        kw = self.json_data[keys[0]][0]['acup_info']
        self.kw = kw

        
        # original dataset has 'Hand' in its name 
        if 'Hand' in os_saved[0]: 
            self.hand_flag = True
            keys_modified = [ keys[i].replace(kw, 'Hand') for i in range(len(keys)) ]
            print(set(os_saved) ^ set(keys_modified))
            assert  set(os_saved) ^ set(keys_modified) == set(), 'Observed file-tag mismatch!'

        else:
            print(set(os_saved) ^ set(keys))
            assert  set(os_saved) ^ set(keys) == set(), 'Observed file-tag mismatch!'

        # keys
        self.json_keys = keys


    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        key = img_name = self.json_keys[idx]
        if self.hand_flag: 
            img_name = key.replace(self.kw, 'Hand')
        
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path + '.' + self.imgformat)

        # output
        hand_pos = self.json_data[key][0]['hand_pos']
        hand_pos = 1 if hand_pos == ('dorsal_right' or 'palmar_right') else 0

        x, y = self.json_data[key][1]['acup_coord']
        target = np.array([hand_pos, x, y])
        sample = {'image': img, 'target': target, 'label': img_name}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

# transformation


class Rescale(Resize):
    ''''''
    def __init__(self, size, interpolation = Image.BILINEAR):
        super(Rescale, self).__init__(size)
        self.size = size
        self.interpolation = interpolation 

    def __call__(self, sample):
        image, target, label = sample['image'], sample['target'], sample['label']
        # x, y
        x = int(target[1] * self.size / image.size[0])
        y = int(target[2] * self.size / image.size[0])
        # image
        image = F.resize(image, self.size, self.interpolation)

        return {'image': image, 'target':  np.array([target[0], x, y]), 'label': label}

class Tensorize(object):
    def __call__(self, sample):
        image, target, label = sample['image'], sample['target'], sample['label']
        image = F.to_tensor(image)
        return {'image': image, 
                'target': torch.from_numpy(target), 
                'label' : label}


class Rotate(object):
    pass
