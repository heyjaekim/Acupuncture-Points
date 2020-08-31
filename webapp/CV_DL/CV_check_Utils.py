from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torch import flatten
import torchvision.models as models
import torch.nn as nn
import cv2

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


class MyResNet(ResNet):
    '''
    basic setting : ResNet-18 with Basic Block
    could use different blocks (i.e. Bottleneck architecture) to create a larger network 
    after feature extraction, FC has two outputs 1. cls, 2. coordinates
    ''' 
    def __init__(self, block = BasicBlock, cfg = [2,2,2,2]):
        super(MyResNet, self).__init__(block, cfg)
        self.Linear1 = nn.Linear(512,1)
        self.Linear2 = nn.Linear(512,2)
        
        self.Linear3 = nn.Linear(512*4,1)
        self.Linear4 = nn.Linear(512*4,2)
        self.block = block
        
    def forward(self, x):
        # change forward here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = flatten(x, 1)
        if self.block == BasicBlock:
            y_cls = self.Linear1(x)
            y_coord = self.Linear2(x)
        if self.block == Bottleneck:  
            y_cls = self.Linear3(x)
            y_coord = self.Linear4(x)
    
        return y_cls, y_coord


def create_model(model_name = 'resnet34'):
    if 'resnet' in model_name: 
        block = model_cfgs[model_name][0]
        cfg = model_cfgs[model_name][1]
        return MyResNet(block =block, cfg = cfg )
    return None    


model_cfgs = {
    'resnet18': [BasicBlock, [2,2,2,2]],
    'resnet34': [BasicBlock, [3,4,6,3]],
    'resnet50': [Bottleneck, [3,4,6,3]],
    'resnet101': [Bottleneck, [3,4,23,3]],
    'resnet152': [Bottleneck, [3,8,36,3]]
}

class DL_Prediction(object):
    def __getitem__(self, img, coord):
        return img, coord


#############################################################
# Ensemble Model

class MyEnsemble(nn.Module):
    def __init__(self, *models):
        super(MyEnsemble, self).__init__()
        self.models = models
    
    def forward(self, *inputs):
        result = [ model(input_val) for model, input_val in zip(self.models, inputs) ]
        return result