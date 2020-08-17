import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
import os 
# from datetime import date
import torch
import torch.nn as nn
import torchvision
## ---------------------------------------------------------------------
# NETWORK
# f, k, s, p
# output feature_map, kernel, stride, padding

class D_vanilla(nn.Module):
    def __init__(self, features):
        super(D_vanilla, self).__init__()
        self.features = features
        self.net = nn.Sequential()
    
    def forward(self, x):
        x = self.features(x)
        return x

class G_vanilla(nn.Module):
    def __init__(self, features):
        super(G_vanilla, self).__init__()
        self.features = features
        self.net = nn.Sequential()
    
    def forward(self, x):
        x = self.features(x)
        return x

def make_layers_vanilla(cfg, kw = 'D', channel_noise = 100):
    layers = []
    in_channels = 3
    if kw == 'D': 
        for items in cfg:
            if 'LR' in items:
                layers += [nn.LeakyReLU(0.2)]
            elif 'BN' in items: 
                layers += [nn.BatchNorm2d(in_channels)]
            else:           
                conv2d = nn.Conv2d(in_channels, out_channels = items[0], kernel_size=items[1], 
                                  stride = items[2], padding = items[3])
                layers += [conv2d]
                in_channels = items[0]      
        layers += [nn.Sigmoid()]
    if kw == 'G':
        in_channels = channel_noise
        for items in cfg:
            if 'R' in items:
                layers += [nn.ReLU()]
            elif 'BN' in items: 
                layers += [nn.BatchNorm2d(in_channels)]
            else:           
                deconv2d = nn.ConvTranspose2d(in_channels, out_channels = items[0], kernel_size=items[1], 
                                  stride = items[2], padding = items[3])
                layers += [deconv2d]
                in_channels = items[0]
        layers += [nn.Tanh() ]
    return nn.Sequential(*layers)        


cfg_256 =  {
    'D_1' : [ [64, 4, 2, 1],'LR', [64, 4, 2, 1], 'BN', 'LR', [128, 4, 2, 1], 'BN', 'LR',
             [128, 4, 2, 1],'BN', 'LR',[256, 4, 2, 1],'BN', 'LR', [512, 4, 2, 1], 
             'BN', 'LR',[1, 4, 2, 0] ] ,
    'G_1' :[ [1024, 4, 1, 0], 'BN', 'R', [512, 4, 2, 1], 'BN', 'R',  [256, 4, 2, 1], 'BN', 'R',  
            [256, 4, 2, 1], 'BN', 'R', [128, 4, 2, 1], 'BN', 'R', [64, 4, 2, 1], 
            'BN', 'R', [3, 4, 2, 1] ]
} 

def buildNet_vaniila(cfg, kw):
    if kw == 'D':
        return D_vanilla(make_layers_vanilla(cfg, kw))
    if kw == 'G':
        return G_vanilla(make_layers_vanilla(cfg, kw))