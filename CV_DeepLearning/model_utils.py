import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch.nn as nn
from torchsummary import summary



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
        x = torch.flatten(x, 1)
        if self.block == BasicBlock: 
            y_cls = self.Linear1(x)
            y_coord = self.Linear2(x)
        if self.block == Bottleneck:  
            y_cls = self.Linear3(x)
            y_coord = self.Linear4(x)
    
        return y_cls, y_coord

