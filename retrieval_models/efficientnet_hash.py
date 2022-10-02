# -*- coding： utf-8 -*-


import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision.transforms.functional import rotate
from torchvision.transforms import CenterCrop
from PIL import Image


class EfficientNet_RM_Hash(nn.Module):
    '''
    efficientnetb0_RM_5-9
    '''
    def __init__(self,theta_interval=0, pretrained=True, classes=8, hash_bit=16,dropout=0., device='cuda'):
        super(EfficientNet_RM_Hash, self).__init__()
        self.efficentnet = models.efficientnet_b0(pretrained=pretrained)
        self.device = device
        self.theta_interval = theta_interval
        if self.theta_interval != 0:
            self.rotate_number = int(360 / self.theta_interval)
        else:
            self.rotate_number = 0

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hash_bit, classes),
        )
        self.hash_layer = nn.Linear(1280,hash_bit)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        if self.theta_interval ==0:
            x = self.efficentnet.features(x)
        else:
            x_size = 7
            channel = 1280
            x = self.efficentnet.features[0](x) #224*224*32
            x = self.efficentnet.features[1](x) #112*112*16
            x = self.efficentnet.features[2](x) #112*112*24
            x = self.efficentnet.features[3](x)
            x = self.efficentnet.features[4](x)
            x = self.efficentnet.features[5](x)
            x = self.efficentnet.features[6](x)
            x_rm = torch.zeros((self.rotate_number, channel, 1280,
                                x_size, x_size), device=self.device)
            for k in range(0,self.rotate_number):
                out = rotate(x,k*self.theta_interval,resample=Image.BILINEAR,expand=True)

                out = self.efficentnet.features[7](out)
                out = self.efficentnet.features[8](out)
                out = rotate(out, -k * self.theta_interval, resample=Image.BILINEAR, expand=True)
                x_rm[k]=CenterCrop(x_size)(out)
            x = torch.mean(x_rm, 0)  # meanout
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.hash_layer(x)
        x_hash = self.sigmoid(x)
        x = self.classifier(x)
        return x_hash,x




if __name__ == '__main__':
    model = EfficientNet_RM_Hash(theta_interval=90,pretrained=True).cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    print(model(Variable(x)))
    # for name, param in model.named_parameters():
    #     print(name)
    # print("finish")
