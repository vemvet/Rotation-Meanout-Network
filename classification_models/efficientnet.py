# -*- coding： utf-8 -*-
# -*- coding： utf-8 -*-
'''
@Time: 2022/2/16 22:41
@Author:YilanZhang
@Filename:efficientnet_RM.py
@Software:PyCharm
@Email:zhangyilan@buaa.edu.cn
'''

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision.transforms.functional import rotate
from torchvision.transforms import CenterCrop
from PIL import Image


class EfficientNet_RM(nn.Module):
    '''
    efficientnetb0_RM_5-9
    '''
    def __init__(self,theta_interval=0, pretrained=True, classes=8, dropout=0.2, device='cuda'):
        super(EfficientNet_RM, self).__init__()
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
            nn.Linear(1280, classes),
        )

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
        x = self.classifier(x)
        return x




if __name__ == '__main__':
    model = EfficientNet_RM(theta_interval=90,pretrained=True).cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    print(model(Variable(x)).size())
    for name, param in model.named_parameters():
        print(name)
    print("finish")