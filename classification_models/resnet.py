# -*- coding： utf-8 -*-

import torchvision.models as models
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms.functional import rotate
from torchvision.transforms import CenterCrop
from PIL import Image

class ResNet_RM(nn.Module):
    '''
    resnet18_RM_res5-8 and resnet34_RM_res8-16
    '''
    def __init__(self,model_name,theta_interval = 0,pretrained= True,classes = 8,dropout = 0.,device = 'cuda'):
        super(ResNet_RM,self).__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        if model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        self.device = device
        self.theta_interval = theta_interval
        if self.theta_interval != 0:
            self.rotate_number = int(360/self.theta_interval)
        else:
            self.rotate_number = 0

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(512, classes),
        )
    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x_size = x.size()[2] // 8
        channel = x.size()[1] * 8
        if self.theta_interval == 0:
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
        else:
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x_rm = torch.zeros((self.rotate_number,x.size()[0],channel,x_size,x_size),device = self.device)
            for k in range(0, self.rotate_number):
                # expanding
                out = rotate(x,k*self.theta_interval,resample=Image.BILINEAR,expand=True)
                out = self.model.layer3(out)
                out = self.model.layer4(out)
                # realign
                out = rotate(out,-k*self.theta_interval,resample=Image.BILINEAR,expand=True)
                x_rm[k] = CenterCrop(x_size)(out)
            x = torch.mean(x_rm, 0) # meanout operation

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = ResNet_RM(model_name='resnet34',theta_interval=90,pretrained=True).cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    print(model(Variable(x)).size())
    for name, param in model.named_parameters():
        print(name)
    print("finish")
