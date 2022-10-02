# -*- coding： utf-8 -*-
import torchvision.models as models
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms.functional import rotate
from torchvision.transforms import CenterCrop
from PIL import Image
import torch.nn.functional as F

class Inception_V3_RM_Hash(nn.Module):
    '''
    inception_v3_RM_C-E
    '''

    def __init__(self, theta_interval=0, pretrained=True, classes=8, hash_bit=16,dropout=0.2, device='cuda'):
        super(Inception_V3_RM_Hash,self).__init__()
        self.inception = models.inception_v3(pretrained=pretrained)
        self.inception.aux_logits = False
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
        self.hash_layer = nn.Linear(2048, hash_bit)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3,stride=2)
        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        if self.theta_interval == 0:
            x = self.inception.Mixed_5b(x) #A
            x = self.inception.Mixed_5c(x)
            x = self.inception.Mixed_5d(x)
            x = self.inception.Mixed_6a(x) #B
            x = self.inception.Mixed_6b(x) #C
            x = self.inception.Mixed_6c(x)
            x = self.inception.Mixed_6d(x)
            x = self.inception.Mixed_6e(x)
            x = self.inception.Mixed_7a(x) #D
            x = self.inception.Mixed_7b(x) #E
            x = self.inception.Mixed_7c(x)
        else:
            x = self.inception.Mixed_5b(x)  # A
            x = self.inception.Mixed_5c(x)
            x = self.inception.Mixed_5d(x)
            x = self.inception.Mixed_6a(x)  # B
            x_size = 8
            channel = 2048
            x_rm = torch.zeros((self.rotate_number, x.size()[0], channel, x_size, x_size), device=self.device)
            for k in range(0,self.rotate_number):
                out = rotate(x,k*self.theta_interval,resample=Image.BILINEAR,expand=True)
                out = self.inception.Mixed_6b(out)  # C
                out = self.inception.Mixed_6c(out)
                out = self.inception.Mixed_6d(out)
                out = self.inception.Mixed_6e(out)
                out = self.inception.Mixed_7a(out)  # D
                out = self.inception.Mixed_7b(out)  # E
                out = self.inception.Mixed_7c(out)
                out = rotate(out, -k * self.theta_interval, resample=Image.BILINEAR, expand=True)
                x_rm[k]=CenterCrop(x_size)(out)
            x = torch.mean(x_rm,0)
            x = self.avgpool(x)
            x = torch.flatten(x,1)
            x = self.hash_layer(x)
            x_hash = self.sigmoid(x)
            x = self.classifier(x)
            return x_hash,x

if __name__ == "__main__":
    model = Inception_V3_RM_Hash(theta_interval=90,pretrained=True).cuda()
    x = torch.randn(1, 3, 299, 299).cuda()
    print(model(Variable(x)))
    # for name, param in model.named_parameters():
    #     print(name)
    print("finish")