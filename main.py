
import torch
from torch.autograd import Variable

from classification_models.resnet import ResNet_RM
from classification_models.inception_v3 import Inception_V3_RM
from classification_models.efficientnet import EfficientNet_RM
from retrieval_models.resnet_hash import ResNet_RM_Hash
from retrieval_models.inception_v3_hash import Inception_V3_RM_Hash
from retrieval_models.efficientnet_hash import EfficientNet_RM_Hash

if __name__ == "__main__":
    model = ResNet_RM(model_name='resnet34', theta_interval=90, pretrained=True,device='cpu')
    x = torch.randn(1, 3, 224, 224)
    print(model(Variable(x)).size())
    for name, param in model.named_parameters():
        print(name)
    print("finish")