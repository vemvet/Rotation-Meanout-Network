# Rotation-Meanout-Network

## Introduction
The implementation of "A Rotation Meanout Network with Invariance for Dermoscopy Image Classification and Retrieval" 

## Enviroments

- Windows/Linux both support
- python 3.8
- PyTorch 1.9.0
- torchvision

## Model Usage
### Classification Task
```python
import torch
from classification_models.resnet import ResNet_RM

model = ResNet_RM(
        model_name='resnet18',
        thete_interval=90,
        pretrained=True,
        classes = 8, #ISIC2019
        device='cpu'
)

img = torch.randn(1, 3, 224, 224)
preds = model(img) #(1,8)
```

### Retrieval Task
```python
import torch
from retrieval_models.resnet_hash import ResNet_RM_Hash

model = ResNet_RM_Hash(
        model_name='resnet18',
        thete_interval=90,
        pretrained=True,
        classes = 8, #ISIC2019
        hash_bit = 16,
        device='cpu'
)

img = torch.randn(1, 3, 224, 224)
hashcode, preds = model(img)
```
## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details
