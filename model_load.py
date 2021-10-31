import torch
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter, writer
#1
model = torch.load("vgg16_method1.pth")
print(model)
#2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model2 = torch.load("vgg16_method2.pth")
print(vgg16)
