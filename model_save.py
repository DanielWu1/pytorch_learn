import torch
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter, writer
vgg16 = torchvision.models.vgg16(pretrained= False)
#1
torch.save(vgg16, "vgg16_method1.pth")
#2
torch.save(vgg16.state_dict(), "vgg16_mehod2.pth")


