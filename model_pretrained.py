import torch
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter, writer

vgg16_false = torchvision.models.vgg16(pretrained = False)
vgg16_true = torchvision.models.vgg16(pretrained = True)

print(vgg16_true)
# print(vgg16_false)
dataset = torchvision.datasets.CIFAR10('./dataset', train =False, transform = torchvision.transforms.ToTensor(), download = True)
dataloader = DataLoader(dataset, batch_size=64)


vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)