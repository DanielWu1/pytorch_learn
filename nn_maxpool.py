import torch
import torchvision
from torch import nn
from torch.nn.modules.pooling import MaxPool2d
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10('./dataset', train =False, transform = torchvision.transforms.ToTensor(), download = True)
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype = torch.float32)



dataloader = DataLoader(dataset, batch_size=64)
input = torch.reshape(input, (-1, 1, 5, 5))
print(input.shape)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

writer = SummaryWriter('./logs')

step = 0
tudui = Tudui()

for data in dataloader:
    imags, target = data
    writer.add_images("input", imags, step)
    output = tudui(imags)
    writer.add_images("output", output, step)
    step+=1




# output = tudui(input)
print(output)