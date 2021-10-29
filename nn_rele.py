import torch
from torch import nn
from torch.nn import ReLU, Sigmoid
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


input = torch.tensor([[1, -0.5],
                      [-1, 3]])

dataset = torchvision.datasets.CIFAR10('./dataset', train =False, transform = torchvision.transforms.ToTensor(), download = True)

output = torch.reshape(input, (-1, 1, 2, 2))
print(output.shape)

dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.rulu1 = ReLU()
        # self.sigmoid1 = Sigmoid()

        

    def forward(self, input):
        output = self.rulu1(input)
        # output = self.sigmoid1(input)
        return output

tudui = Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter('./logs')
step = 0


for data in dataloader:
    imags, target = data
    writer.add_images("input", imags, step)
    output = tudui(imags)
    writer.add_images("output", output, step)
    step+=1

writer.close()
