import torch
from torch import nn
from torch.nn import Linear
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter



dataset = torchvision.datasets.CIFAR10('./dataset', train =False, transform = torchvision.transforms.ToTensor(), download = True)


dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)
        # self.sigmoid1 = Sigmoid()

        

    def forward(self, input):
        output = self.linear1(input)
        # output = self.sigmoid1(input)
        return output

tudui = Tudui()
# output = tudui(input)
# print(output)

writer = SummaryWriter('./logs')
step = 0


for data in dataloader:
    imags, target = data
    # print(imags.shape)
    # writer.add_images("input", imags, step)
    output = torch.reshape(imags,(1, 1, 1, -1))
    # output = torch.flatten(imags)
    # print(output.shape)
    # writer.add_images("output", output, step)

    output = tudui(output)
    writer.add_images("output", output, step)
    # print(output.shape)
    step+=1

writer.close()