from torch._C import device
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss
import time

device = torch.device("gpu")

train_data = torchvision.datasets.CIFAR10('./dataset', train =True, transform = torchvision.transforms.ToTensor(), download = True)

test_data = torchvision.datasets.CIFAR10('./dataset', train =False, transform = torchvision.transforms.ToTensor(), download = True)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("xun lian : {}".format(train_data_size))
print("ce shi : {}".format(test_data_size))

train_dataloader = DataLoader(train_data, batch_size = 64)
test_dataloader = DataLoader(test_data, batch_size = 64)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()

        self.modle1 = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )

    def forward(self, x):

        x = self.modle1(x)
        return x

tudui = Tudui()
#tudui.to(device)
if torch.cuda.is_available():
    tudui = tudui.cuda()

#los_fn.to(device)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr = learning_rate)

total_train_step = 0

total_test_step = 0

epoch = 10

writer = SummaryWriter("./log")
# start_time = time.time()

for i in range(epoch):
    print("--------------di {} lun------------". format(i+1))
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        #imgs = imgs.to(device)
        targets = targets.cuda()
        #targets=targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 ==0:
            # end_time = time.time()
            # print(end_time - start_time)
            print( "xun lian: {}, loss: {}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            #imgs = imgs.to(device)
            targets = targets.cuda()
            #targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accurecy = (outputs.argmax(1) == targets).sum()
            total_acc = total_acc + accurecy
    print("zheng ti :{}".format(total_test_loss))
    print("zheng que lv :{}".format(total_acc/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_acc", total_acc/test_data_size, total_test_step)
    total_test_step += 1

writer.close()


