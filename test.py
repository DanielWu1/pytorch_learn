import torchvision
from PIL import Image
import torch
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, CrossEntropyLoss

image_path = "./dataset/dog.png"

image = Image.open(image_path)

print(image)
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()])


image = transform(image)

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


model = torch.load("tudui_2.pth")
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1))