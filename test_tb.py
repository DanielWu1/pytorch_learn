from re import I
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torch.utils.tensorboard.summary import image
import numpy as np

writer = SummaryWriter('logs')
image_path = 'data/train/ants_image/0013035.jpg'

img = Image.open(image_path)
img_array = np.array(img)
# print(img)
# print(img_array.shape)

writer.add_image('test', img_array,1, dataformats = 'HWC')

for i in range (100):
    writer.add_scalar('y=x',i,i)

writer.close()


# tensorboard --logdir=logs