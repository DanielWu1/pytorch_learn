from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import ToTensor

writer = SummaryWriter('logs')
img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

# ToTensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image('ToTensor', tensor_img)
print(tensor_img)

# Normalize

print (tensor_img[0][0][0])
trans_norm = transforms.Normalize([6, 3, 2], [9, 3, 5])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image('Normalize', img_norm, 1)

#Resize

print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)

img_resize=tensor_trans(img)
writer.add_image('Resize', img_resize, 0)

print(img_resize)

#compose
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2 , tensor_trans])
img_resize_2 = trans_compose(img)
writer.add_image('Reszie', img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image('RandomCrop', img_crop, i)



writer.close()
