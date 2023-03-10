import imp
from this import d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import os
from IPython.display import display, Image

# display image
with Image.open("filepath") as im:
    display(im)

# second way to display image for images in a folder
display(Image(filename=f"{os.path.join(base_cats_dir, os.listdir(base_cats_dir)[0])}"))

# convert image to numpy array
def convert_to_array(image):
    image = Image.open(image).convert("RGB")
    image_array = np.array(image)
    return image_array

# get new shape of the image
def get_new_shape(image, layer):
    n = image[0][0].shape[0]
    f = layer.kernel_size[0]
    s = layer.stride[0]
    p = layer.padding[0]
    
    return int(((n+(2*p)-f)/s)+1)

# create a DataFrame of image sizes (width x height)
path = "filepath"
img_names = []

for folder, subfolders, filenames in os.walk(path):
    for img in filenames:
        img_names.append(folder+'/'+img)

img_sizes = []
rejected = []

for item in img_names:
    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)
    except:
        rejected.append(item)

image_size_df = pd.DataFrame(img_sizes)

# get rgb values of each pixel
def get_rgb_values(image_path):
    dog = Image.open(image_path)
    r, g, b = dog.getpixel((0, 0)) # pixel at (0,0)
    return r, g, b # returns values of r, g, b between 0 and 255

# transforming images
cat = Image.open('filepath')
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.CenterCrop(150), # If size is an integer instead of sequence like (h, w), a square crop of (size, size) is made.
    transforms.RandomRotation(30), # If degrees is a number instead of sequence like (min, max), the range of degrees will be (-degrees, +degrees).
    transforms.RandomHorizontalFlip(p=1), # Horizontally flip the given PIL image randomly with a given probability
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225]) 
])
im = transform(cat)

# basic convolution layer module
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)  # 3 channel, 6 filters, 3x3 kernel, stride 1
        self.conv2 = nn.Conv2d(6, 16, 3, 1) # 6 channel, 16 filters, 3x3 kernel, stride 1
        self.fc1 = nn.Linear(6*6*16, 120)   # flattening the output of conv2d
        self.fc2 = nn.Linear(120,84) # fully connected layer
        self.fc3 = nn.Linear(84, 10) # fully connected layer

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 6*6*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1) #it is for multiclass classification

# save model
torch.save(model.state_dict(), 'filename.pt')

# download pretrained model from torchvision.models
AlexNetmodel = models.alexnet(pretrained=True)
DenseNetmodel = models.DenseNet(pretrained=True)
VGGmodel = models.vgg16(pretrained=True)

# Freeze feature parameters
for param in AlexNetmodel.parameters():
    param.requires_grad = False
