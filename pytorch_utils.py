import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models # add models to the list
from torchvision.utils import make_grid
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")

# Creating a Dataset
root = 'folder_path'

train_data = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)

# Giving manual seed to the random number generator
torch.manual_seed(42)

# Creating a DataLoader
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)
class_names = train_data.classes

# Transforming Images
train_transform = transforms.Compose([
        transforms.RandomRotation(5),      # rotate +/- 5 degrees
        transforms.RandomHorizontalFlip(0.3),  # reverse 30% of images
        transforms.Resize(448),             # resize shortest side to 448 pixels
        transforms.CenterCrop(250),         # crop longest side to 250 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

test_transform = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(250),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

