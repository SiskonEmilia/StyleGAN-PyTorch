# Import necessary modules
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import math

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

get_ipython().run_line_magic('matplotlib', 'inline')

image_folder_path = './dataset/'

def gain_sample(dataset, batch_size, image_size=4):
    transform = transforms.Compose([
            transforms.Resize(image_size),          # Resize to the same size
            transforms.CenterCrop(image_size),      # Crop to get square area
            transforms.RandomHorizontalFlip(),      # Increase number of samples
            transforms.ToTensor(),            
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

    dataset.transform = transform
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)

    return loader

dataset        = datasets.ImageFolder(image_folder_path)
origin_loader = gain_sample(dataset, 240, 64)
data_loader = iter(origin_loader)

for i in range(10):
    real_image, label = next(data_loader)
    torchvision.utils.save_image(real_image, f'./previews/preview{i}.png', nrow=24, padding=2, normalize=True, range=(-1,1))