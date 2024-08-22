import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision.io import read_image 
from collections import OrderedDict
import json
from PIL import Image


def load_data(data_dir, type, batch_size=64):
    if type == 'train':
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
        data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
        dataloader = torch.utils.data.DataLoader(data, batch_size, shuffle=True)
        
    elif type == 'valid':
        valid_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        data = datasets.ImageFolder(data_dir + '/valid', transform=valid_transforms)
        dataloader = torch.utils.data.DataLoader(data, batch_size)
        
    elif type == 'test':
        test_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])
        data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms) 
        dataloader = torch.utils.data.DataLoader(data, batch_size)
        
    return dataloader, data

def process_image(img):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #image = Image.open(image_path)
    #image = read_image(image_path)
    
    img_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    """
    # 1. Resize such that the shortest side is 256 pixes
    w_max, h_max = image.size

    if w_max > h_max:
        image.thumbnail((w_max, 256))
    else:
        image.thumbnail((256, h_max))
        
    # 2. Crop out center portion of image
    w, h = image.size
    w_new, h_new = 224, 224
    
    l = (w - w_new) // 2
    t = (h - h_new) // 2
    r = l + w_new
    b = t + h_new
    
    image = image.crop((l, t, r, b))
    
    # 3. Normalize color channels to 0->1 range
    np_image = np.array(image)
    np_image = np_image/255
    
    # 4. Normalize image mean and standard deviation 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    # 5. Move color channel from 3rd dimension (as in PIL) to 1st dimension for PyTorch
    np_image = np_image.transpose((2, 0, 1))

    return np_image
    """
    image = img_transforms(img)
    
    return image
