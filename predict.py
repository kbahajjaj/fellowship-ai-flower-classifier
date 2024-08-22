import argparse
import utilfun as uf
import modelfun as mf
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

import json
from PIL import Image


##--------------------------------------------------------##
##--------------------ARGUMENT PARCING--------------------##
##--------------------------------------------------------##

"""
Parser stuff for predict.py

* Predict flower name and probability from an image with. input single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint

Options: 
* Return top K most likely classes: python predict.py input checkpoint --top_k 3 
* Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json 
* Use GPU for inference: python predict.py input checkpoint --gpu

"""


parser = argparse.ArgumentParser(description='Parser for predict.py')

parser.add_argument('image_path', default='flowers/test/1/image_06752.jpg', action="store", type=str)
parser.add_argument('checkpoint_path', default='./checkpoint.pth', action="store", type=str)
parser.add_argument('--top_k', default=5, action="store", type=int)
parser.add_argument('--category_to_names', action="store", default='./cat_to_name.json', type=str)
parser.add_argument('--gpu', default="gpu", action="store", type=str)

args = parser.parse_args()
image_path = args.image_path
top_k = args.top_k
checkpoint_path = args.checkpoint_path
label_map = args.category_to_names
power = args.gpu


def main():
    
    ##--------------------------------------------------##
    ##--------------------LOAD MODEL--------------------##
    ##--------------------------------------------------##
    # save/load model
    new_model, optimizer = mf.load_checkpoint(checkpoint_path, power)

    ##-----------------------------------------------------##
    ##----------------------INFERENCE----------------------##
    ##-----------------------------------------------------##
    # Use the label map specified by the user
    with open(label_map, 'r') as f:
        cat_to_name = json.load(f)
    
    # return the top K probabilities, their classes and categories (flower names)
    k_probs, k_classes, k_categ, t_prob, t_flr = mf.predict(image_path, new_model, cat_to_name, top_k, power)
    print(f'Top {top_k} Probabilities:\n    {k_probs}\nTop {top_k} Classes:\n    {k_classes}\n\
            Top {top_k} Categories:\n    {k_categ}\nTop Probability & Flower: [{t_prob}  ->  {t_flr}]\n')


if __name__== "__main__":
    main()