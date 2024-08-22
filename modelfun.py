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
import utilfun as uf 

in_features = {"vgg16":25088,
                "densenet121":1024}

def build_model(model_architecture, hidden_units):
    # Load a pre-trained network
    model = eval('models.' + model_architecture + '(pretrained=True)')
    #model = models.densenet121(pretrained=True)
    
    # Freeze parameters of pre-trained network
    for param in model.parameters():
        param.requires_grad = False
        
    # Define an untrained classifier and replace the model's pre-trained classifer with it
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(in_features[model_architecture], hidden_units[0])),
                              ('relu', nn.ReLU()),
                              ('drop1', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(hidden_units[0], hidden_units[1])),
                              ('relu', nn.ReLU()),
                              ('drop2', nn.Dropout(0.2)),
                              ('fc3', nn.Linear(hidden_units[1], 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    # Replace DenseNet121 classifier by our untrained classifier
    model.classifier = classifier
    return model

def save_checkpoint(train_data, model, model_architecture, optimizer, learning_rate, epochs, file_path):
    # Save the checkpoint
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'arch': model_architecture,
                  'classifier': model.classifier,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'learning_rate': learning_rate,
                  'epochs': epochs}

    torch.save(checkpoint, file_path)
    print(f'Model saved to file: {file_path}')
    
    
    
    
    

def load_checkpoint(file_path, power):
    # Loading the checkpoint
    # Load a pre-trained network with saved architecture
    device = torch.device("cuda" if (power=='gpu' and torch.cuda.is_available()) else "cpu")
    checkpoint = torch.load(file_path, map_location=torch.device(device))
    new_model = eval('models.' + checkpoint['arch'] + '(pretrained=True)')

    # Freeze parameters of pre-trained network
    for param in new_model.parameters():
        param.requires_grad = False

    # Load new model attributes from checkpoint
    new_model.classifier = checkpoint['classifier']         # Replace densenet121 classifier structure
    new_model.load_state_dict(checkpoint['state_dict'])     # Load weights & biases
    new_model.class_to_idx = checkpoint['class_to_idx']     # Load classes to indices mapping
    optimizer = optim.Adam(new_model.classifier.parameters(), checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])         # Load optimizer state

    return new_model, optimizer

def predict(image_path, model, cat_to_name, top_k, power):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if (power=='gpu' and torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()
    img = Image.open(image_path)
    #img = read_image(image_path) 
    
    tensor_img = uf.process_image(img)     # image -> numpy array
    #tensor_img = torch.from_numpy(np_img).type(torch.FloatTensor) # numpy array -> torch tensor
    tensor_img.unsqueeze_(0)   # adding 1 dimension of 1 to the beginning of the tensor for the batch size
    tensor_img = tensor_img.to(device)
    
    with torch.no_grad():
        log_ps = model(tensor_img)
        ps = torch.exp(log_ps)
        top_probs, top_indices = ps.topk(top_k, dim=1)  # Top k predictions
        top_probs = top_probs.tolist()[0]           # tensors -> lists
        top_indices = top_indices.tolist()[0]       # tensors -> lists

        # Inverting class_to_idx by generating idx_to_class using dictionary comprehensions
        idx_to_class = {value:key for (key, value) in model.class_to_idx.items()}
        top_classes, top_categ, top_dic = [], [], dict() # Mapping top indices -> top_classes & top_categ (flower names)

        # nestested dictionary to get name of flower with top probability
        for count, idx in enumerate(top_indices):
            top_classes.append(idx_to_class[idx])
            top_categ.append(cat_to_name[idx_to_class[idx]])
            top_dic[top_probs[count]] = {'class': top_classes[-1], 'categ': top_categ[-1]}
            
        top_prob = max(top_probs)
        top_flower = top_dic[top_prob]['categ']
        
    model.train();
    return top_probs, top_classes, top_categ, top_prob, top_flower 