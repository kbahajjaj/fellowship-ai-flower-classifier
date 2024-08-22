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
from collections import OrderedDict
import json


##--------------------------------------------------------##
##--------------------ARGUMENT PARCING--------------------##
##--------------------------------------------------------##

"""
Parser stuff for train.py

Basic usage: 
* python train.py data_directory
* Prints out training loss, validation loss, and validation accuracy as the network trains

Options: 
* Set directory to save checkpoints: python train.py data_dir --save_dir save_directory 
* Choose architecture: python train.py data_dir --arch "vgg13" 
* Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 
* Use GPU for training: python train.py data_dir --gpu

"""

parser = argparse.ArgumentParser(description='Parser for train.py')

parser.add_argument('data_dir', action="store", type=str, default="flowers")
parser.add_argument('--save_dir', action="store", type=str, default="checkpoint.pth")
parser.add_argument('--arch', action="store", type=str, default="densenet121")
parser.add_argument('--learning_rate', action="store", type=float, default=0.001)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, nargs=2, default=[512, 256])
parser.add_argument('--epochs', action="store", default=5, type=int)
parser.add_argument('--gpu', action="store", default="gpu", type=str)

args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
learning_rate = args.learning_rate
model_architecture = args.arch
hidden_units = args.hidden_units
power = args.gpu
epochs = args.epochs


print(args)


def main():
    
    # label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    ##--------------------------------------------------##
    ##--------------TRAIN & VALIDATE MODEL--------------##
    ##--------------------------------------------------##
    batch_size = 64
    trainloader, train_data = uf.load_data(data_dir, 'train', batch_size) # Load Data
    validloader, valid_data = uf.load_data(data_dir, 'valid', batch_size) # Load Data
    model = mf.build_model(model_architecture, hidden_units)                        # Build model architecture     
    # Hyperparameters
    print_every = 16          
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    # Initial values
    steps, running_loss = 0, 0
    # Use GPU if it's available
    device = torch.device("cuda" if (power == 'gpu' and torch.cuda.is_available()) else "cpu")
    model.to(device);
    start = time.time() # start tracking training time
    
    # Model Training
    for e in range(epochs):
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)    # Move tensors to default device
            optimizer.zero_grad() # initialize all gradients to zero
            log_ps = model(images)  # feedforward
            loss = criterion(log_ps, labels)    # calculate loss (error) per batch
            loss.backward()     # backpropagate
            optimizer.step()    # adjust weights & biases
            running_loss += loss.item() # training error cummulative counter

            # Model Validation every "print_every" steps 
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()    # disable dropout during validation

                with torch.no_grad(): # Turn off gradient calculation during validation
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model(images)                     # feed forward validation data output of LogSoftmax
                        loss = criterion(log_ps, labels)            # validation loss
                        valid_loss += loss.item()                   # validation loss cummulative counter
                        ps = torch.exp(log_ps)                      # output probabilities (of softmax)
                        top_p, top_class = ps.topk(1, dim=1)        # Highest probability output (prob & class index)
                        equals = top_class == labels.view(*top_class.shape)     # find matches between top prob output and label
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()   # calculate accuracy

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0
                # Enable training mode again for the next "print_every" steps
                model.train()
   
    # Training & Validation time calculation:
    seconds = time.time() - start
    m, s =  divmod(seconds, 60)
    h, m = divmod(m, 60)
    print(f'Total Training Time: {h:1.0f} hr :{m:2.0f} min :{s:2.0f} sec')                
    
    ##--------------------------------------------------##
    ##--------------------TEST MODEL--------------------##
    ##--------------------------------------------------##
    testloader, test_data = uf.load_data(data_dir, 'test', batch_size)
    start = time.time()
    test_loss, accuracy = 0, 0
    model.eval() # disable dropout during Testing

    with torch.no_grad():    # Turn off gradient calculation during Testing
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # Calculate accuracy
            log_ps = model(images)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print("Testing Accuracy: {:.3f}".format(accuracy/len(testloader)))
    model.train(); # Enable training mode again 
    
    # Testing time calculation:
    seconds = time.time() - start
    m, s =  divmod(seconds, 60)
    h, m = divmod(m, 60)
    print(f'Total Testing Time: {h:1.0f} hr :{m:2.0f} min :{s:2.0f} sec')

    ##--------------------------------------------------##
    ##--------------------SAVE MODEL--------------------##
    ##--------------------------------------------------##
    # save/load model
    mf.save_checkpoint(train_data, model, model_architecture, optimizer, learning_rate, epochs, save_dir)

if __name__ == "__main__":
    main()

