import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import argparse
import time

print(torch.__version__)
print(torch.cuda.is_available()) 
torch.cuda.empty_cache()



def get_input_args(mode):
    """
    Parses command-line arguments for train.py and predict.py based on the given mode.
    
    Args:
        mode (str): Mode for the argument parser. Accepts 'train' or 'predict'.
        
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Parse command-line arguments for training or predicting."
    )
    
    if mode == 'train':
        # Required positional argument
        parser.add_argument('data_dir', type=str,default='flowers', help='Directory of the dataset')
        
        # Optional arguments
        parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
        parser.add_argument('--arch', type=str, default='vgg13', choices=['vgg13', 'vgg16', 'resnet50'],
                            help='Model architecture (default: vgg13)')
        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
        parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units (default: 512)')
        parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
        parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    
    elif mode == 'predict':
        # Required positional arguments
        parser.add_argument('input', type=str, help='Path to the input image')
        parser.add_argument('checkpoint', type=str, help='Path to the saved model checkpoint')
        
        # Optional arguments
        parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes (default: 5)')
        parser.add_argument('--category_names', type=str, help='Path to JSON file mapping categories to names')
        parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    else:
        raise ValueError("Mode must be 'train' or 'predict'")
    
    return parser.parse_args()



class Classifier_Layer(nn.Module):
    """
    Defines a simple custom classifier layer for the network.
    
    INPUT:
    ---------
    input_size: int, size of the input layer (e.g., 25088 for VGG)
    output_size: int, size of the output layer (e.g., number of classes)
    hidden_layers: list of ints, sizes of hidden layers (e.g., [1024, 256])
    drop_p: float, dropout probability (default 0.2)
    """
    
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        super().__init__()
        
        # Initialize the layers
        self.hidden_layers = nn.ModuleList()
        
        # Add the first hidden layer (input -> first hidden layer)
        self.hidden_layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        # Add the remaining hidden layers
        for i in range(1, len(hidden_layers)):
            self.hidden_layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        
        # Add the output layer (last hidden layer -> output)
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        """ Forward pass through the network, returns the output logits """
        # Flatten the tensor
        x = torch.flatten(x, 1)

        # Pass through each hidden layer
        for layer in self.hidden_layers:
            x = F.relu(layer(x))  # Apply activation
            x = self.dropout(x)   # Apply dropout after activation

        # Pass through the output layer
        x = self.output(x)

        return F.log_softmax(x, dim=1)
    


