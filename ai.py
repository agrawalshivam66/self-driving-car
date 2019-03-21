# AI for Self Driving Car

# Importing the libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#creating the architecture of the neural network

class Network(nn.module): #inheriting nn.module parent class
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fcl = nn.Linear(input_size, 30)#full connection betwwen input layer and hidden layer
        self.fc2 = nn.Linear(30, nb_action)#full connection betwwen output layer and hidden layer
        
        
