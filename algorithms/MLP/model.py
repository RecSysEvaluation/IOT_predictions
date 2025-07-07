# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 21:26:19 2023

@author: shefai
"""

from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import ReLU
from torch.nn import Module
from torch.optim import SGD

from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform
from torch.nn.init import xavier_uniform
from torch.utils.data import DataLoader


class MlpModel(Module):
    def __init__(self, inputs):
        super(MlpModel, self).__init__()
        
        self.h1 = Linear(inputs, 20)
        kaiming_uniform(self.h1.weight, nonlinearity='relu')
        self.act1 = ReLU()
            
        self.h2 = Linear(20, 10)
        kaiming_uniform(self.h2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        
        self.h3 = Linear(10, 1)
        xavier_uniform(self.h3.weight)
        self.act3 = Sigmoid()
        
    def forward(self, x):
        x = self.h1(x)
        x = self.act1(x)
        
        x = self.h2(x)
        x = self.act2(x)
        
        x = self.h3(x)
        x = self.act3(x)
        
        return x
            
            
            
        