# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 21:26:19 2023

@author: shefai
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import random_split

class DataClass(Dataset):
    def __init__(self, df):
        self.df = df
        self.X = np.array(df.iloc[:,:-1])
        self.y = np.array(df.iloc[:,-1])
        
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        
        self.y = self.y.reshape(len(self.y), 1)
        
        self.length = len(df)
    def __getitem__(self, index):
        return [self.X[index], self.y[index]]
    
    def __len__(self):
        return self.length
    
    def getdata(self):
        
        
        return random_split(self, [self.length])
        
        
