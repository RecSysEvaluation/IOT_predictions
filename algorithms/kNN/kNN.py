# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 15:34:57 2023

@author: shefai
"""
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class kNN:
    def __init__(self, n_neighbors = 10):
        self.n_neighbors = n_neighbors
        
             
        
    def fit(self, train, test):
        clf = KNeighborsClassifier(n_neighbors = self.n_neighbors)
        clf.fit(train.iloc[:, :-1], train.iloc[:, -1])
        self.clf = clf
        
    def predict(self, test):
        
        y_predict = self.clf.predict_proba(test.iloc[:,:-1])
        return np.round(y_predict[:, 1])
    
    def clear(self):
        self.solver = 0
        self.penalty = 0
        self.C = 0
        