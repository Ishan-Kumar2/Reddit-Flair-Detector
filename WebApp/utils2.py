#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 00:43:34 2020

@author: ishan
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, tqdm_notebook
from torch.utils.data import DataLoader,Dataset
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class Dataset(Dataset):
    def __init__(self,data_title,data_context,data_score):
        
        #self.vectorizer=CountVectorizer()
        
        #self.vectorizer.fit_transform(data_context+data_title)
        vocab=torch.load('./vocab')
        self.token2idx=vocab
        
        self.sequences=[]
        self.sequences_context=[]
        
        sent_brok=[]
        for word in data_title.split():
            try:
            
                sent_brok.append(self.token2idx[word])
            except KeyError:
                sent_brok.append('0')
        self.sequences.append(sent_brok)
        
        
        print(data_context.split())
        sent_brok=[]
        for word in data_context.split():
            try:
                
                sent_brok.append(self.token2idx[word])
            except KeyError:
                sent_brok.append('0')
        self.sequences_context.append(sent_brok)
        
        #self.sequences_context=[self.token2idx[word] for word in data_context]
        
        self.score=data_score
        
        
        
        self.idx2token={idx:token for token,idx in self.token2idx.items()}
        
    def __getitem__(self,i):
        return np.array(self.sequences[i],dtype=int),np.array(self.sequences_context[i],dtype=int),np.array(self.score,dtype=int)
    
    def __len__(self):
        return len(self.sequences)