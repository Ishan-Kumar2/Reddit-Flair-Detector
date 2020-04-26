#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:22:06 2020

@author: ishan
"""

#from commons import getinput
#from inference import get_flair_name

#import torch
import requests
from Model_used import Model,fastText
from flask import Flask,request,render_template
import numpy as np
from utils2 import Dataset

flair_dict={

0:'Coronavirus',
1:'Politics',
2:'Non Political',
3:'Others-Policy/Economy, Photography, Buisness/Finance, Science/Technology, Food',
4:'Ask India'}


import torch
pretrained_wts=torch.load('./pretrained_wts')
import praw


reddit = praw.Reddit(client_id='XHHTF77rrqH7NA', 
                     client_secret='0z_mq6DJXF4VjKffFewCSXkmh8Y', 
                     user_agent='Reddit Classifier')

embedding_dim=50
hidden_dim=256
enc_dropout=0.5
Input_Vocab=pretrained_wts.shape[0]

embedding_dim=50
Input_Vocab=pretrained_wts.shape[0]


model_context=fastText(vocab_size=Input_Vocab,
        embedding_dim=embedding_dim,
        hidden_size=128,
        output_size=512,
        pretrained_wv=pretrained_wts)



model=Model(n_vocab=Input_Vocab,pretrained_vec=pretrained_wts,
            embedding_dim=embedding_dim,hidden_dim=hidden_dim,
            dropout=enc_dropout)





def url_extractor(url):
    
    req=url[40:46]
    return req

def get_data(id):
    hot_post=reddit.subreddit('India').hot(limit=10000)
    
    for post in hot_post:
        if(post.id==id):
            title=post.title
            body=post.selftext
            num=(post.num_comments,post.score)
            return title,body,num

import pandas as pd
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
    


@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        data=request.form['message']
        
        id=url_extractor(data)
        title,body,num=get_data(id)
        
        
        
        dataset=Dataset(data_title=title,data_context=body,data_score=num)
        
        
        x,x_context,x2=dataset[0]
        x=torch.tensor(x).unsqueeze(0)
            
        x_context=torch.tensor(x_context).unsqueeze(0)
            
        x2=torch.tensor(x2).unsqueeze(0)
            
        print(x2)
        print(x2.shape)
        
        
      
        ##vect=torch.tensor(data)
        ## model=Model(*args,**kwargs)
        model.load_state_dict(torch.load('./MODEL_GD'))
        #model_context=FastText(*args,**kwargs)
        
        model_context.load_state_dict(torch.load('./MODEL_CONTEXTGD'))
        
        model.eval()
        model_context.eval()
        
        context=model_context(x_context)
        preds=model(data=x,num_data=x2.float(),context=context)
        preds=preds.squeeze(0)
        prediction=torch.max(preds,0)[1]
        
        #preds=model.predict(x)
        
    return render_template('result.html',prediction=flair_dict[int(prediction)])
    


@app.route('/automated_testing',methods=['GET'])        
def automated_testing():
	#files={'upload_file':open('file.txt','rb')}
	r=requests.post('https://www.reddit.com/r/india/comments/g7ra97/india_press_freedom_rapidly_deteriorating/')
	print(r)
	

"""
@app.route('/',methods=['GET','POST'])
def hello_world():
    errors=[]
    results={}
    if request.method=='GET':
        
        return render_template('index.html',value="req_leleo")
    
    if request.method=='POST':
        url=request.data.decode('utf-8')
        #try:
            
        #r=requests.get(url)
        #print(r.text)
            
        #except:
            #print("NONON")
            #url="LKD"
            #errors.append(
                    #"Unable to ")
            
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file=request.files['file']
        print(f"File {file}")
        image=file.read()
        
        print(f"File {image}")
        #category, flower_name=get_flower_name(image)
        
        #tensor=get_tensor(image)
        
        return render_template('result.html',flair_name="HEllo",category=url)
"""
    
   


if __name__=='__main__':
    app.run(debug=True)
    
