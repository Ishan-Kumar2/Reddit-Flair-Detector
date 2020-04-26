import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self,n_vocab,
                 pretrained_vec,
                batch_size=16,
                embedding_dim=50,
                hidden_dim=64,
                num_layer=2,
                 dropout=0.3,
                 output_dims=5,
                 second_output=16,
                bidirectional=True):
        super(Model,self).__init__()
        self.n_vocab=n_vocab
        self.batch_size=batch_size
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.num_layer=num_layer
        self.bidirectional=bidirectional
        self.dropout=dropout
        
        self.output_dims=output_dims
        
        self.embedding=nn.Embedding(n_vocab,embedding_dim)
        self.embedding.weight.data.copy_(pretrained_vec)
        #self.embedding.weight.requires_grad=False
        
        self.second_output=nn.Linear(2,16)
        
        self.rnn=nn.LSTM(self.embedding_dim, self.hidden_dim,
                       num_layers=self.num_layer,
                       batch_first=True,bidirectional=self.bidirectional,
                        dropout=0.5)
        
       
        self.fc1=nn.Linear(hidden_dim*6+16,self.output_dims)
        
        self.dropout=nn.Dropout(dropout)
        self.logsoftmax=nn.LogSoftmax()
        
    def apply_attention(self,final_hid, all_hid):
    #final hid dim [batch,seqlen]
        final_hid=final_hid.unsqueeze(2)
    
    #Final hid_dim [batch,1,SeqLen]
        attention_scores=torch.bmm(all_hid,final_hid).squeeze(2)
        soft_attention_weights=F.softmax(attention_scores).unsqueeze(2)
        
        attention_output=torch.bmm(all_hid.permute(0,2,1),soft_attention_weights).squeeze(2)
   
        return attention_output
    
    def forward(self,data,num_data,context):
        
        embedded=self.embedding(data)
        output,(hidden_,_)=self.rnn(embedded)
        hidden_=torch.cat((hidden_[-1,:,:],hidden_[-2,:,:]),dim=1)
        
        attention_output=self.apply_attention(final_hid=hidden_,all_hid=output)
        
        concat_vec=torch.cat([hidden_,attention_output],dim=1)
        hidden=self.dropout(concat_vec)
        
        num_output=self.dropout(self.second_output(num_data))
        hidden=torch.cat([hidden,num_output],dim=1)
        
        context=self.apply_attention(final_hid=hidden_,all_hid=context)
        hidden=self.dropout(torch.cat([hidden,context],dim=1))
        
        
        output=self.fc1(hidden)
        
        return output


class fastText(nn.Module):
    def __init__(self,vocab_size,embedding_dim,
                 hidden_size,output_size,pretrained_wv):
        super(fastText,self).__init__()
        self.embedding_dim=embedding_dim
        self.pretrainedwts=pretrained_wv
        self.output_size=output_size
        self.hidden_size=hidden_size
        self.dropout=nn.Dropout(0.4)
        
        
        #Embedding Layer
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.embedding.weight.data.copy_(self.pretrainedwts)
        self.embedding.weight.requires_grad=False
        
        self.fc1=nn.Linear(self.embedding_dim,self.hidden_size)
        
        self.fc2=nn.Linear(self.hidden_size,self.output_size)
        self.softmax=nn.Softmax()

    def forward(self,x):
        embedded=self.embedding(x)
        h=self.dropout(self.fc1(embedded))
        z=self.dropout(self.fc2(h))
        return self.softmax(z,)
