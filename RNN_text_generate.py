import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score

from nltk.stem import WordNetLemmatizer # lemmatizier
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')
from collections import Counter



import time
device=torch.device('cuda ' if torch.cuda.is_available() else 'cpu')
         

from tqdm import tqdm
tqdm.pandas()


import warnings
warnings.filterwarnings('ignore')

import string
special = string.punctuation
from wordcloud import WordCloud



def remove_punctuations(text):
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    
    for p in punctuations:
        text = text.replace(p, f' {p} ')

    text = text.replace('...', ' ... ')
    
    if '...' not in text:
        text = text.replace('..', ' ... ')
    
    return text
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)


def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)


def clean(text):
    """Removes links and non-ASCII characters"""    
    text = ''.join([x for x in text if x in string.printable])
    # Removing URLs
    text = re.sub(r"http\S+", "", text)
    return text

def preprocessing(word):
   # replace digits with no space
     word = re.sub(r"\d", '', word)
   # Replace all runs of whitespaces with no space
     word = re.sub(r"\s+", '', word)
     return word




lines=[]
with open('alice.txt','r') as f:
        lines = f.readlines()
a=[line.lower() for line in lines]

words=[]
for sentence in a :
    if sentence!='\n':
     for word in word_tokenize(sentence):
        words.append(preprocessing(clean(remove_punctuations(preprocessing(remove_punct(word).strip())))))

word_list=[]
for word in words :
    if word!='':
        word_list.append(word)



corpus=Counter(word_list)     
corpus_=sorted(corpus,key=corpus.get,reverse=True)
one_hot_dict={w:i+1 for i,w in enumerate(corpus_)}

text_words=' '.join(word_list)


text=[]
for i,s in enumerate(word_list):
  text.append(one_hot_dict[s])
     
  
text[:300] #text list preview




#model hyper parameters
batch_size=32
timestep=30 #each time steps occuring 30 words len
vocab_size=len(one_hot_dict)+1 #extra 1 for padding
embed_size=128 #Input features to the LSTM
hidden_size=512  #Number of LSTM units

rep_tensor=torch.LongTensor(np.array(text).astype(int)) #text list converting to  tensor 
num_batch=rep_tensor.shape[0]//batch_size #net number of batches
rep_tensor=rep_tensor[:num_batch*batch_size] #rep tensor 
rep_tensor=rep_tensor.view(32,-1) 
num_batches=rep_tensor.shape[1]//timestep



class textGenerator(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size):
        super(textGenerator,self).__init__()

        self.embed=nn.Embedding(vocab_size,embed_size)

        self.fc=nn.Linear(hidden_size,vocab_size)

        self.lstm=nn.LSTM(input_size=embed_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True)
        
        self.drop=nn.Dropout(0.3)

        self.relu=nn.ReLU()

    
    def forward(self,x):
        input=x.clone()
        
        # Perform Word Embedding 
        x=self.embed(x)
        #x = x.view(batch_size,timesteps,embed_size)
        x,_=self.lstm(x)
        # (batch_size*timesteps, hidden_size)
        x=x.contiguous().view(-1, hidden_size)
        x=self.drop(x)
        out=self.fc(x)
        #Decode hidden states of all time steps
        out=out.view(input.shape[0],input.shape[1],vocab_size)
        
        return out[:,-1]
    
    
model=textGenerator(vocab_size,embed_size,hidden_size)
optimizer=torch.optim.Adam(model.parameters())
criterion=nn.CrossEntropyLoss()

train_loss=[]
train_l=0

model.train()

for epoch in range(10):
    t0 = time.time()

    for i in range(0 ,rep_tensor.shape[1]-timestep):
    
        inputs=rep_tensor[:,i:i+timestep]
        labels=rep_tensor[:,i+timestep]

        outputs=model(inputs)
        loss = criterion(outputs, labels.reshape(-1))
        
        train_l+=loss
        
        model.zero_grad()
        loss.backward()
        optimizer.step()
        acc=accuracy_score(torch.argmax(outputs,dim=1),labels)
        
    print('epoch {:} acc {:},  seconds {:.2f}'.format(epoch,acc,(time.time() - t0)/60))
        
    
    
    
#torch.save(model.state_dict(), '/content/drive/MyDrive/textGen.pt') #save to model 
     

#model.load_state_dict(torch.load('/content/drive/MyDrive/textGen.pt'))#load to model 
   
    
    
text_n=np.concatenate(np.array(lines[-10:],dtype=object))
for i,s in  enumerate(text_n):
    text_n[i]=one_hot_dict[preprocessing(s)]

text_n



model.eval()
with torch.no_grad():

  with open('/content/drive/MyDrive/result.txt','w'):

    input=torch.tensor(text_n.astype('int').reshape(1,-1))
    for i in range(1000):

        output=model(input)

        output_item=torch.argmax(output,dim=1)

        output=torch.cat((input.reshape(1,-1),output_item.reshape(1,-1) ),1).reshape(1,-1)

        input=output.clone()

    listt=[]
    for i in output[0]:
      listt.append(word_list[i-1]+' ')
    end=' '.join(listt)

    
    
    
    

    
    
    
    
    