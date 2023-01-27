import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as  plt
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score

from nltk.stem import WordNetLemmatizer # lemmatizier
from nltk.tokenize import word_tokenize
import os
import nltk
nltk.download('punkt')

from collections import Counter

import time
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
         
from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings('ignore')

import string
special = string.punctuation
from wordcloud import WordCloud



def remove_punct(text):
    punctuations = '!"#$%&\()*+-/<=>?@[\\]^_`{|}~'
    table=str.maketrans('','',punctuations)
    return text.translate(table)



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
            if len((preprocessing(remove_punct(word)).strip()))>0:
                words.append(preprocessing(remove_punct(word)).strip())

word_list=[]
for word in words :
    if word!='':
        word_list.append(word)

#-----------------------------------------------------------------------------------------

corpus=Counter(word_list) ;
corpus_=sorted(corpus,key=corpus.get,reverse=True)
corpus_=dict(list(filter(lambda x:  x[1]>2  ,dict(corpus).items())))
one_hot_dict={w:i+1 for i,w in enumerate(list(corpus_))} #.keys()
text_words=' '.join(word_list)


text=[]
for i,s in tqdm(enumerate(word_list)):
    try : 
        text.append(one_hot_dict[s])
    except :
        pass
     
print(f'Unique words count :{len(one_hot_dict)}')
#-----------------------------------------------------------------------------------------


class textGenerator(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,num_layer,prob):
        super(textGenerator,self).__init__()
        
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.prob=prob
        self.num_layers=num_layer
        self.hidden_size=hidden_size
        self.fc=nn.Linear(self.hidden_size,self.vocab_size)
        self.fc2=nn.Linear(128,self.vocab_size)
 
        self.lstm=nn.LSTM(input_size=self.embed_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)
        
        self.drop=nn.Dropout(self.prob)
        self.relu=nn.ReLU()

    
    def forward(self,h,x):
        
        
        
        input_=x.clone()
        
        batch_sz=input_.shape[0]
        input_=input_.view(batch_sz,-1)
        
        # Perform Word Embedding 
        x=self.embed(x)
        #x = x.view(batch_size,timesteps,embed_size)
        x,hiddens=self.lstm(x,h)
        # (batch_size*timesteps, hidden_size)
        x=x.contiguous().view(-1, self.hidden_size)
        x=self.relu(x)
        x=self.drop(x)
        out=self.fc(x)
#         x=self.drop(x)
#         x=self.relu(x)
#         out=self.fc2(x)
        #Decode hidden states of all time steps
        out=out.view(input_.shape[0],input_.shape[1],self.vocab_size)
        
        return out[:,-1]
    
    
    def init_hidden(self,batch_size):
    
        h0=torch.zeros((self.num_layers,batch_size,self.hidden_size)).to(device)
        c0=torch.zeros((self.num_layers,batch_size,self.hidden_size)).to(device)
        hidden=(h0,c0)
        return hidden


#base Train step

vocab_size=len(one_hot_dict)+1
batch_size=128
timestep=20#each time steps occuring 30 words len
embed_size=200 #Input features to the LSTM
hidden_size=128  #Number of LSTM units
num_layer=1
prob=0.2

model = textGenerator(vocab_size,embed_size,hidden_size,num_layer,prob).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
criterion=nn.CrossEntropyLoss()

#     if checkpoint_dir:
#             model_state, optimizer_state = torch.load(
#                 os.path.join(checkpoint_dir, "checkpoint"))
#             model.load_state_dict(model_state)
#             optimizer.load_state_dict(optimizer_state)

def create_rep_data(text,batch_size):
    rep_tensor=torch.LongTensor(np.array(text).astype(int)) #text list converting to  tensor 
    num_batch=rep_tensor.shape[0]//batch_size #net number of batches
    rep_tensor=rep_tensor[:num_batch*batch_size] #rep tensor 
    rep_tensor=rep_tensor.view(batch_size,-1) 
    num_batches=rep_tensor.shape[1]//timestep

    return rep_tensor,num_batches



rep_tensor,_ = create_rep_data(text,batch_size)


data_list=[]
label_list=[]
for  k,i in tqdm(enumerate(range(0 ,rep_tensor.shape[1]-timestep))):

    inputs=rep_tensor[:,i:i+timestep]
    labels=rep_tensor[:,i+timestep]
    data_list.append(inputs)
    label_list.append(labels)

data=np.concatenate(data_list)
labels=np.concatenate(label_list)



data_train,data_val,train_labels,val_labels=train_test_split(data,labels,test_size=0.2,random_state=32)

data_train=torch.FloatTensor(data_train)
train_labels=torch.FloatTensor(train_labels)

data_val=torch.FloatTensor(data_val)
val_labels=torch.FloatTensor(val_labels)

train_data=TensorDataset(data_train,train_labels)
val_data=TensorDataset(data_val,val_labels)

train_loader=DataLoader(dataset=train_data,shuffle=True,batch_size=batch_size,drop_last=True)
val_loader=DataLoader(dataset=val_data,shuffle=True,batch_size=batch_size,drop_last=True)

history={'train_Acc':[],'train_loss':[],'train_preds':{},'val_acc':[],'val_loss':[],'val_preds':{},'models':{}}



for epoch in tqdm(range(55)):
    t0 = time.time()
    input_labels=[]
    train_preds=[]
    train_loss_sum=0


    hidden=model.init_hidden(batch_size)
    model.train()
    for idx_train,(input_train,label_train) in enumerate(train_loader):

        input_train=input_train.to(device).long()
        label_train=label_train.to(device).long()
        h = tuple([each.data for each in hidden])

        preds=model(h,input_train)
        loss = criterion(preds, label_train.reshape(-1))

        train_loss_sum+=loss.item()
        model.zero_grad()
        loss.backward()
        optimizer.step()

        input_labels.append(label_train.cpu())
        train_preds.append(torch.argmax(preds,dim=1).cpu())

    train_acc=accuracy_score(np.concatenate(train_preds),np.concatenate(input_labels))
    Train_loss_mean=train_loss_sum/len(train_loader)


    val_loss=[]
    val_labels=[]
    val_preds=[]
    val_loss_sum=0

    model.eval()
    for idx_val,(input_val,label_val) in enumerate(val_loader):

        input_val=input_val.to(device).long()
        label_val=label_val.to(device).long()
        h = tuple([each.data for each in hidden])

        val_pred=model(h,input_val)

        val_loss = criterion(val_pred, label_val)
        val_loss_sum+=val_loss.item()


        val_labels.append(label_val.cpu())
        val_preds.append(torch.argmax(val_pred,dim=1).cpu())

    val_acc=accuracy_score(np.concatenate(val_preds),np.concatenate(val_labels))
    Val_loss_mean=val_loss_sum/len(val_loader)

    saved_model=model.state_dict()

    history['models'].update({'epoch_'+str(epoch):saved_model})
    history['train_Acc'].append(train_acc)
    history['train_loss'].append(Train_loss_mean)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(Val_loss_mean)
    history['train_preds'].update({'epoch_'+str(epoch):np.concatenate(train_preds)})
    history['val_preds'].update({'epoch_'+str(epoch):np.concatenate(val_preds)})

#     with tune.checkpoint_dir(epoch) as checkpoint_dir:
#         path = os.path.join(checkpoint_dir, "checkpoint")
#         torch.save((model.state_dict(), optimizer.state_dict()), path)
#     tune.report(loss=(Val_loss_mean / len(val_loader)), accuracy=val_acc)

    print('epoch {:} Train Accurarcy {:.2f}, Train Loss Mean {:.4f} ***-*** Validation Accuracy {:.2f} ,Accurarcy Loss Mean {:.4f} ---- > Time {:.2f} seconds'.format(epoch,train_acc,Train_loss_mean,val_acc,Val_loss_mean,(time.time() - t0)/60))








from ray import tune

config = {
        "vocab_size": len(one_hot_dict)+1,
        "embed_size": tune.choice([100,150,200]),
        "hidden_size": tune.choice([64,128,512]),
        "timestep": tune.choice([30,40]),
        'num_layer':tune.choice([1,2]),
        'prob':tune.choice([0.5,0.7,0.9]),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "lr": tune.loguniform(1e-3, 1e-1)}

def train(config, checkpoint_dir=None):
    

    model = textGenerator(config["vocab_size"], config["embed_size"], config["hidden_size"], config["num_layer"],config["prob"]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config["lr"])
    criterion=nn.CrossEntropyLoss()

    if checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint"))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    batch_size=config['batch_size']
    timestep=config['timestep']

    def create_rep_data(text,batch_size=config['batch_size']):
        rep_tensor=torch.LongTensor(np.array(text).astype(int)) #text list converting to  tensor 
        num_batch=rep_tensor.shape[0]//batch_size #net number of batches
        rep_tensor=rep_tensor[:num_batch*batch_size] #rep tensor 
        rep_tensor=rep_tensor.view(batch_size,-1) 
        num_batches=rep_tensor.shape[1]//timestep

        return rep_tensor,num_batches



    rep_tensor,_ = create_rep_data(text,config['batch_size'])


    data_list=[]
    label_list=[]
    for  k,i in tqdm(enumerate(range(0 ,rep_tensor.shape[1]-timestep))):

        inputs=rep_tensor[:,i:i+timestep]
        labels=rep_tensor[:,i+timestep]
        data_list.append(inputs)
        label_list.append(labels)

    data=np.concatenate(data_list)
    labels=np.concatenate(label_list)



    data_train,data_val,train_labels,val_labels=train_test_split(data,labels,test_size=0.5,random_state=32)

    data_train=torch.FloatTensor(data_train)
    train_labels=torch.FloatTensor(train_labels)

    data_val=torch.FloatTensor(data_val)
    val_labels=torch.FloatTensor(val_labels)

    train_data=TensorDataset(data_train,train_labels)
    val_data=TensorDataset(data_val,val_labels)



    train_loader=DataLoader(dataset=train_data,shuffle=True,batch_size=batch_size,drop_last=True)
    val_loader=DataLoader(dataset=val_data,shuffle=True,batch_size=batch_size,drop_last=True)






    # model=textGenerator(vocab_size,embed_size,hidden_size,num_layer=1,prob=0.2).to(device)



    history={'train_Acc':[],'train_loss':[],'train_preds':{},'val_acc':[],'val_loss':[],'val_preds':{},'models':{}}



    for epoch in tqdm(range(55)):
        t0 = time.time()
        input_labels=[]
        train_preds=[]
        train_loss_sum=0


        hidden=model.init_hidden(batch_size)
        model.train()
        for idx_train,(input_train,label_train) in enumerate(train_loader):

            input_train=input_train.to(device).long()
            label_train=label_train.to(device).long()
            h = tuple([each.data for each in hidden])

            preds=model(h,input_train)
            loss = criterion(preds, label_train.reshape(-1))

            train_loss_sum+=loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            input_labels.append(label_train.cpu())
            train_preds.append(torch.argmax(preds,dim=1).cpu())

        train_acc=accuracy_score(np.concatenate(train_preds),np.concatenate(input_labels))
        Train_loss_mean=train_loss_sum/len(train_loader)


        val_loss=[]
        val_labels=[]
        val_preds=[]
        val_loss_sum=0

        model.eval()
        for idx_val,(input_val,label_val) in enumerate(val_loader):

            input_val=input_val.to(device).long()
            label_val=label_val.to(device).long()
            h = tuple([each.data for each in hidden])

            val_pred=model(h,input_val)

            val_loss = criterion(val_pred, label_val)
            val_loss_sum+=val_loss.item()


            val_labels.append(label_val.cpu())
            val_preds.append(torch.argmax(val_pred,dim=1).cpu())

        val_acc=accuracy_score(np.concatenate(val_preds),np.concatenate(val_labels))
        Val_loss_mean=val_loss_sum/len(val_loader)

        saved_model=model.state_dict()

        history['models'].update({'epoch_'+str(epoch):saved_model})
        history['train_Acc'].append(train_acc)
        history['train_loss'].append(Train_loss_mean)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(Val_loss_mean)
        history['train_preds'].update({'epoch_'+str(epoch):np.concatenate(train_preds)})
        history['val_preds'].update({'epoch_'+str(epoch):np.concatenate(val_preds)})

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=(Val_loss_mean / len(val_loader)), accuracy=val_acc)

        print('epoch {:} Train Accurarcy {:.2f}, Train Loss Mean {:.4f} ***-*** Validation Accuracy {:.2f} ,Accurarcy Loss Mean {:.4f} ---- > Time {:.2f} seconds'.format(epoch,train_acc,Train_loss_mean,val_acc,Val_loss_mean,(time.time() - t0)/60))
    return  

    


from ray.tune import CLIReporter
reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

#-----------------------------------------------------------------------------
gpus_per_trial = 1
from functools import partial
result = tune.run(
    partial(train),
    resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
    config=config,
    num_samples=20,
#     scheduler=scheduler,
    progress_reporter=reporter,
    checkpoint_at_end=False 
)
    
    

best_trial = result.get_best_trial("loss", "min", "last")
print("Best trial config: {}".format(best_trial.config))
print("Best trial final validation loss: {}".format(
    best_trial.last_result["loss"]))
print("Best trial final validation accuracy: {}".format(
    best_trial.last_result["accuracy"]))

best_trained_model = textGenerator(best_trial.config["vocab_size"], best_trial.config["embed_size"], best_trial.config["hidden_size"],best_trial.config["num_layer"],best_trial.config["prob"]).to(device)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
    if gpus_per_trial > 1:
        best_trained_model = nn.DataParallel(best_trained_model)
best_trained_model.to(device)



best_checkpoint_dir = best_trial.checkpoint.dir_or_data
model_state, optimizer_state = torch.load(os.path.join(
    best_checkpoint_dir, "checkpoint"))
best_trained_model.load_state_dict(model_state)




plt.figure(figsize=(15,15))
plt.title('Train And Validation Accuracy throughout epochs')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.plot(history['val_loss'])
plt.plot(history['train_loss'])
plt.legend(['validation','train']);


plt.figure(figsize=(15,15))
plt.title('Train And Validation Accuracy throughout epochs')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.plot(history['val_acc']);
plt.plot(history['train_Acc']);
plt.legend(['validation','train'])

    
    
    
    