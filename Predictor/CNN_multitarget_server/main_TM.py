from __future__ import print_function

import os

#from Utilities.SaveAnimation import Video

from druida import Stack
from druida import setup

from druida.DataManager import datamanager
#from druidaHFSS.modules import tools
from druida.tools import utils

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optimizer

from torchsummary import summary
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from IPython.display import HTML

import glob
from tqdm.notebook import tqdm

import argparse

import json



# Clip 
from typing import List

from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel

torch.set_printoptions(profile="full")
torch.manual_seed(90)




# Arguments
parser = argparse.ArgumentParser()
# boxImagesPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Images Jorge Cardenas 512\\"
# DataPath="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\Exports\\output\\"
# simulationData="\\data\\francisco_pizarro\\jorge-cardenas\\data\\MetasufacesData\\DBfiles\\"

boxImagesPath="../../../data/MetasufacesData/Images Jorge Cardenas 512/"
DataPath="../../../data/MetasufacesData/Exports/output/"
simulationData="../../../data/MetasufacesData/DBfiles/"


Substrates={"Rogers RT/duroid 5880 (tm)":0}
Materials={"copper":0,"pec":1}
Surfacetypes={"Reflective":0,"Transmissive":1}
TargetGeometries={"circ":0,"box":1, "cross":2}



def arguments():

    parser.add_argument("run_name",type=str)
    parser.add_argument("epochs",type=int)
    parser.add_argument("batch_size",type=int)
    parser.add_argument("workers",type=int)
    parser.add_argument("gpu_number",type=int)
    parser.add_argument("device",type=str)
    parser.add_argument("learning_rate",type=float)
    parser.add_argument("condition_len",type=float) #This defines the length of our conditioning vector
    parser.add_argument("metricType",type=float) #This defines the length of our conditioning vector

    parser.run_name = "Predictor Training"
    parser.epochs = 30
    parser.batch_size = 5
    parser.workers=0
    parser.gpu_number=1
    parser.image_size = 512
    parser.dataset_path = os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/')
    parser.device = "cpu"
    parser.learning_rate = 1e-6
    parser.condition_len = 768
    parser.metricType='AbsorbanceTM' #this is to be modified when training for different metrics.

    categories=["box", "circle", "cross"]




# Images loading
    

def load_images():
    pass

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Data pre-processing
def join_simulationData():
    df = pd.DataFrame()
    for file in glob.glob(simulationData+"*.csv"): 
        df2 = pd.read_csv(file)
        df = pd.concat([df, df2], ignore_index=True)
    
    df.to_csv('out.csv',index=False)
    

# Load Model
def loadModel(device):

    fwd_test = Stack.Predictor_CNN(cond_input_size=parser.condition_len, 
                                ngpu=1, image_size=parser.image_size ,
                                output_size=8, channels=3,
                                features_num=1000,hiden_num=3000, #Its working with hiden nums. Features in case and extra linear layer
                                dropout=0.2, 
                                Y_prediction_size=601) #size of the output vector in this case frenquency points
    
    fwd_test.apply(weights_init)

    """using weigth decay regularization"""
    opt = optimizer.Adam(fwd_test.parameters(), lr=parser.learning_rate, betas=(0.5, 0.999),weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    fwd_test.train()
    return fwd_test, opt, criterion

class CLIPTextEmbedder(nn.Module):
    """
    ## CLIP Text Embedder
    """

    def __init__(self, version: str = "openai/clip-vit-large-patch14", device="cuda:0", max_length: int = 77):
        """
        :param version: is the model version
        :param device: is the device
        :param max_length: is the max length of the tokenized prompt
        """
        super().__init__()
        # Load the tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(version,device_map = device)
        # Load the CLIP transformer
        self.transformer = CLIPTextModel.from_pretrained(version,device_map = device).eval()

        self.device = device

        print(self.device)
        self.max_length = max_length

    def forward(self, prompts: List[str]):
        """
        :param prompts: are the list of prompts to embed
        """
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt").to(self.device)
        # Get token ids
        tokens = batch_encoding["input_ids"]
        # Get CLIP embeddings
        return self.transformer(input_ids=tokens).last_hidden_state


# Conditioning
def set_conditioning(target,path,categories,clipEmbedder,df,device):
    
    arr=[]

    for idx,name in enumerate(path):
        series=name.split('_')[-1].split('.')[0]
        batch=name.split('_')[4]
        iteration=series.split('-')[-1]
        row=df[(df['sim_id']==batch) & (df['iteration']==int(iteration))  ]

        
        target_val=target[idx]
        category=categories[idx]
        geometry=TargetGeometries[category]
        
        """"
        surface type: reflective, transmissive
        layers: conductor and conductor material / Substrate information
        """
        surfacetype=row["type"].values[0]
        surfacetype=Surfacetypes[surfacetype]
        
        layers=row["layers"].values[0]
        layers= layers.replace("'", '"')
        layer=json.loads(layers)
        
        materialconductor=Materials[layer['conductor']['material']]
        materialsustrato=Substrates[layer['substrate']['material']]
        
        
        if (target_val==2): #is cross. Because an added variable to the desing 
            
            sustratoHeight= json.loads(row["paramValues"].values[0])
            sustratoHeight= sustratoHeight[-2]
        else:
        
            sustratoHeight= json.loads(row["paramValues"].values[0])
            sustratoHeight= sustratoHeight[-1]
        
        arr.append([geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight,1,1,1,1,1])
    
        datos=" ".join([str(element) for element in  [geometry,surfacetype,materialconductor,materialsustrato,sustratoHeight,1,1,1,1,1]])
        embedding=clipEmbedder(prompts=(datos))
        
    return arr, embedding



def train(opt,criterion,fwd_test, clipEmbedder,device):
    #### #File reading conf

    a = []
    idx=0
    iters=0
    loss_values, valid_loss_list = [], []
    acc,acc_val=[], []
    df = pd.read_csv("out.csv")

    for epoch in range(parser.epochs):
        x=0
        running_loss = 0.0
        i=0
        acc_val=[]
        print('Epoch {}/{}'.format(epoch, parser.epochs - 1))
        print('-' * 10)
        
        
        dataloader = utils.get_data_with_labels(parser.image_size,parser.image_size,0.98, boxImagesPath,parser.batch_size,drop_last=True)

        
        for data in tqdm(dataloader):
            
            inputs, classes, names, classes_types = data
            inputs = inputs.to(device)
            classes = classes.to(device)
            
            opt.zero_grad()
            #Loading data
            a = []
            idx=0
            """lookup for data corresponding to every image in training batch"""
            for name in names:
                series=name.split('_')[-1].split('.')[0]
                batch=name.split('_')[4]

                for name in glob.glob(DataPath+batch+'/files/'+'/'+parser.metricType+'*'+series+'.csv'): 
                    #loading the absorption data
                    train = pd.read_csv(name)
                    values=np.array(train.values.T)
                    a.append(values[1])
                    
                    
            a=np.array(a)     

            
            array, embedded=set_conditioning(classes, names, classes_types,clipEmbedder,df,device)
            embedded=embedded.to(device)
            #conditioningArray=torch.FloatTensor(array)
            
            if embedded.shape[2]==parser.condition_len:
                
            
                conditioningTensor = torch.nn.functional.normalize(embedded, p=2.0, dim = 1)

                y_predicted=fwd_test(input_=inputs, conditioning=conditioningTensor.to(device) ,b_size=inputs.shape[0])
                y_predicted=torch.nn.functional.normalize(y_predicted, p=2.0, dim = 1)
                
                y_predicted=y_predicted.to(device)
                
                y_truth = torch.tensor(a).to(device)
                
            
                errD_real = criterion(y_predicted.float(), y_truth.float())
                errD_real.backward()
                loss=errD_real.item()

                opt.step()
                # scale = torch.tensor([10.0])

                running_loss +=loss*inputs.size(0)
                acc_val= (y_predicted.argmax(dim=-1) == y_truth.argmax(dim=-1)).float().mean()

                x += 1
                i += 1


                if i % 300 == 5:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss / 10:.3f} running loss:  {running_loss / 10:.3f}')
                    print(f'accuracy: {acc_val.mean() :.3f} ')

                iters += 1
            else:
            
                break
        
        loss_values.append(running_loss)
        acc.append(acc_val.cpu().mean())

    
    return running_loss,loss_values,acc





def main():

    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

    print("Access main")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)

    arguments()

    join_simulationData()  

    fwd_test, opt, criterion=loadModel(device)
    fwd_test = fwd_test.to(device)

    ClipEmbedder=CLIPTextEmbedder(version= "openai/clip-vit-large-patch14",device=device, max_length = parser.batch_size)

    running_loss,loss_values,acc=train(opt,criterion,fwd_test,ClipEmbedder,device)

    PATH = 'trainedModelTM_abs_19March.pth'
    torch.save(fwd_test.state_dict(), PATH)

    try:
        np.savetxt('loss_ABS_TM_19March.out', loss_values, delimiter=',')
        
    except:
        np.savetxt('loss_ABS_TM_19March.out', [], delimiter=',')

    try:
        np.savetxt('acc_TM_19March.out', acc, delimiter=',')
    except:
        np.savetxt('acc_TM_19March.out', [], delimiter=',')
    
    try:
        np.savetxt('runninLoss_TM_19March.out', running_loss, delimiter=',')
    except:
        np.savetxt('runninLoss_TM_19March.out', [], delimiter=',')

if __name__ == "__main__":
    main()