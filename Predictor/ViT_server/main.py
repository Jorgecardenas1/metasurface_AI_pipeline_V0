
from __future__ import print_function

import os

#from Utilities.SaveAnimation import Video

from druida import Stack
from druida import setup

from druida.DataManager import datamanager
from druidaHFSS.modules import tools
from druida.tools import utils


#import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.optim as optimizer

import torchvision

from torchvision import transforms

import glob
from tqdm.notebook import tqdm
import random
import numpy as np
import pandas as pd

from IPython.display import HTML
import json

import argparse


# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/VisionTransformers/")

# Setting the seed
#L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)


parser = argparse.ArgumentParser()
boxImagesPath="../../../data/MetasufacesData/Images Jorge Cardenas 512/"
validationImages="../../../data/MetasufacesData/testImages/"
DataPath="../../../data/MetasufacesData/Exports/output/"
simulationData="../../../data/MetasufacesData/DBfiles/"


Substrates={"Rogers RT/duroid 5880 (tm)":0}
Materials={"copper":0,"pec":1}
Surfacetypes={"Reflective":0,"Transmissive":1}
TargetGeometries={"circ":0,"box":1, "cross":2}
metricType=['AbsorbanceTM','AbsorbanceTE' ]

categories=["box", "circle", "cross"]

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
    parser.add_argument("patch_size",type=int)
    parser.add_argument("image_size",type=int)

    parser.run_name = "Predictor Training"
    parser.epochs = 1
    parser.batch_size = 10
    parser.workers=0
    parser.gpu_number=1
    parser.image_size = 512
    parser.dataset_path = os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/')
    parser.device = "cpu"
    parser.learning_rate = 1e-6
    parser.condition_len = 10
    parser.metricType='AbsorbanceTM' #this is to be modified when training for different metrics.
    parser.patch_size=16


    model_kwargs={
            "batch_size":parser.batch_size,
            "embed_dim":  3 * (parser.patch_size)**2 ,
            "hidden_dim":  2*(3 * (parser.patch_size)**2),
            "num_heads": 8, #probar menos
            "num_layers": 4, #probar menos
            "patch_size": parser.patch_size,
            "num_channels": 3,
            "num_patches": (parser.image_size//parser.patch_size)**2,
            "num_classes": 601,
            "dropout": 0.1,
            "image_size":parser.image_size,
            "conditionalIn":True,
            "conditionalLen":10
        }




    return model_kwargs


    
def train(model):
    #### #File reading conf


    """using weigth decay regularization"""
    opt = optimizer.Adam(model.parameters(), lr=parser.learning_rate, betas=(0.5, 0.999),weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()


    loss_per_batch=0
    loss_per_val_batch=0
    loss_values, valid_loss_list = [], []
    acc=[]
    acc_val=[]

    
    if parser.device!='cpu':
        model.to(device)

    # prepare model fr training


    for epoch in range(parser.epochs):
        dataloader = utils.get_data_with_labels(parser.image_size, parser.image_size,1, boxImagesPath,parser.batch_size, drop_last=True)
        vdataloader = utils.get_data_with_labels(parser.image_size, parser.image_size,1, validationImages,10, drop_last=True)

        i=0
        i_val=0

        running_loss = 0.
        running_vloss = 0.0
        total_correct = 0
        
        total_samples=0
        total_samples_val=0.0
        last_loss = 0.

        print('Epoch {}/{}'.format(epoch, parser.epochs - 1))
        print('-' * 10)
        
        #dataloader = utils.get_data_with_labels(512, 512,0.9, boxImagesPath,parser.batch_size,drop_last=True)

        #Training
        model.train()

        """batch"""
        for data in tqdm(dataloader):
            
            #if parser.device!='cpu':
            images, classes, names, classes_types = data
            
            images = images.to(device)
            classes = classes.to(device)
            #else:
            #    images, classes, names, classes_types = data

            a = [] #array with truth values
            
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


            #Aun sin CLIP
            conditioningArray=torch.FloatTensor(set_conditioning(classes, names, classes_types))
            conditioningArray=conditioningArray.to(device)

            #Train
            #if conditioningArray.shape[1]==parser.condition_len:
                
                
            opt.zero_grad()

            #for conditioning in case required
            #outmap_min, _ = torch.min(conditioningArray, dim=1, keepdim=True)
            #outmap_max, _ = torch.max(conditioningArray, dim=1, keepdim=True)
            #conditioningTensor = (conditioningArray - outmap_min) / (outmap_max - outmap_min)

            conditioningArray = torch.nn.functional.normalize(conditioningArray, p=2.0, dim=-1, eps=1e-12, out=None)
            # feedforward data

            y_predicted = model(images,condition=conditioningArray)
            y_predicted=y_predicted.to(device)
            #y_predicted= F.softmax (y_predicted,dim=-1)
            
            y_truth = torch.tensor(a).to(device)

            errD_real = criterion(y_predicted.float(), y_truth.float())                
            errD_real.backward()
            opt.step()

            #Metrics
            #acc_train= (y_predicted.argmax(dim=-1) == y_truth.argmax(dim=-1)).float().mean()

            total_correct += (y_predicted == y_truth).sum().item()

            loss_per_batch=errD_real.item()

            total_samples += y_truth.size(0)
            acc_train = 100 * total_correct / total_samples


            last_loss=loss_per_batch 
            running_loss +=last_loss

            i += 1


            if i % 100 ==  99:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {last_loss/i :.3f} running loss:  {running_loss/i:.3f}')
                print(f'accuracy: {acc_train :.3f} ')

            #else:

               # break


        loss_values.append(running_loss/i)
        acc.append(acc_train)
        print("train acc",acc)
        

        """validation"""

        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        with torch.no_grad():
            for vdata in tqdm(vdataloader):
                images, classes, names, classes_types = data = vdata
                

                images = images.to(device)
                classes = classes.to(device)

            
                a = [] #array with truth values
                
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
                #Aun sin CLIP
                conditioningArray=torch.FloatTensor(set_conditioning(classes, names, classes_types))
                conditioningArray=conditioningArray.to(device)

                y_predicted = model(images,condition=conditioningArray)
                y_predicted=y_predicted.to(device)
                
                y_truth = torch.tensor(a).to(device)

                i_val+=1
                loss_per_val_batch = criterion(y_predicted.float(), y_truth.float())
                running_vloss += loss_per_val_batch.item()

                #Metrics

                total_samples_val += y_truth.size(0)
                acc_val_ = 100 * total_correct / total_samples


                last_loss=loss_per_batch 
                running_loss +=last_loss


        valid_loss_list.append(running_vloss/i_val)
        acc_val.append(acc_val_)
        print("val vloss",running_vloss/i_val)
        print("val loss list",valid_loss_list)
        print("acclist",acc_val)

    return running_loss,loss_values,acc,valid_loss_list,acc_val

def metrics():
    pass

    
   

           
def set_conditioning(target,path,categories):
    df = pd.read_csv("out.csv")
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
    
    return arr

def join_simulationData():
    df = pd.DataFrame()
    for file in glob.glob(simulationData+"*.csv"): 
        df2 = pd.read_csv(file)
        df = pd.concat([df, df2], ignore_index=True)
    
    df.to_csv('out.csv',index=False)

def main():
    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

    print("Access main")

    model_kwargs=arguments()

    join_simulationData()  

    vision_transformer = Stack.VisionTransformer(**model_kwargs )

    vision_transformer.to(device)

    running_loss,loss_values,acc,valid_loss_list,acc_val=train(vision_transformer)

    PATH = 'trainedModelTM_abs_21March.pth'
    torch.save(vision_transformer.state_dict(), PATH)

    try:
        np.savetxt('output/loss_Train_TM_21March.out', loss_values, delimiter=',')
        
    except:
        np.savetxt('output/loss_Train_TM_21March.out', [], delimiter=',')

    try:
        np.savetxt('output/acc_Train_TM_21March.out', acc, delimiter=',')
    except:
        np.savetxt('output/acc_Train_TM_21March.out', [], delimiter=',')
    
    try:
        np.savetxt('output/acc_val_21March.out', acc_val, delimiter=',')
    except:
        np.savetxt('output/acc_val_21March.out', [], delimiter=',')


if __name__ == "__main__":
    main()