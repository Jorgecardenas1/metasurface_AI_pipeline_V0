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

# Clip 
from typing import List

from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel

torch.set_printoptions(profile="full")
torch.manual_seed(90)




# Arguments
parser = argparse.ArgumentParser()

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
    parser.epochs = 1
    parser.batch_size = 5
    parser.workers=0
    parser.gpu_number=0
    parser.image_size = 512
    parser.dataset_path = os.path.normpath('/content/drive/MyDrive/Training_Data/Training_lite/')
    parser.device = "cpu"
    parser.learning_rate = 5e-6
    parser.condition_len = 768
    parser.metricType='AbsorbanceTM' #this is to be modified when training for different metrics.

    categories=["box", "circle", "cross"]




# Images loading
def load_images():
    boxImagesPath="C:\\Users\\jorge\\Dropbox\\Public\\MetasufacesData\\Images Jorge Cardenas 512\\"
    DataPath="C:\\Users\\jorge\\Dropbox\\Public\\MetasufacesData\\Exports\\output\\"
    simulationData="C:\\Users\\jorge\\Dropbox\\Public\\MetasufacesData\\DBfiles\\"
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Load Model
def loadModel():

    fwd_test = Stack.Predictor_CNN(cond_input_size=parser.condition_len, 
                                ngpu=0, image_size=parser.image_size ,
                                output_size=8, channels=3,
                                features_num=1000,hiden_num=1000, #Its working with hiden nums. Features in case and extra linear layer
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
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        # Load the CLIP transformer
        self.transformer = CLIPTextModel.from_pretrained(version).eval()

        self.device = device
        self.max_length = max_length

    def forward(self, prompts: List[str]):
        """
        :param prompts: are the list of prompts to embed
        """
        # Tokenize the prompts
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        # Get token ids
        tokens = batch_encoding["input_ids"].to(self.device)
        # Get CLIP embeddings
        return self.transformer(input_ids=tokens).last_hidden_state

def main():
    print("Access main")
    arguments()
    fwd_test, opt, criterion=loadModel()
    # print(fwd_test)
    ClipEmbedder=CLIPTextEmbedder(version= "openai/clip-vit-large-patch14", device="cuda:0", max_length = parser.batch_size)

    
if __name__ == "__main__":
    main()