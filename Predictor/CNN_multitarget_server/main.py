from __future__ import print_function

import os

#from Utilities.SaveAnimation import Video

from druida import Stack
from druida import setup

from druida.DataManager import datamanager
from druidaHFSS.modules import tools
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

torch.set_printoptions(profile="full")
torch.manual_seed(90)

def arguments():
    parser = argparse.ArgumentParser()

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
    print(categories)



def main():
    print("main")
    arguments()
    
if __name__ == "__main__":
    main()