from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F


class get_target():
    def __init__(self):
        self.target_data = pd.read_csv('HAM10000_metadata.csv')
        
    def get(self,index):
        kamus = {'nv':1,
            'mel':2,
            'bkl':3,
            'bcc' :4,
            'akiec' :5,
            'df' : 6,
            'vasc':7,
            }
        return kamus[self.target_data['dx'][index]]
        
               
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,9,12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(9, 16, 12)
        self.fc1 = nn.Linear( 336, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

