#!/usr/bin/env python
# coding: utf-8


import os, sys
import math
import random
import cv2
import pickle
import numpy as np
import pandas as pd
from PIL import Image

import torch, torchvision
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import models, transforms

sys.dont_write_bytecode = True
from utils import models, data
 
########## Define Morphable Training Hyperparameters  ############

# Training Hyperparameters
BATCH_SIZE = 28
IMG_SIZE = 224

NUM_EPOCHS = 50
LR = .0005
LRD = .25
PATIENCE = 3


# Model Preparing
MODEL_NAMES = ['resnext50','alexnet','vgg16','densenet201']

# Dataset Preparing
DATASETS = ['c_tinyimagenet', 'd_pascal', 'c_ham1', 's_glas']


########## Define Constant Elements  ############

# Model
MODEL_HANDLER = models.ModelsV1()

# models = []
# for mn in MODEL_NAMES:
#     model = MODEL_HANDLER.initialize_model(mn)
#     print(f"Loaded {mn}: " )
#     print(repr(model)); print()

# Data Handling
DATASET_HANDLER = data.DatasetHandler(DATASETS)

# Data Descriptions
NORM_MEAN = None
NORM_STD = None



# Print crucial training info
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device ({DEVICE})..")
print(f"Training on models:({MODEL_NAMES})..")
print(f"Training on datasets: ({DATASETS})..")







### Data Methods and Containers



### Instantiate Data Constants
train_transform = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                      transforms.ToTensor(), 
                                      transforms.Normalize(NORM_MEAN, NORM_STD)])
val_transform = transforms.Compose([transforms.Resize((IMG_SIZE,IMG_SIZE)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize(NORM_MEAN, NORM_STD)])



### Model Functions
## Import ModelHandler from modelhandler
model = model_handler.initialize_model(model_name)


### Statistics and Evaluation
class StatTracker:
    def __init__(self):
        self.iter_train_loss = []
        self.iter_train_acc = []
        self.full_train_acc = []
        self.full_train_loss = []
        self.full_val_acc = []
        self.full_val_loss = []
    def iter_update(self, tloss, tacc):
        self.iter_train_loss.append(tloss)
        self.iter_train_acc.append(tacc)
    def full_update(self, tacc, tloss, vacc, vloss):
        self.full_train_acc.append(tacc)
        self.full_train_loss.append(tloss)
        self.full_val_acc.append(vacc)
        self.full_val_loss.append(vloss)


def validate(tracker, model, criterion, optimizer):
    sum_vl, sum_vacc = 0., 0.
    with torch.no_grad():
        print('Computing val stats..')
        for i, data in enumerate(VAL_LOADER):
            images, labels = data
            N = images.size(0)  # batch size
            images = Variable(images).to(DEVICE)
            labels = Variable(labels).to(DEVICE)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]
            num_correct = prediction.eq(labels.view_as(prediction)).sum().item()

            sum_vl += criterion(outputs, labels).item()
            sum_vacc += num_correct/N
    vacc, vloss = sum_vacc/len(VAL_LOADER), sum_vl/len(VAL_LOADER)
    return vloss, vacc


### Training Functions

def train_model(train_loader, model, criterion, optimizer,
                epochs=10, tracker=None):
    model.train()
    
    for epoch in range(epochs):
        print(f'\n=========\nTraining Epoch {epoch+1} ({model.name})\n=========\n')
        sum_tl, sum_tacc = 0., 0.
        for i, data in enumerate(train_loader):
            images, labels = data
            images = Variable(images).to(DEVICE)
            labels = Variable(labels).to(DEVICE)
            N = images.size(0)
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            prediction = output.max(1, keepdim=True)[1]
            num_correct = prediction.eq(labels.view_as(prediction)).sum().item()
            
            # Stats and status
            sum_tl += loss.item()
            sum_tacc += num_correct/N
            if i % 100 == 0:
                tacc = num_correct/N
                print(f'[Epoch {epoch+1}], [Iter {i+1}/{len(train_loader)+1}], '
                      f'[TrnLoss {loss.item():.4}], [TrnAcc {tacc:.4}]')
                tracker.iter_update(loss.item(),tacc)
        tloss, tacc = sum_tl/len(train_loader), sum_tacc/len(train_loader)
        vloss, vacc = validate(tracker, model, criterion, optimizer)
        tracker.full_update(tacc, tloss, vacc, vloss)
        print(f'-------\n[Epoch {epoch+1}], [Train Acc {tacc:.4}, '
                      f'[Val Acc {vacc:.4}], [Val Loss {vloss:.4}]')

def train_models():
    trackers = []
    
    # Train
    for modelname in MODEL_NAMES:
        model_ft, input_size = initialize_model(modelname, 
                                                len(CLASSES), 
                                                feature_extract=False, 
                                                use_pretrained=True)
        model = model_ft.to(DEVICE)
        model.name = modelname
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=.01)
        criterion = nn.CrossEntropyLoss().to(DEVICE)
        tracker = StatTracker()
        
        train_model(TRAIN_LOADER,
                    model,
                    criterion,
                    optimizer,
                    epochs=NUM_EPOCHS,
                    tracker=tracker)
        trackers.append(tracker)
        torch.save({'epoch': NUM_EPOCHS,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    f'./{modelname}-v3-ep{NUM_EPOCHS}-decay-balancedtrainset.pth')
    
    return trackers
    


if __name__ == '__main__':
    trackers = train_models()
    with open('tracker_list.pkl', 'wb') as f:
        pickle.dump(trackers, f)





