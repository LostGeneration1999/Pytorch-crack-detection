import glob
import os
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models,transforms

#GPU設定
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

#random seed
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

class ImageTransform():
    
    #constructor
    def __init__(self, resize, mean, std):
        
        self.data_transform = {
            #data augumation
            'train': transforms.Compose([
                transforms.RandomResizedCrop(
                resize, scale=(0.5,1.0)),
                transforms.RandomHorizontalFlip(),
                #convert to tensor 
                transforms.ToTensor(),
                #標準化
                transforms.Normalize(mean, std)        
            ]),
            'validation': transforms.Compose([
                transforms.RandomResizedCrop(
                resize, scale=(0.5,1.0)),
                transforms.RandomHorizontalFlip(),
                #convert to tensor 
                transforms.ToTensor(),
                #標準化
                transforms.Normalize(mean, std)        
            ])
            
        }
    
    def __call__(self, img, phase='train'):
        #前処理モードを指定
        return self.data_transform[phase](img)

def make_datapath_list(phase='train'):
    
    path = './data/'
    target_path = os.path.join(path+phase+'/**/*.jpg')
    print(target_path)
    
    path_list = []
    
    for path in glob.glob(target_path):
        path_list.append(path)
        
    return path_list

class HymenopterDataset(data.Dataset):
    
    def __init__(self, file_list, transform=None, phase='train', lis=[]):
        #set filelist
        self.file_list = file_list
        #前処理クラスのインスタンス
        self.transform = transform
        #train or validation
        self.phase = phase
        self.classes_list = lis
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        
        img_path = self.file_list[index]
        #print(img_path)
        #PIL Image
        img = Image.open(img_path)
        
        #画像の前処理
        img_transformed = self.transform(img, self.phase)
        
        #label name
        #label nameのコードを使いやすいように改変
        if self.phase == 'train':
            path_lis = img_path.split('/')
            categorical_label = path_lis[3]
        elif self.phase == 'validation':
            path_lis = img_path.split('/')
            categorical_label = path_lis[3]
            
        
        if not path_lis[3] in self.classes_list:
            self.classes_list.append(categorical_label)
        #print(self.classes_list)
        
        label = self.classes_list.index(categorical_label)
        #print('label : %d'%label)
        
        return img_transformed, label
