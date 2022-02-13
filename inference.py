import numpy as np
import pandas as pd
from glob import glob

import cv2
import json
import os
import sys

from sklearn import metrics

from tqdm import tqdm
import uuid

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import imgaug
from imgaug import augmenters as iaa
import pickle
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import tcga.utils as utils
import copy
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_name', default='', type=str)
parser.add_argument('--fold', default=5, type=int)
parser.add_argument('--uuid', default='', type=str)
parser.add_argument('--baseline', default='resnet18', type=str)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--data', default='MSI', type=str)
args = parser.parse_args()

CFG = {'data':args.data,
       'fold':args.fold, 
       'baseline': args.baseline,
       'epochs':None,
       'epochs_patience':None,
       'lr':None,
       'L2_lambda':None,
       'freeze_layer':None, #if 0, not freeze, [0,5,6,7]
       'batch_size':args.batch_size,
       'num_classes':args.num_classes,
       'k_ratio':None,
       'cohort':'COAD',
       'max_iter':None, #validation steps
       'model_unique_name':args.model_name,
       'model_unique_identifier':args.uuid}

print(CFG)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=-1).reshape(-1,1)

def set_seed(seed = 0):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    return random_state

class ClassificationDataset:
    
    def __init__(self, image_paths, targets, num_classes, transform=None): 
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, item):      

        image = cv2.imread(self.image_paths[item])[:,:,::-1].astype(np.uint8)
        if self.transform:
            image = self.transform(image)
        image = image/128-1
        image = torch.from_numpy(image.copy())
        targets = self.targets[item]
        return {
            #"image": torch.tensor(image, dtype=torch.float).permute(2, 0, 1).contiguous(),
            "image":image.permute(2, 0, 1).contiguous(),
            # "targets": F.one_hot(torch.tensor(targets), num_classes = self.num_classes),
            "targets": F.one_hot(torch.tensor(targets), num_classes = self.num_classes),
        }
    
def evaluate(data_loader, model, criterion, device, k_ratio=None, debug=False):
    model.eval()
    
    final_targets = []
    final_outputs = []
    loss_sum = []
    with torch.no_grad():
        
        for data in tqdm(data_loader, position=0, leave=True, desc='Evaluating'):
            inputs = data["image"].to(device, dtype=torch.float)
            targets = data["targets"].to(device, dtype=torch.float)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # loss = MIL_loss(outputs, targets, criterion = criterion, k=k, isLong=False)
            # loss = MIL_loss(outputs, targets, criterion = criterion, k_ratio=k_ratio, isLong=False)
            targets = targets.detach().cpu().numpy().tolist()
            outputs = outputs.sigmoid().detach().cpu().numpy().tolist()
            
            final_targets.extend(targets)
            final_outputs.extend(outputs)
            loss_value = loss.detach().cpu().numpy().tolist()
            loss_sum.append(loss_value)
            
    valid_loss = np.asarray(loss_sum).mean()        
    return valid_loss, final_outputs, final_targets


random_state = set_seed(2021)    

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

if CFG['data'] == 'MSI':
    df = utils.get_df_msi(isFold = True, k_fold = CFG['fold'])
    #df = df[df['isTrain'] == 'test']
    #df = df[df['cohort'] == CFG['cohort']]
    df['target'] = 1*(df['target'] == 'MSI')
    
elif CFG['data'] == 'TISSUE':
    df = utils.get_df_tissue(isFold = True, k_fold = CFG['fold'])
    df = df[df['isTrain'] == 'test']
else:
    print('No dataset config')
def worker_init_fn(worker_id):
    imgaug.seed(np.random.get_state()[1][0] + worker_id)

baseline_name = CFG['baseline'] + '_' + CFG['model_unique_identifier']
epochs = CFG['epochs']
Batch_Size = CFG['batch_size']
num_classes = CFG['num_classes']
model_unique_name = CFG['model_unique_name']

model_path = f'./checkpoint/{model_unique_name}/'
os.makedirs(model_path, exist_ok=True)
result_log = {}
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.MultiLabelSoftMarginLoss()
#criterion = nn.MultiLabelSoftMarginLoss(reduction='none')


df_test = df.copy().reset_index()


for fold in range(CFG['fold']):

    #model = EfficientNet.from_pretrained(baseline_name, num_classes=num_classes)
    if CFG['baseline'] == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        print('MODEL LOADED : resnet18')
    elif CFG['baseline'] == 'shufflenet':
        model = models.shufflenet_v2_x1_0(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        print('MODEL LOADED : shufflenet_v2_x1_0')
    elif CFG['baseline'] == 'efficientnet':
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        print('MODEL LOADED : efficientnet-b0')
    elif CFG['baseline'] == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        print('MODEL LOADED : resnext50_32x4d')
    model.to(device)
    #PATH = f'./checkpoint/Kather_RP_resnet_8F/resnet18_da5d781c_F{fold}_best.pt'
    PATH = f'{model_path}/{baseline_name}_F{fold}_best.pt'
    model.load_state_dict(torch.load(PATH))
    model.eval()

    valid_images, valid_targets= df_test['path'], df_test['target']
    seq_valid_dataset = ClassificationDataset(image_paths=valid_images, targets=valid_targets, num_classes=num_classes)
    seq_valid_loader = torch.utils.data.DataLoader(seq_valid_dataset, batch_size=Batch_Size, shuffle=False, num_workers=8)
    valid_loss, predictions, valid_targets = evaluate(data_loader=seq_valid_loader, model=model, criterion=criterion, device=device)

    df_test[f'pred_argmax_F{fold}'] = np.argmax(predictions,axis=-1)
    for class_ind in range(num_classes):
        df_test[f'pred_raw_{class_ind}_F{fold}'] = np.asarray(predictions)[:,class_ind] 
        df_test[f'pred_softmax_{class_ind}_F{fold}'] = softmax(predictions)[:,class_ind] 

        
df_test.to_csv(f'{model_path}/{baseline_name}_result_MSI_all.csv')

