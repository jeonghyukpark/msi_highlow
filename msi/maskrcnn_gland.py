
import os
import time
import random
import collections

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import os
import pickle
from tqdm import tqdm
import cv2


# Fix randomness

def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
fix_all_seeds(2021)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Normalize:
    def __call__(self, image, target):
        image = F.normalize(image, RESNET_MEAN, RESNET_STD)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image.copy())
        return image, target
    

def get_transform(train):
    transforms = [ToTensor()]
    NORMALIZE = False
    if NORMALIZE:
        transforms.append(Normalize())
    
    # Data augmentation for train
    if train: 
        transforms.append(HorizontalFlip(0.5))
        transforms.append(VerticalFlip(0.5))

    return Compose(transforms)


def get_model():
    NUM_CLASSES = 1+2
    BOX_DETECTIONS_PER_IMG = 100
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                              box_detections_per_img=BOX_DETECTIONS_PER_IMG)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, NUM_CLASSES)
    return model


def result_nms_pp(result, conf_thrs = 0.8, seg_ths = 0.5, nms_ths = 0.5):

    masks = result['masks'].cpu()
    labels = result['labels'].cpu()
    mask_scores_raw = np.asarray(result['scores'].cpu())
    mask_scores = mask_scores_raw>=conf_thrs
    
    masks = np.asarray(masks[mask_scores].cpu())
    labels = labels[mask_scores]
    mask_scores_raw = mask_scores_raw[mask_scores]

    len_pred = len(masks)
    buffer = np.zeros((len_pred, len_pred))
    class_ind = 0
    
    for ind_pred_1 in range(len_pred):
        for ind_pred_2 in range(ind_pred_1+1, len_pred):
            intersection = ((masks[ind_pred_1, class_ind]>=seg_ths) * (masks[ind_pred_2, class_ind]>=seg_ths)).sum()
            union = ((masks[ind_pred_1, class_ind]>=seg_ths) + (masks[ind_pred_2, class_ind]>=seg_ths)>=1).sum()
            buffer[ind_pred_1, ind_pred_2] = intersection/union


    buffer = buffer >= nms_ths

    overlapped_idx = []
    sorted_idx = list(np.argsort(mask_scores_raw)[::-1])
    for idx in sorted_idx:
        overlapped = np.where(buffer[idx] == 1)
        if len(overlapped[0]) > 0:
            #print(overlapped[0])
            overlapped_idx.extend(overlapped[0])

    nms_post_idx = np.delete(sorted_idx, overlapped_idx)
    if len(nms_post_idx) == 0:
        return [], [], []
    masks = masks[nms_post_idx]
    mask_scores_raw = mask_scores_raw[nms_post_idx]
    labels = labels[nms_post_idx]
    return masks, mask_scores_raw, labels

def metric_counter(masks_gt, masks_pred, class_ind = 0, seg_ths = 0.5):
    len_gt, len_pred = len(masks_gt), len(masks_pred)
    buffer = np.zeros((len_gt, len_pred))

    for ind_gt in range(len_gt):
        for ind_pred in range(len_pred):
            buffer[ind_gt, ind_pred] = (masks_gt[ind_gt] * masks_pred[ind_pred, class_ind]>=seg_ths).sum()/masks_gt[ind_gt].sum()

    TP = ((buffer.max(axis=1) >= 0.5) == 1).sum()
    FN = ((buffer.max(axis=1) >= 0.5) == 0).sum()
    FP = ((buffer.max(axis=0) >= 0.5) == 0).sum()
    return {'TP':TP, 'FN':FN, 'FP':FP}

class SingleImageDataset(Dataset):
    
    def __init__(self, imgs, transforms=None):
        self.transforms = transforms
        self.imgs = imgs
        #print(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image_name = img_path.split('/')[-1]
        image = cv2.imread(img_path)[:,:,::-1]
        if self.transforms is not None:
            image, _ = self.transforms(image=image, target=None)
        return {'image': image, 'image_name': image_name}

    def __len__(self):
        return len(self.imgs)

def inference(image_paths, model_path, device, save_dir=None):
    SAVE_NPZ = False

    DEVICE=device

    model = get_model()
    model.to(DEVICE)

    for param in model.parameters():
        param.requires_grad = True

    model.load_state_dict(torch.load(model_path))
    model.eval();


    image_sample_paths = ['./data/patch_sample/blk-TMIGEPVTWTVC-TCGA-AA-3837-01Z-00-DX1.png',
                          './data/patch_sample/blk-EYIAGQICRLWL-TCGA-CM-4743-01Z-00-DX1.png']

    simage = SingleImageDataset(image_sample_paths, transforms=get_transform(train=False))


    predictions = {}


    for sample in tqdm(simage):

        img = sample['image']
        img_id = sample['image_name'].split('.')[0]
        with torch.no_grad():
            result = model([img.to(DEVICE)])[0]
        masks, mask_scores_raw, labels = result_nms_pp(result)

        predictions[img_id] = {}
        predictions[img_id]['masks'] = np.asarray(masks)
        predictions[img_id]['scores'] = np.asarray(mask_scores_raw)
        predictions[img_id]['labels'] = np.asarray(labels)
        if SAVE_NPZ == True:
            np.savez(f'{save_dir}/{img_id}.npz', masks=np.asarray(masks), scores=np.asarray(mask_scores_raw), labels = np.asarray(labels))
    return predictions