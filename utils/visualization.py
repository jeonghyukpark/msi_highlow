import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cv2

from pathlib import Path

def plot(ax, df, idx, patch_sample_base = './data/patch_sample', alpha=0, fontsize=12):
    
    cell_labels = ['no-label', 'neoplastic', 'inflammatory', 'connective', 'necrosis', 'non-neoplastic']
    tissue_labels = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    gland_labels = ['benign_gland', 'malignant_gland']
    
    gland_colors = {1:(246,232,195), 2:(128,205,193)}
    # tissue_colors = {'LYM':[30,148,189], 'ADI':[141,141,141], 'BACK':[53,53,53], 'DEB': [186,39,154],
    #       'MUC':[252,215,203], 'MUS':[67,172,34], 'NORM':[200,103,1], 'STR':[245,184,9],
    #       'TUM':[231,71,23]}
    tissue_colors = {'LYM': (0.11764705882352941, 0.5803921568627451, 0.7411764705882353),
                     'ADI': (0.5529411764705883, 0.5529411764705883, 0.5529411764705883),
                     'BACK': (0.20784313725490197, 0.20784313725490197, 0.20784313725490197),
                     'DEB': (0.7294117647058823, 0.15294117647058825, 0.6039215686274509),
                     'MUC': (0.9882352941176471, 0.8431372549019608, 0.796078431372549),
                     'MUS': (0.2627450980392157, 0.6745098039215687, 0.13333333333333333),
                     'NORM': (0.7843137254901961, 0.403921568627451, 0.00392156862745098),
                     'STR': (0.9607843137254902, 0.7215686274509804, 0.03529411764705882),
                     'TUM': (0.9058823529411765, 0.2784313725490196, 0.09019607843137255)}

    patch_name = str(Path(df.loc[idx, 'filename']).stem)
    
    tissue_argmax = df.loc[idx, 'tissue_type_ind']
    cell_overlay_path = f'{patch_sample_base}/cell_overlay_{patch_name}.png'
    gland_npz_path = f'{patch_sample_base}/gland_{patch_name}.npz'
    with np.load(gland_npz_path, 'rb') as a:
        masks = a['masks']
        labels = a['labels']
    
    cell_overlay_img = cv2.imread(cell_overlay_path)
    cell_overlay_img = cv2.cvtColor(cell_overlay_img, cv2.COLOR_BGR2RGB)
    cell_overlay_img = cv2.resize(cell_overlay_img, (224,224), interpolation = cv2.INTER_AREA)

    for mask_ind in range(len(masks)):
        mask_bin = 255*(masks[mask_ind][0]>0.5).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        gland_class = labels[mask_ind]
        for cnt in contours:
            cv2.drawContours(cell_overlay_img, [cnt], 0,  gland_colors[gland_class], 5)  # blue

    tissue_label = tissue_labels[tissue_argmax]
    ax.imshow(cell_overlay_img)
    ax.text(4, -8, tissue_label, color='w', backgroundcolor=tissue_colors[tissue_label], fontsize=fontsize)