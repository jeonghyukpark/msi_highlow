import numpy as np
import pandas as pd
import torch
from efficientnet_pytorch import EfficientNet
import random
import os
import cv2
from tqdm import tqdm

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
    
    def __init__(self, image_paths, num_classes, targets, transform=None): 
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
        if self.targets:
            targets = self.targets[item]
            return {
                "image":image.permute(2, 0, 1).contiguous(),
                "targets": F.one_hot(torch.tensor(targets), num_classes = self.num_classes),
            }
        else:
            return {
                "image":image.permute(2, 0, 1).contiguous()
            }      

def inference(data_loader, model, device):

    model.eval()
    predictions = []
    with torch.no_grad():
        for data in tqdm(data_loader, position=0, leave=True, desc='Inference'):
            inputs = data["image"].to(device, dtype=torch.float)
            outputs = model(inputs)
            outputs = outputs.detach().cpu().numpy().tolist()
            predictions.extend(outputs)

    return predictions


 

def worker_init_fn(worker_id):
    imgaug.seed(np.random.get_state()[1][0] + worker_id)
    
    
def run_msi(image_paths, CFG, device):

    random_state = set_seed(2021)   
    baseline_name = CFG['baseline'] + '_' + CFG['model_unique_identifier']
    Batch_Size = CFG['batch_size']
    num_classes = CFG['num_classes']
    model_unique_name = CFG['model_unique_name']
    model_dir = CFG['model_dir']


    if CFG['baseline'] == 'efficientnet':
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        print('MODEL LOADED : efficientnet-b0')

    df_result = {}
    df_result['path'] = image_paths

    for fold in range(CFG['fold']):

        df_result[f'pred_F{fold}'] =[]
        PATH = f'{model_dir}/{model_unique_name}_{baseline_name}_F{fold}.pt'

        model.to(device)
        model.load_state_dict(torch.load(PATH))
        model.eval()

        dataset = ClassificationDataset(image_paths=image_paths, targets=None, num_classes=num_classes)
        loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_Size, shuffle=False, num_workers=8)
        predictions = inference(data_loader=loader, model=model, device=device)

        df_result[f'pred_F{fold}'].extend(softmax(np.asarray(predictions))[:,1]) #MSI score

    df_result = pd.DataFrame(df_result)
    return df_result

def run_tissue(image_paths, CFG, device):
    
    random_state = set_seed(2021)   
    baseline_name = CFG['baseline'] + '_' + CFG['model_unique_identifier']
    Batch_Size = CFG['batch_size']
    num_classes = CFG['num_classes']
    model_unique_name = CFG['model_unique_name']
    model_dir = CFG['model_dir']


    if CFG['baseline'] == 'efficientnet':
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        print('MODEL LOADED : efficientnet-b0')


    image_sample_paths = ['./data/patch_sample/blk-TMIGEPVTWTVC-TCGA-AA-3837-01Z-00-DX1.png',
                          './data/patch_sample/blk-EYIAGQICRLWL-TCGA-CM-4743-01Z-00-DX1.png']

    df_result = {}
    df_result['path'] = image_sample_paths
    tissue_labels = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

    for fold in range(CFG['fold']):
        for tissue_label in tissue_labels:    
            df_result[f'prob_{tissue_label}_F{fold}'] =[]
        PATH = f'{model_dir}/{model_unique_name}_{baseline_name}_F{fold}.pt'

        model.to(device)
        model.load_state_dict(torch.load(PATH))
        model.eval()

        dataset = ClassificationDataset(image_paths=image_sample_paths, targets=None, num_classes=num_classes)
        loader = torch.utils.data.DataLoader(dataset, batch_size=Batch_Size, shuffle=False, num_workers=8)
        predictions = inference(data_loader=loader, model=model, device=device)
        predictions = softmax(np.asarray(predictions))
        for ind, tissue_label in enumerate(tissue_labels):
            df_result[f'prob_{tissue_label}_F{fold}'].extend(predictions[:,ind]) #MSI score

    df_result = pd.DataFrame(df_result)
    for tissue_label in tissue_labels:
        df_result[f'prob_{tissue_label}'] = df_result[[f'prob_{tissue_label}_F{fold}' for fold in range(CFG['fold'])]].mean(axis=1)

    tissue_type_inds = list(np.argmax(np.asarray(df_result[[f'prob_{tissue_label}' for tissue_label in tissue_labels]]), axis=1))
    df_result['tissue_type_ind'] = tissue_type_inds
    df_result['tissue_type'] = df_result['tissue_type_ind'].apply(lambda x: tissue_labels[x])
    return df_result