import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle5 as pickle

import scipy

from sklearn import metrics
from sklearn.preprocessing import StandardScaler

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def preprocess_df(df):

    def quantile_25(series):
        return np.quantile(series, q=0.25)
    def quantile_50(series):
        return np.quantile(series, q=0.50)
    def quantile_75(series):
        return np.quantile(series, q=0.75)

    tissue_features = ['prob_ADI', 'prob_BACK', 'prob_DEB', 'prob_LYM', 
                       'prob_MUC', 'prob_MUS', 'prob_NORM', 'prob_STR', 'prob_TUM']
    cell_features = ['no-label', 'neoplastic', 'inflammatory', 'connective', 'necrosis', 'non-neoplastic']
    gland_features = ['benign_gland', 'malignant_gland']

    feature_cols =  tissue_features + cell_features + gland_features
    df_case = df.groupby(['case_id'])[feature_cols].agg(['mean', 'std', 'max', 'min', quantile_25,quantile_50,quantile_75]).reset_index()

    df_case.columns = ['_'.join(col) for col in df_case.columns]
    for idx in df_case.index:
        case_id = df_case.loc[idx,'case_id_']
        for col in ['MSS_or_MSI', 'fold', 'train_or_test', 'cohort']:
            col_val = df[df['case_id'] == case_id][col].iloc[0]
            df_case.loc[idx, col] = col_val
    return df_case, feature_cols

def inference(df_case, feature_cols, model_path):
    result_df = {}
    result_df['auc'] = []
    result_df['lambda'] = []
    result_df['learning_rate'] = []
    result_df['feature_fraction'] = []
    
    target_columns = []
    for prefix in feature_cols:
        for appendix in ['mean', 'std', 'max', 'min', 'quantile_25', 'quantile_50', 'quantile_75']:
            target_columns.append(f'{prefix}_{appendix}')

    with open(model_path, 'rb') as file:
        models, params, scalers = pickle.load(file)

    df_case_copy = df_case.copy()

    for ind, model in enumerate(models):
        X_all_transform = scalers[ind].transform(df_case[target_columns])
        y_pred = model.predict(X_all_transform, num_iteration=model.best_iteration)
        df_case_copy[f'pred_F{ind}'] = y_pred

    return df_case_copy


def evaluate(df_case, feature_cols, model_path):
    result_df = {}
    result_df['model_path'] = []
    result_df['auc'] = []
    result_df['lambda'] = []
    result_df['learning_rate'] = []
    result_df['feature_fraction'] = []
    
    target_columns = []
    for prefix in feature_cols:
        for appendix in ['mean', 'std', 'max', 'min', 'quantile_25', 'quantile_50', 'quantile_75']:
            target_columns.append(f'{prefix}_{appendix}')

    with open(model_path, 'rb') as file:
        models, params, scalers = pickle.load(file)

    df_case_copy = df_case.copy()
    df_case_copy['target']= df_case['MSS_or_MSI'] == 'MSI' 

    for ind, model in enumerate(models):
        X_all_transform = scalers[ind].transform(df_case[target_columns])
        y_pred = model.predict(X_all_transform, num_iteration=model.best_iteration)
        df_case_copy[f'pred_{ind}'] = y_pred

    aucs = []
    target_df = df_case_copy[(df_case_copy['cohort']=='CRC') & (df_case_copy['train_or_test'] == 'test')]
    for fold in range(5):

        y_valid = target_df[f'target']
        y_pred = target_df[f'pred_{fold}']

        fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        aucs.append(auc)

    cis = mean_confidence_interval(aucs)

    result_df['model_path'].append(model_path)
    result_df['auc'].append(f'{cis[0]:.4f} ({cis[1]:.4f}-{cis[2]:.4f})')
    result_df['lambda'].append(params['lambda_l1'])
    result_df['learning_rate'].append(params['learning_rate'])
    result_df['feature_fraction'].append(params['feature_fraction'])
    result_df=pd.DataFrame(result_df)
    return result_df