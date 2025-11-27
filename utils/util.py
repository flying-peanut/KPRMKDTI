import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler, Sampler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import random

def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    if (tn + fp) == 0:
        return 1.0  
    return tn / (tn + fp)


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     


# ------------------- Metrics and Logging -------------------
def calculate_metrics(labels, preds):
    roc_auc =  roc_auc_score(labels, preds)
    pr_auc = average_precision_score(labels, preds)
    accuracy = accuracy_score(labels, (preds > 0.5).astype(int))
    precision = precision_score(labels, (preds > 0.5).astype(int))
    recall = recall_score(labels, (preds > 0.5).astype(int))
    f1 = f1_score(labels, (preds > 0.5).astype(int))
    specificity = specificity_score(labels, (preds > 0.5).astype(int))
    return accuracy, precision, recall, f1, specificity, roc_auc, pr_auc

def get_F(df, length):
    row_E = []
    for i in range(length):
        t = 'Feature_' + str(i + 1)
        row_E.append(t)
    df_features = []
    for index, row in df.iterrows():
        feature = row[row_E].values.tolist()  
        df_features.append(feature)
    return df_features

def load_featurized_data(path, length):
    df = pd.read_csv(path)
    df_label = df['Label'].tolist()
    df_feature = get_F(df, length)
    return df_feature, df_label

def standard_feature(path, cols):
    train_path = path + '/train.csv'
    test_path = path + '/test.csv'
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    train = df_train[cols].values.tolist()
    test = df_test[cols].values.tolist()
    scaler = StandardScaler()
    standarded_train = scaler.fit_transform(train).tolist()
    standarded_test = scaler.transform(test).tolist()
    return standarded_train, standarded_test

def load_feature(path_drug, path_target, drug_cols, target_cols):
    drug_train, drug_test = standard_feature(path_drug, drug_cols)
    target_train, target_test = standard_feature(path_target, target_cols)
    path_train = path_drug + '/train.csv'
    path_test = path_drug + '/test.csv'
    df_train = pd.read_csv(path_train)
    y_train = df_train['Label'].values.tolist()
    df_test = pd.read_csv(path_test)
    y_test = df_test['Label'].values.tolist()
    X_train = []
    X_test = []
    for i in range(len(drug_train)):
        x = drug_train[i] + target_train[i]
        X_train.append(x)
    for i in range(len(drug_test)):
        x = drug_test[i] + target_test[i]
        X_test.append(x)
    return X_train, y_train, X_test, y_test


# pytorch datalaoder
class MyDataset(Dataset):
    def __init__(self, drug, target, y, drug_id, target_id):
        self.drug = drug
        self.target = target
        self.drug_id = drug_id
        self.target_id = target_id
        self.y = y
        
    def __getitem__(self, index):
        label = self.y[index]
        drug_index = self.drug_id[index]
        target_index = self.target_id[index]
        x1 = self.drug[drug_index]
        x2 = self.target[target_index]
        return torch.tensor(x1).float(), torch.tensor(x2).float(), torch.tensor(label).float(), drug_index, target_index

    def __len__(self):
        return len(self.y)



def load_encodered(path, drug_cols, target_cols):
    drug_path = path + '/drugs.csv'
    target_path = path + '/targets.csv'
    interaction_path = path + '/interactions.csv'
    df_drug = pd.read_csv(drug_path)
    df_target = pd.read_csv(target_path)
    df_interaction = pd.read_csv(interaction_path)
    drugs =  df_drug[drug_cols].values.tolist()
    targets = df_target[target_cols].values.tolist()
    labels = df_interaction['label'].values.tolist()
    drug_idx = df_interaction['drug_id'].values.tolist()
    target_idx = df_interaction['target_id'].values.tolist()
    dataset = MyDataset(drugs, targets, labels, drug_idx, target_idx)
    return dataset

def load_dataset(path, drug_cols, target_cols):
    train_path = path + '/train'
    test_path = path + '/test'
    train_dataset = load_encodered(train_path, drug_cols, target_cols)
    test_dataset = load_encodered(test_path, drug_cols, target_cols)
    return train_dataset, test_dataset

