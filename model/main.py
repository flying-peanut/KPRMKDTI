import sys
import math
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import  StratifiedKFold
import warnings
from tqdm import tqdm
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryAUROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from MKAN import KPRMKDTI, SDDataset

sys.path.append("./utils/")
from util import set_random_seed, calculate_metrics, load_feature

warnings.filterwarnings("ignore", category=Warning, module="torchvision")
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def Cross_verification(X, y, device, batch_size):
    roc_auc_scores = []
    pr_auc_scores = []
    precision_scores = []
    accuracy_scores = []
    recall_scores = []
    f1_scores = []
    specificity_scores = []
    pos_weight_list = []
    fold_idx = 0
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=47)
    for train_index, val_index in cv.split(X, y):
        fold_idx += 1
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 创建DataLoader
        train_dataset = SDDataset(X_train, y_train, device)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = SDDataset(X_val, y_val, device)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()

        pos_weight_list.append(num_neg / num_pos)
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model = KPRMKDTI(criterion)
        model.to(device)

        logger = TensorBoardLogger(
            save_dir="lightning_logs",
            name="KKAN",
            version=f"fold_{fold_idx}"
        )
        # 训练配置
        trainer = pl.Trainer(
            max_epochs=200,
            accelerator='gpu',
            accumulate_grad_batches=2,
            devices=1,
            precision='16-mixed',
            logger=logger,
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    monitor='val_auc',
                    dirpath=f"checkpoints/fold_{fold_idx}",
                    filename='best_model',
                    mode='max'
                ),
                pl.callbacks.EarlyStopping(
                    monitor='val_auc',
                    patience=15,
                    mode='max'
                )
            ]
        )
        
        # 训练与验证
        trainer.fit(model, train_loader, val_loader)
        
        # 记录最佳结果
        accuracy_scores.append(trainer.logged_metrics['val_acc'].item())
        precision_scores.append(trainer.logged_metrics['val_pre'].item())
        recall_scores.append(trainer.logged_metrics['val_re'].item())
        f1_scores.append(trainer.logged_metrics['val_f1'].item())
        specificity_scores.append(trainer.logged_metrics['val_sp'].item())
        roc_auc_scores.append(trainer.logged_metrics['val_auc'].item())
        pr_auc_scores.append(trainer.logged_metrics['val_ap'].item())

    return accuracy_scores, precision_scores, recall_scores, f1_scores, specificity_scores, roc_auc_scores, pr_auc_scores, pos_weight_list


def Independent_test(pos_weight_list, test_loader, device):
    
    for i in range(5):
        pos_weight = torch.tensor([pos_weight_list[i]], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        #初始化新模型
        lightning_model = KPRMKDTI.load_from_checkpoint(f'./checkpoints/fold_{i + 1}/best_model.ckpt', criterion = criterion)
        model = lightning_model.to(device)
        model.eval()

        predictions = []
        all_labels = []
            
        with torch.no_grad():# 此处不需要梯度计算
            for batch in test_loader:
                X, y = batch
                scores = model(X)
                preds = torch.sigmoid(scores)
                predictions.append(preds.cpu())
                all_labels.append(y.cpu())
        predictions = torch.cat(predictions).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        accuracy, precision, recall, f1, specificity, roc_auc, pr_auc = calculate_metrics(all_labels, predictions)
        print(f"=== Independent Test Results Fold{i} ===")
        print(f'accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, f1: {f1:.3f}, specificity: {specificity:.3f}, ROC AUC: {roc_auc:.3f}, PR AUC: {pr_auc:.3f}')




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_random_seed(47)
    batch_size = 32
    
    # drug_path = './dataset/new_featurized/k_bert'
    # target_path = './dataset/new_featurized/boruta'
    # drug_cols = [f'k_bert_{i}' for i in range(1, 769)]
    # target_cols = [f'Probert_{i}' for i in range(1, 513)]
    X_train, y_train, X_test, y_test = load_feature(drug_path, target_path, drug_cols, target_cols)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    test_dataset = SDDataset(X_test, y_test, device)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    accuracy_scores, precision_scores, recall_scores, f1_scores, specificity_scores, roc_auc_scores, pr_auc_scores, pos_weight_list = Cross_verification(X_train, y_train, device, batch_size)

    print(pos_weight_list)
    print("\n=== Cross Validation Results ===")
    print(f'accuracy: {sum(accuracy_scores) / len(accuracy_scores):.3f}, precision: {sum(precision_scores) / len(precision_scores):.3f}, recall: {sum(recall_scores) / len(recall_scores):.3f}, f1: {sum(f1_scores) / len(f1_scores):.3f}, specificity: {sum(specificity_scores) / len(specificity_scores):.3f}, ROC AUC: {sum(roc_auc_scores) / len(roc_auc_scores):.3f}, PR AUC: {sum(pr_auc_scores) / len(pr_auc_scores):.3f}')

    Independent_test(pos_weight_list, test_loader, device)

        
if __name__ == "__main__":
    main()

  
