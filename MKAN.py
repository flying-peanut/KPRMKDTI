import sys
import torch
import warnings
import numpy as np
from math import log
import torch.nn as nn
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryAUROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from KAN import KANLinear

sys.path.append("./utils/")
from util import calculate_metrics

ALL_DIM = 1280      
MKFF_CHANNELS = 256 
pooled_length = 128

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,), dilation=(1,), if_bias=False,
                 same_padding=True, relu=True, bn=True):
        super(ConvBlock, self).__init__()
        p0 = 'same' if same_padding else 0

        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=p0, dilation=dilation, bias=True if if_bias else False)
        self.batchnorm1d = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv1d(x)
        if self.batchnorm1d is not None:
            x = self.batchnorm1d(x)
        if self.relu is not None:
            x = self.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        return x

class GLDF(nn.Module):
    def __init__(self, in_channels, hidden_channels, b=1, gamma=2, r=4):
        super(GLDF, self).__init__()

        self.conv1 = ConvBlock(in_channels, hidden_channels, kernel_size=1, stride=1, same_padding=False, relu=False,
                               bn=False)

        adp_kernel_size = int(abs((log(hidden_channels, 2) + b) / gamma))
        adp_kernel_size = adp_kernel_size if adp_kernel_size % 2 else adp_kernel_size + 1

        out_channel = hidden_channels // r
        self.conv_l1 = ConvBlock(hidden_channels, out_channel, kernel_size=adp_kernel_size, stride=1)
        self.conv_l2 = ConvBlock(out_channel, hidden_channels, kernel_size=adp_kernel_size, stride=1, relu=False)
        self.softmax_l = nn.Softmax(dim=1)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_g = ConvBlock(1, 1, kernel_size=adp_kernel_size, relu=False)
        self.sigmoid_g = nn.Sigmoid()

        self.conv2 = ConvBlock(hidden_channels * 2, in_channels, kernel_size=1, stride=1, same_padding=False,
                               relu=False, bn=False)

    def forward(self, x1, x2, x3, x4):
        x12 = torch.cat([x1, x2, x3, x4], dim=1)
        x12_fc = self.conv1(x12)

        # Local
        localF = self.conv_l1(x12_fc)
        localF = self.conv_l2(localF)
        localF = self.softmax_l(localF)  

        # Global
        avg_pool = self.avg_pool(x12_fc)
        globalF = self.conv_g(avg_pool.transpose(-1, -2)).transpose(-1, -2)
        globalF = self.sigmoid_g(globalF)  

        localFNew = x12_fc * localF.expand_as(x12_fc) 
        globalFNew = x12_fc * globalF.expand_as(x12_fc) 

        output = torch.cat([localFNew, globalFNew], dim=1)
        output = self.conv2(output)

        return output


class MKFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MKFF, self).__init__()

        self.kernel1 = ConvBlock(in_channel, out_channel, kernel_size=1, same_padding=False)

        self.kernel2 = nn.Sequential(ConvBlock(in_channel, out_channel, kernel_size=1, same_padding=False, bn=False),
                                     ConvBlock(out_channel, out_channel, kernel_size=3))

        self.kernel3 = nn.Sequential(ConvBlock(in_channel, out_channel, kernel_size=1, same_padding=False),
                                     ConvBlock(out_channel, out_channel, kernel_size=5))

        self.kernel4 = nn.Sequential(nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                                     ConvBlock(in_channel, out_channel, kernel_size=1, same_padding=False))

        self.GLDF = GLDF(out_channel * 4, out_channel)

    def forward(self, x):
        kernel1 = self.kernel1(x)
        kernel2 = self.kernel2(x)
        kernel3 = self.kernel3(x)
        kernel4 = self.kernel4(x)

        kernel_all = self.GLDF(kernel1, kernel2, kernel3, kernel4)

        return kernel_all + x

class MKANDTI(nn.Module):
    def __init__(self, all_dim, mkff_channels=128, pooled_length = 128):

        super(MKANDTI, self).__init__()

        self.total_dim = all_dim
        self.pool = nn.AdaptiveAvgPool1d(output_size=pooled_length)
        self.stem = ConvBlock(1, mkff_channels, kernel_size=3)

        self.mkff = MKFF(in_channel=mkff_channels, out_channel=mkff_channels // 4)

        self.kan_classifier = nn.Sequential(
            nn.Flatten(),
            KANLinear(mkff_channels * pooled_length, 128, wavelet_type='dog'),
            nn.Dropout(0.3),
            KANLinear(128, 64, wavelet_type='dog'),
            nn.Dropout(0.3),
            KANLinear(64, 1, wavelet_type='dog')
        )


    def forward(self, features):

        x = features
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.mkff(x)
        x = self.pool(x)

        output = self.kan_classifier(x)
        
        return output




class SDDataset(Dataset): 
    def __init__(self, X, y, device):
        self.X = X.to(device)  
        self.y = y.to(device)
 
    def __len__(self):
        return len(self.X)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class KPRMKDTI(pl.LightningModule):
    def __init__(self, criterion):
        super().__init__()
        self.model = MKANDTI(
            all_dim=ALL_DIM,
            mkff_channels=MKFF_CHANNELS,
            pooled_length = pooled_length
        )
        self.save_hyperparameters()

        self.loss_fn = criterion

        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, feature):
        return self.model(feature)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        preds = torch.sigmoid(outputs)
        loss = self.loss_fn(outputs, labels)
        self.validation_outputs.append({'preds': preds, 'labels': labels})
        return {'val_loss': loss, 'preds': preds, 'labels': labels}

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_outputs]).cpu().numpy()
        labels = torch.cat([x['labels'] for x in self.validation_outputs]).cpu().numpy()

        fpr, tpr, thresholds = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Receiver Operating Characteristic (Epoch {self.current_epoch})')
        ax.legend(loc="lower right")
        
        tensorboard_logger = self.logger.experiment
        tensorboard_logger.add_figure('ROC_Curve', fig, self.current_epoch)
        plt.close(fig) 


        accuracy, precision, recall, f1, specificity, roc_auc, pr_auc = calculate_metrics(labels, preds)

        metrics = {
            'val_auc': roc_auc,
            'val_ap': pr_auc,
            'val_acc': accuracy,
            'val_pre': precision,
            'val_re': recall,
            'val_f1': f1,
            'val_sp': specificity
        }
        self.log_dict(metrics, prog_bar=True)
        
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        preds = torch.sigmoid(outputs)
        self.test_outputs.append({'preds': preds, 'labels': labels})
        return {'preds': preds,'labels': labels}

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_outputs]).cpu().numpy()
        labels = torch.cat([x['labels'] for x in self.test_outputs]).cpu().numpy()
        accuracy, precision, recall, f1, specificity, roc_auc, pr_auc = calculate_metrics(labels, preds)
        # 计算指标
        metrics = {
            'test_auc': roc_auc,
            'test_ap': pr_auc,
            'test_acc': accuracy,
            'test_pre': precision,
            'test_re': recall,
            'test_f1': f1,
            'test_sp': specificity
        }
        
        self.log_dict(metrics, prog_bar=True)
        
        self.test_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2.5e-3, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]