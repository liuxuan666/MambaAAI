# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 09:36:54 2025
"""
import numpy as np
import random
import torch
import os
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, \
     auc, precision_recall_curve

def Metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, y_score=y_pred)
    aupr_score = auc(lr_recall, lr_precision)
    auc_score = roc_auc_score(y_true=y_true, y_score=y_pred)
    acc_score = accuracy_score(y_true=y_true, y_pred=(y_pred > 0.5).astype('int'))
    f1_value = f1_score(y_true=y_true, y_pred=(y_pred > 0.5).astype('int'))
    
    return auc_score, aupr_score, f1_value, acc_score

def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

class AntibodyAntigenDataset(Dataset):
    def __init__(self, pair_list):
        self.pairs = pair_list  # [(ab_id, ag_id, label)]
    
    def __getitem__(self, idx):
        ab_id, ag_id, label = self.pairs[idx]
        ab_emb = np.load(f"Data/Pretrained_Cov/ab/{ab_id}.npy")  # (L_ab, 1280)
        ag_emb = np.load(f"Data/Pretrained_Cov/ag/{ag_id}.npy")  # (L_ag, 1280)
        return (
            torch.FloatTensor(ab_emb),
            torch.FloatTensor(ag_emb),
            torch.FloatTensor([int(label)])
        )
    
    def __len__(self):
        return len(self.pairs)
    
def custom_collate_fn(batch):
    ab_embs = torch.stack([item[0] for item in batch])
    ag_embs = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return ab_embs, ag_embs, torch.squeeze(labels)


# computer K attentions
def computer_attentions(delta, A, B, C):
    # delta = delta.cpu().detach()
    # A = A.cpu().detach()
    # B = B.cpu().detach()
    # C = C.cpu().detach()
    n = A.shape[1]
    b, l, d_in = delta.shape
    deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
    deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)
    C_expanded = C.unsqueeze(2)  # (B, L, D, N)
    K = torch.zeros((b, l, l), device=delta.device)
    for i in range(l):
        current_cum_prod = torch.ones((b, d_in, n), device=delta.device)
        for j in range(i, -1, -1):
            if j < i:
                current_cum_prod = current_cum_prod * deltaA[:, j]
                K[:, i, j] = torch.einsum('b d n, b d n -> b', 
                                C_expanded[:, i] * current_cum_prod, 
                                deltaB[:, j])
    return K.mean(dim=2)