import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
def compute_optimal_threshold(preds, true_labels):
    thresholds = np.linspace(0, 1, num=100)
    best_f1 = 0
    best_threshold = 0
    for thresh in thresholds:
        # Convert predictions to binary using threshold
        binary_preds = (preds > thresh).int()
        # Compute F1 score
        f1 = f1_score(true_labels.cpu(), binary_preds.cpu(), zero_division=1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    return best_threshold
def compute_accuracy_per_group(preds, true_labels, sensitive_attrs, threshold, num_groups):
    accuracies = []
    for group in range(num_groups):
        idx = (sensitive_attrs == group)
        group_preds = (preds[idx] > threshold).int()
        group_true = true_labels[idx]
        accuracies.append((group_preds == group_true).float().mean().item())
    return accuracies
class OptimalThresholdSelector(nn.Module):
    def __init__(self):
        super(OptimalThresholdSelector, self).__init__()
        self.y_preds = []
        self.y_trues = []
        
    def add_data(self, y_pred, y_true):
        self.y_preds.append(y_pred)
        self.y_trues.append(y_true)
    def compute_optimal_threshold(self):
        y_preds = torch.concat(self.y_preds, dim=0)
        y_trues = torch.concat(self.y_trues, dim=0)
        thres = compute_optimal_threshold(y_preds, y_trues)
        return thres
    
class GroupBasedAccuracy(nn.Module):
    def __init__(self, num_groups):
        super(GroupBasedAccuracy, self).__init__()
        self.y_preds = []
        self.y_trues = []
        self.group_attrbs = []
        self.num_groups = num_groups
        self.thres = None
    def forward(self, y_pred, y_true, group):
        self.y_preds.append(y_pred)
        self.y_trues.append(y_true)
        self.group_attrbs.append(group)
    def set_thres(self, th):
        self.thres = th
    def computer_per_group_acc(self):
        y_preds = torch.concat(self.y_preds, dim=0)
        y_trues = torch.concat(self.y_trues, dim=0)
        group_attrbs = torch.concat(self.group_attrbs, dim=0)
        group_accs = []
        group_gaps = []
        for i in range(group_attrbs.shape[-1]):
            accuracies = compute_accuracy_per_group(y_preds, y_trues, group_attrbs[:, i], self.thres, self.num_groups[i])
            gap = max(accuracies) - min(accuracies)
            group_accs.append(accuracies)
            group_gaps.append(gap)
        return group_accs, group_gaps
            