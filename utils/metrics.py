import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
import torchmetrics


def compute_optimal_threshold(preds, true_labels):
    thresholds = np.linspace(0.01, 0.99, num=99)
    delta_th = 0.01
    f1_scores = [torchmetrics.functional.classification.binary_f1_score(
        preds, true_labels, threshold=th, validate_args=True) for th in thresholds]
    f1_scores = torch.stack(f1_scores)
    best_f1_score, th_index = torch.max(f1_scores, dim=0)
    return best_f1_score.item(), th_index.item() * delta_th + 0.01


def compute_accuracy_per_group(preds, true_labels, sensitive_attrs, threshold, num_groups):
    accuracies = []
    for group in range(num_groups):
        idx = (sensitive_attrs == group)
        group_preds = (preds[idx] > threshold).int()
        group_true = true_labels[idx]
        accuracies.append((group_preds == group_true).float().mean().item())
    return accuracies


def fpr(preds, true_labels, threshold):
    # [tp, fp, tn, fn, sup] (sup stands for support and equals tp + fn).
    stat_scores = torchmetrics.functional.classification.binary_stat_scores(
        preds, true_labels, threshold)
    fp = stat_scores[..., 1]
    tn = stat_scores[..., 2]
    return fp / (fp + tn)

def classification_metrics(preds, true_labels, threshold):
    # [tp, fp, tn, fn, sup] (sup stands for support and equals tp + fn).
    stat_scores = torchmetrics.functional.classification.binary_stat_scores(
        preds, true_labels, threshold)
    # compute all metrics that we need and return its as a dictionary
    outputs = {}
    tp = stat_scores[..., 0]
    fp = stat_scores[..., 1]
    tn = stat_scores[..., 2]
    fn = stat_scores[..., 3]
    outputs['accuracy'] = (tp + tn) / (tp + fp + tn + fn)
    outputs['fpr'] = fp / (fp + tn)
    outputs['tpr'] = tp / (tp + fn)
    outputs['fdr'] = fp / (tp + fp)
    return outputs
def compute_per_group_metrics(preds, true_labels, sensitive_attrs, threshold, num_groups):
    metrics = []
    for group in range(num_groups):
        idx = (sensitive_attrs == group)
        group_preds = preds[idx]
        group_true = true_labels[idx]
        group_metrics = classification_metrics(group_preds, group_true, threshold)
        metrics.append(group_metrics)
    return metrics
def compute_fpr_per_group(preds, true_labels, sensitive_attrs, threshold, num_groups):
    fprs = []
    for group in range(num_groups):
        idx = (sensitive_attrs == group)
        group_preds = preds[idx]
        group_true = true_labels[idx]
        fprs.append(fpr(group_preds, group_true, threshold).item())
    return fprs


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


class GroupBasedStats(nn.Module):
    def __init__(self, num_groups):
        super(GroupBasedStats, self).__init__()
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
            accuracies = compute_accuracy_per_group(
                y_preds, y_trues, group_attrbs[:, i], self.thres, self.num_groups[i])
            gap = max(accuracies) - min(accuracies)
            group_accs.append(accuracies)
            group_gaps.append(gap)
        return group_accs, group_gaps

    def computer_per_group_fpr(self):
        y_preds = torch.concat(self.y_preds, dim=0)
        y_trues = torch.concat(self.y_trues, dim=0)
        group_attrbs = torch.concat(self.group_attrbs, dim=0)
        group_fprs = []
        for i in range(group_attrbs.shape[-1]):
            fprs = compute_fpr_per_group(
                y_preds, y_trues, group_attrbs[:, i], self.thres, self.num_groups[i])
            group_fprs.append(fprs)
        return group_fprs
    
    def computer_per_group_classification_metrics(self):
        y_preds = torch.concat(self.y_preds, dim=0)
        y_trues = torch.concat(self.y_trues, dim=0)
        group_attrbs = torch.concat(self.group_attrbs, dim=0)
        group_metrics = []
        for i in range(group_attrbs.shape[-1]):
            metrics = compute_per_group_metrics(
                y_preds, y_trues, group_attrbs[:, i], self.thres, self.num_groups[i])
            group_metrics.append(metrics)
        return group_metrics
