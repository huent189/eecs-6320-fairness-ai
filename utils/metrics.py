import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
from datasets.mimic_dataset import num_groups_per_attrb, index_to_protected_group


def get_overall_acc(true_labels, predictions):
    return (predictions == true_labels).float().mean().item()

def get_overall_fnr(true_labels, predictions):
    return ((true_labels != predictions) * (true_labels)).float().mean().item()

def get_overall_fpr(true_labels, predictions):
    return ((true_labels != predictions) * (1 - true_labels)).float().mean().item()

def fpr_optimality(true_labels, predictions, lambda_factor):
    accuracy = get_overall_acc(true_labels, predictions)
    fpr = get_overall_fpr(true_labels, predictions)
    return (1 - accuracy) + lambda_factor * fpr


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


def compute_accuracy_per_group(
    preds, true_labels, sensitive_attrs, threshold, num_groups
):
    accuracies = []
    for group in range(num_groups):
        idx = sensitive_attrs == group
        group_preds = (preds[idx] > threshold).int()
        group_true = true_labels[idx]
        accuracies.append((group_preds == group_true).float().mean().item())
    return accuracies


def compute_fpr_per_group(
    preds, true_labels, sensitive_attrs, threshold, num_groups
):
    fprs = []
    for group in range(num_groups):
        idx = sensitive_attrs == group
        group_preds = (preds[idx] > threshold).int()
        group_true = true_labels[idx]
        fprs.append(((group_preds != group_true) * (1 - group_true)).float().mean().item())
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
        print("GAP")
        group_attrbs = torch.concat(self.group_attrbs, dim=0)
        for i in range(group_attrbs.shape[-1]):
            accuracies = compute_accuracy_per_group(
                y_preds, y_trues, group_attrbs[:, i], self.thres, self.num_groups[i]
            )
            fprs = compute_fpr_per_group(
                y_preds, y_trues, group_attrbs[:, i], self.thres, self.num_groups[i]
            )
            acc_gap = max(accuracies) - min(accuracies)
            fpr_gap = max(fprs) - min(fprs)
            print(f"group acc gap {i}: {acc_gap}\n", accuracies)
            print(f"group fpr gap {i}: {fpr_gap}\n", fprs)


class GroupBasedAccuracyVaryingThrs(nn.Module):
    def __init__(self):
        super().__init__()
        self.y_preds = []
        self.y_trues = []
        self.group_attrbs = []
        self.thres = {}

    def forward(self, y_pred, y_true, group):
        self.y_preds.append(y_pred)
        self.y_trues.append(y_true)
        self.group_attrbs.append(group)

    def set_thres(self, th):
        self.thres = th

    def get_acc_per_group_var_thrs(
        self, preds, true_labels, sensitive_attrs, thresholds, attr_idx
    ):
        accuracies = []
        for group in range(num_groups_per_attrb[attr_idx]):
            idx = sensitive_attrs == group
            group_preds = (preds[idx] > thresholds[group]).int()
            group_true = true_labels[idx]
            accuracies.append((group_preds == group_true).float().mean().item())
        return accuracies

    def get_fpr_per_group_var_thrs(
        self, preds, true_labels, sensitive_attrs, thresholds, attr_idx
    ):
        fprs = []
        for group in range(num_groups_per_attrb[attr_idx]):
            idx = sensitive_attrs == group
            group_preds = (preds[idx] > thresholds[group]).int()
            group_true = true_labels[idx]
            fprs.append(((group_preds != group_true) and (group_true == 0)).float().mean().item())
        return fprs


    def compute_per_group_acc(self, attr):
        y_preds = torch.concat(self.y_preds, dim=0)
        y_trues = torch.concat(self.y_trues, dim=0)
        print("GAP")
        group_attrbs = torch.concat(self.group_attrbs, dim=0)
        accuracies = self.get_acc_per_group_var_thrs(
            y_preds, y_trues, group_attrbs, self.thres, attr
        )
        fprs = self.get_fpr_per_group_var_thrs(
            y_preds, y_trues, group_attrbs, self.thres, attr
        )
        acc_gap = max(accuracies) - min(accuracies)
        fpr_gap = max(fprs) - min(fprs)
        print(
            "Accuracy gap for attribute {attr}: {acc_gap}\n".format(
                attr=index_to_protected_group[attr], acc_gap=acc_gap
            )
        )
        print(
            "FPR gap for attribute {attr}: {fpr_gap}\n".format(
                attr=index_to_protected_group[attr], fpr_gap=fpr_gap
            )
        )


class OptimalThresholdPerGroup(nn.Module):
    def __init__(self, lambda_factor=0.5):
        super().__init__()
        self.y_preds = {}
        self.y_trues = {}
        self.lambda_factor = lambda_factor

    def add_data(self, y_pred, y_true, group_idx):
        if group_idx in self.y_preds:
            self.y_preds[group_idx].append(y_pred)
            self.y_trues[group_idx].append(y_true)
        else:
            self.y_preds[group_idx] = [y_pred]
            self.y_trues[group_idx] = [y_true]

    def get_optimal_threshold(self, y_preds, y_trues):
        thresholds = np.linspace(0, 1, num=100)
        best_score = 0
        best_threshold = 0
        for thresh in thresholds:
            # Convert predictions to binary using threshold
            binary_preds = (y_preds > thresh).int()
            score = fpr_optimality(y_trues.cpu(), binary_preds.cpu(), self.lambda_factor)
            if score > best_score:
                best_score = score
                best_threshold = thresh
        return best_threshold

    def compute_optimal_thresholds(self):
        thres = {}
        for group in self.y_preds.keys():
            y_preds = torch.concat(self.y_preds[group], dim=0)
            y_trues = torch.concat(self.y_trues[group], dim=0)
            thres[group] = self.get_optimal_threshold(y_preds, y_trues)
        return thres
