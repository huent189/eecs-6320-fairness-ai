import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import numpy as np
import torchmetrics


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


# TODO: method signature has changed, update usage accordingly!
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
        idx = sensitive_attrs == group
        group_preds = (preds[idx] > threshold).int()
        group_true = true_labels[idx]
        accuracies.append((group_preds == group_true).float().mean().item())
    return accuracies

# TODO: check that this is FPR for No finding lables
def fpr(preds, true_labels, threshold):
    # [tp, fp, tn, fn, sup] (sup stands for support and equals tp + fn).
    stat_scores = torchmetrics.functional.classification.binary_stat_scores(
        preds, true_labels, threshold)
    fp = stat_scores[..., 1]
    tn = stat_scores[..., 2]
    return fp / (fp + tn)


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

# TODO: Class signature changed, it was GroupBasedAccuracy before, adjust accordingly!
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
