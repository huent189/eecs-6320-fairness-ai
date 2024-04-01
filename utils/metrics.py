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

def fpr_optimality(true_labels, ys, lambda_factor, thresh, return_fpr=False):
    predictions = (ys > thresh).int()
    accuracy = get_overall_acc(true_labels, predictions)
    fpr_val = fpr(ys, true_labels, thresh)
    if not return_fpr:
        return (1 - accuracy) + lambda_factor * fpr_val
    else:
        return (1 - accuracy) + lambda_factor * fpr_val, fpr_val


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


    def get_optimal_threshold(self):
        y_preds = torch.concat(self.y_preds, dim=0)
        y_trues = torch.concat(self.y_trues, dim=0)
        best_f1_score, thres = compute_optimal_threshold(y_preds, y_trues)
        print('--')
        print(fpr(y_preds, y_trues, thres))
        print(best_f1_score)
        return best_f1_score, thres

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



class GroupBasedStatsVaryingThrs(nn.Module):
    def __init__(self):
        super().__init__()
        self.y_preds = []
        self.y_trues = []
        self.group_attrbs = []
        # This point to the attribute based on which we will determine the thresholds:
        self.group_atttb_idx = None
        self.thres = {}
        self.threhold_per_sample = None

    def forward(self, y_pred, y_true, group):
        self.y_preds.append(y_pred)
        self.y_trues.append(y_true)
        self.group_attrbs.append(group)

    def set_group_attrb_idx(self, at_idx):
        self.group_atttb_idx = at_idx

    def set_thres(self, th):
        self.thres = th
        self.set_thrshold_per_sample()
    
    def set_thrshold_per_sample(self):
        group_attrbs = torch.concat(self.group_attrbs, dim=0)
        self.threhold_per_sample = []
        for i in range(group_attrbs.shape[0]):
        # for i in range(40):
            value = self.thres[group_attrbs[i, self.group_atttb_idx].item()]
            # self.threhold_per_sample[i] = torch.tensor(value, device='cuda:0', dtype=torch.float32)
            self.threhold_per_sample.append(value)
            # print(self.thres[group_attrbs[i, self.group_atttb_idx].item()])
        self.threhold_per_sample = torch.tensor(self.threhold_per_sample, device='cuda:0', dtype=torch.float32)

    def get_acc_per_group_var_thrs(
        self, preds, true_labels, sensitive_attrs, thresholds, attr_idx
    ):
        accuracies = []
        for group in range(num_groups_per_attrb[attr_idx]):
            idx = sensitive_attrs[:, attr_idx] == group
            group_preds = torch.gt(preds[idx, 0], self.threhold_per_sample[idx]).int()
            group_true = true_labels[idx, 0]
            accuracies.append(torch.eq(group_preds, group_true).float().mean().item())
        return accuracies

    def get_fpr_per_group_var_thrs(
        self, preds, true_labels, sensitive_attrs, thresholds, attr_idx
    ):
        fprs = []
        for group in range(num_groups_per_attrb[attr_idx]):
            idx = sensitive_attrs[:, attr_idx] == group
            # print('----')
            group_preds = torch.gt(preds[idx, 0], self.threhold_per_sample[idx]).int()
            # print(self.threhold_per_sample[idx][:10])
            # print(group_preds[:10])
            group_true = true_labels[idx, 0]
            # print(group_true[:10])
            # print(torch.ne(group_preds[:10], group_true[:10]).int())
            # print(torch.ne(group_preds[:10], group_true[:10]).int() * group_preds[:10])
            # print((torch.ne(group_preds, group_true).int() * group_preds).sum().item())
            fpr = (torch.ne(group_preds, group_true).int() * group_preds).sum().item() / (group_true != 1).int().sum().item()            
            fprs.append(fpr)
        return fprs

    def get_all_stats(self):        
        y_preds = torch.concat(self.y_preds, dim=0)
        y_trues = torch.concat(self.y_trues, dim=0)
        group_attrbs = torch.concat(self.group_attrbs, dim=0)
        accuracy_gaps = []
        fpr_gaps =[]
        accuracies_all_groups = []
        fpr_all_groups = []
        for attr in range(group_attrbs.shape[-1]):
            accuracies = self.get_acc_per_group_var_thrs(
                y_preds, y_trues, group_attrbs, self.thres, attr
            )
            accuracies_all_groups.append(accuracies)
            accuracy_gaps.append(max(accuracies) - min(accuracies))
            fprs = self.get_fpr_per_group_var_thrs(
                y_preds, y_trues, group_attrbs, self.thres, attr
            )
            fpr_all_groups.append(fprs)
            fpr_gaps.append(max(fprs) - min(fprs))
        return accuracies_all_groups, accuracy_gaps, fpr_all_groups, fpr_gaps


    def get_per_group_stats(self, attr):
        y_preds = torch.concat(self.y_preds, dim=0)
        y_trues = torch.concat(self.y_trues, dim=0)
        group_attrbs = torch.concat(self.group_attrbs, dim=0)
        accuracies = self.get_acc_per_group_var_thrs(
            y_preds, y_trues, group_attrbs, self.thres, attr
        )
        fprs = self.get_fpr_per_group_var_thrs(
            y_preds, y_trues, group_attrbs, self.thres, attr
        )
        return accuracies, fprs

    # def compute_per_group_stats(self, attr):
    #     accuracies, fprs = self.get_per_group_stats(attr)
    #     acc_gap = max(accuracies) - min(accuracies)
    #     fpr_gap = max(fprs) - min(fprs)
    #     print(
    #         "Accuracy gap for attribute {attr}: {acc_gap}\n".format(
    #             attr=index_to_protected_group[attr], acc_gap=acc_gap
    #         )
    #     )
    #     print(
    #         "FPR gap for attribute {attr}: {fpr_gap}\n".format(
    #             attr=index_to_protected_group[attr], fpr_gap=fpr_gap
    #         )
    #     )


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
        thresholds = np.linspace(0.01, 0.99, num=100)
        best_loss = 9e10
        best_threshold = None
        best_fpr = None
        for thresh in thresholds:
            # Convert predictions to binary using threshold
            # binary_preds = (y_preds > thresh).int()
            loss, fpr = fpr_optimality(y_trues.cpu(), y_preds.cpu(), self.lambda_factor, thresh, return_fpr=True)
            if loss < best_loss:
                best_loss = loss
                best_threshold = thresh
                best_fpr = fpr
        assert best_threshold is not None, 'assertion error, no best threshold was found, change the initial value of the best loss.'
        print(best_fpr)
        return best_threshold

    def get_optimal_thresholds(self):
        thres = {}
        for group in self.y_preds.keys():
            y_preds = torch.concat(self.y_preds[group], dim=0)
            y_trues = torch.concat(self.y_trues[group], dim=0)
            thres[group] = self.get_optimal_threshold(y_preds, y_trues)
        return thres
