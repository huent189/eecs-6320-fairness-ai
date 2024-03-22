import torch
from torch import nn
import torch.optim as optim
from lightning.pytorch.core.module import LightningModule
import torchmetrics
from datasets.mimic_dataset import num_groups_per_attrb, protected_group_to_index, group_labels, disease_labels, NO_FINDING_INDEX, ATTRB_LABELS
from utils.metrics import GroupBasedStats, OptimalThresholdSelector, OptimalThresholdPerGroup, GroupBasedAccuracyVaryingThrs
import copy
from losses.gce import GeneralizedCELoss
import wandb

import torch_optimizer

class MIMICTrainer(LightningModule):
    def __init__(
        self,
        model,
        learning_rate=0.001,
        end_lr_factor=1.0,
        weight_decay=0.0,
        decay_steps=1000,
    ):
        super(MIMICTrainer, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_steps = decay_steps
        self.end_lr_factor = end_lr_factor
        # For binary classification, adjust if necessary
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.acc = torchmetrics.classification.MultilabelAccuracy(
            num_labels=len(disease_labels))
        self.average_auroc = torchmetrics.classification.MultilabelAUROC(
            num_labels=len(disease_labels))
        self.per_label_auroc = torchmetrics.classification.MultilabelAUROC(
            num_labels=len(disease_labels), average='none')
        self.group_metrics = GroupBasedStats(num_groups_per_attrb)
        self.group_labels = group_labels
        self.optimal_thres_selector = OptimalThresholdSelector()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self(x)
        loss = self.loss(y_hat, y).mean()
        self.log_dict({'train/loss': loss})
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]

        # implement your own
        out = self(x)
        loss = self.loss(out, y).mean()
        # calculate acc
        out = torch.sigmoid(out)
        y = y.int()
        val_acc = self.acc(out, y)
        self.average_auroc.update(out, y)
        # log the outputs!
        self.log_dict({'val/loss': loss, 'val/acc': val_acc})

    def on_validation_end(self):
        super().on_validation_end()
        self.logger.experiment.log({'val/auroc': self.average_auroc.compute()})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch[:2]
        y = y.int()
        # implement your own
        out = self(x)
        out = torch.sigmoid(out)
        # calculate acc
        avg_acc = self.acc(out, y)

        # log the outputs!
        if dataloader_idx == 0:
            self.optimal_thres_selector.add_data(
                out[:, NO_FINDING_INDEX, None], y[:, NO_FINDING_INDEX, None])
            self.average_auroc.update(out, y)
            self.log_dict({'val/final_acc': avg_acc})
        else:
            self.group_metrics(out[:, NO_FINDING_INDEX, None],
                               y[:, NO_FINDING_INDEX, None], batch[2])
            self.per_label_auroc.update(out, y)
            self.log_dict({'test/acc': avg_acc})

    def on_test_end(self):
        super().on_test_end()

        self.logger.experiment.log({'val/auroc': self.average_auroc.compute()})
        test_per_label_auroc = self.per_label_auroc.compute()
        # First column is disease name, second column is auroc
        table_data = [[d, test_per_label_auroc[i].item()]
                      for i, d in enumerate(disease_labels)]
        table_data.append(['Average', test_per_label_auroc.mean().item()])
        self.logger.experiment.log(
            {"test/per_label_auroc": wandb.Table(data=table_data, columns=["disease", "auroc"])})

        best_f1_score, th = self.optimal_thres_selector.compute_optimal_threshold()
        self.logger.experiment.log({'test/optimal_threshold': th,
                                    'test/f1_score': best_f1_score})

        self.group_metrics.set_thres(th)
        group_fprs = self.group_metrics.computer_per_group_fpr()
        # First column is attb, second column is fpr, third column is fpr_gap
        table_data = []
        for i, group in enumerate(group_fprs):
            mu_fpr = sum(group) / len(group)
            fpr_plot_per_attb = []
            attb = ATTRB_LABELS[i]
            for j, fpr in enumerate(group):
                sub_group_label = group_labels[i][j]
                table_data.append([sub_group_label, fpr, fpr - mu_fpr])
                fpr_plot_per_attb.append([sub_group_label, fpr])
            table_data.append(
                [f'Average fpr of {attb} attribute', mu_fpr, -1])
            fpr_per_attb_table = wandb.Table(data=fpr_plot_per_attb, columns=[
                                             "Label", "FPR"])
            self.logger.experiment.log(
                {f'test/{attb}': wandb.plot.bar(fpr_per_attb_table, 'Label', 'FPR', title=f'FPR for {attb}')})
        self.logger.experiment.log(
            {"test/fpr_summary": wandb.Table(data=table_data, columns=["Attribute", "fpr", "fpr_gap"])})

    def configure_optimizers(self):
        optimizer = torch_optimizer.Lamb(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, verbose=True, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


class BiasCouncilTrainer(MIMICTrainer):
    def __init__(self, model, bias_model, num_bias_models=1, learning_rate=0.001, end_lr_factor=1.0, weight_decay=0.0, decay_steps=1000):
        super(BiasCouncilTrainer, self).__init__(
            model, learning_rate, end_lr_factor, weight_decay, decay_steps)
        self.bias_councils = nn.ModuleList(
            [copy.deepcopy(bias_model) for _ in range(num_bias_models)])
        for m in self.bias_councils:
            m.init_weights()
        self.gce = GeneralizedCELoss()
        self.num_bias_models = num_bias_models

    def training_step(self, batch, batch_idx):
        if not isinstance(batch[0], list):
            batch = [batch for _ in range(self.num_bias_models)]
        biased_loss = 0
        unbiased_loss = 0
        for i in range(self.num_bias_models):
            x, y = batch[i][:2]
            biased_y_hat = self.bias_councils[i](x)
            biased_loss = self.gce(biased_y_hat, y).mean() + biased_loss

        p_y_hat = None
        x, y = batch[0][:2]
        unbiased_y_hat = self.model(x)
        # ensemble prediction
        for i in range(self.num_bias_models):
            biased_y_hat = self.bias_councils[i](x)
            biased_y_hat = biased_y_hat.detach()
            p_by = torch.sigmoid(biased_y_hat)
            p_by = y * p_by + (1 - y) * (1 - p_by)
            if p_y_hat is None:
                p_y_hat = p_by
            else:
                p_y_hat = torch.max(p_y_hat, p_by)
        unbiased_loss = unbiased_loss + self.loss(unbiased_y_hat, y) * p_y_hat
        unbiased_loss = unbiased_loss + \
            (1 - p_y_hat) * self.loss(unbiased_y_hat, 1 - p_y_hat)
        unbiased_loss = unbiased_loss.mean()
        loss = biased_loss * 5 + unbiased_loss
        self.log_dict({'biased_loss': biased_loss,
                      'unbiased_loss': unbiased_loss, 'loss': loss})
        return loss

    def configure_optimizers(self):
        params = list(self.model.parameters())
        for m in self.bias_councils:
            params += list(m.parameters())
        optimizer = torch.optim.Adam(
            params, lr=self.learning_rate, weight_decay=self.weight_decay, capturable=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True, factor=0.5)
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler,
                                 'monitor': 'val/loss', }}


class CEWeightedBiasCouncilTrainer(BiasCouncilTrainer):
    def training_step(self, batch, batch_idx):
        if not isinstance(batch[0], list):
            batch = [batch for _ in range(self.num_bias_models)]
        biased_loss = 0

        for i in range(self.num_bias_models):
            x, y = batch[i][:2]
            biased_y_hat = self.bias_councils[i](x)
            biased_loss = self.gce(biased_y_hat, y).mean() + biased_loss

        x, y = batch[0][:2]
        unbiased_y_hat = self.model(x)
        # ensemble prediction
        ce_biased_loss = 0
        for i in range(self.num_bias_models):
            biased_y_hat = self.bias_councils[i](x)
            biased_y_hat = biased_y_hat.detach()
            ce_biased_loss = ce_biased_loss + self.loss(biased_y_hat, y)
        unbiased_loss = self.loss(unbiased_y_hat, y)
        unbiased_loss = unbiased_loss * ce_biased_loss / \
            (ce_biased_loss + unbiased_loss.detach())
        unbiased_loss = unbiased_loss.mean()
        loss = biased_loss + unbiased_loss
        self.log_dict({'biased_loss': biased_loss,
                      'unbiased_loss': unbiased_loss, 'loss': loss})
        return loss


class SBS_THR_Trainer(LightningModule):
    """
        Implements using SBS and varying thresholds at the same time. 
    """
    def __init__(
        self,
        model,
        learning_rate=0.001,
        end_lr_factor=1.0,
        weight_decay=0.0,
        decay_steps=1000,
        protected_attribute='gender'
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_steps = decay_steps
        self.end_lr_factor = end_lr_factor
        self.loss = (
            nn.BCEWithLogitsLoss()
        )  # For binary classification, adjust if necessary
        self.acc = torchmetrics.Accuracy(task="binary")
        self.auroc = torchmetrics.AUROC(task="binary")
        self.group_acc = GroupBasedAccuracyVaryingThrs()
        self.optimal_thres_selector = OptimalThresholdPerGroup()
        self.protected_attribute = protected_attribute
        self.protected_attribute_idx = protected_group_to_index[self.protected_attribute]


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_dict({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]

        # implement your own
        out = self(x)
        loss = self.loss(out, y)
        out = torch.sigmoid(out)
        # calculate acc
        val_acc = self.acc(out, y)
        val_auc = self.auroc(out, y)
        # log the outputs!
        self.log_dict({"val_loss": loss, "val_acc": val_acc, "val_auc": val_auc})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch[:2]
        out = self(x)
        # calculate acc
        out = torch.sigmoid(out)
        # calculate acc
        test_acc = self.acc(out, y)
        test_auc = self.auroc(out, y)
        # log the outputs!
        self.log_dict({"test_acc": test_acc, "test_auc": test_auc})
        if dataloader_idx == 0:
            self.group_acc(out, y, batch[2][:, self.protected_attribute_idx])
        else:
            # dataloader idx for validation sets start from 1, since the first 
            # one is the actual test set, we need the actual group index to retrive the
            # gruop name later on, so we substract idx by one to get the actual value! 
            self.optimal_thres_selector.add_data(out, y, dataloader_idx - 1)

    def on_test_end(self):
        super().on_test_end()
        # Get different data groups             |  X
        # For each data group create a dataset  |  X
        # Find best threshold for each dtaset   | [ ]
        # Measure metrics                       | [ ]
        ths = self.optimal_thres_selector.compute_optimal_thresholds()
        self.group_acc.set_thres(ths)
        self.group_acc.compute_per_group_acc(self.protected_attribute_idx)


    def configure_optimizers(self):
        optimizer = torch_optimizer.Lamb(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, verbose=True, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }



class SBS_MIMICTrainer(LightningModule):
    def __init__(
        self,
        model,
        learning_rate=0.001,
        end_lr_factor=1.0,
        weight_decay=0.0,
        decay_steps=1000,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_steps = decay_steps
        self.end_lr_factor = end_lr_factor
        self.loss = (
            nn.BCEWithLogitsLoss()
        )  # For binary classification, adjust if necessary
        self.acc = torchmetrics.Accuracy(task="binary")
        self.auroc = torchmetrics.AUROC(task="binary")
        self.group_acc = GroupBasedStats(num_groups_per_attrb)
        self.optimal_thres_selector = OptimalThresholdSelector()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_dict({"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]

        # implement your own
        out = self(x)
        loss = self.loss(out, y)
        out = torch.sigmoid(out)
        # calculate acc
        val_acc = self.acc(out, y)
        val_auc = self.auroc(out, y)
        # log the outputs!
        self.log_dict({"val_loss": loss, "val_acc": val_acc, "val_auc": val_auc})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch[:2]

        # implement your own
        out = self(x)
        # calculate acc
        out = torch.sigmoid(out)
        # calculate acc
        test_acc = self.acc(out, y)
        test_auc = self.auroc(out, y)
        # log the outputs!
        if dataloader_idx != 0:
            # validation set
            self.optimal_thres_selector.add_data(out, y)
            self.log_dict({"final_val_acc": test_acc, "final_val_auc": test_auc})
        else:
            # test set
            self.group_acc(out, y, batch[2])
            self.log_dict({"test_acc": test_acc, "test_auc": test_auc})

    def on_test_end(self):
        super().on_test_end()
        th = self.optimal_thres_selector.compute_optimal_threshold()
        self.group_acc.set_thres(th)
        self.group_acc.computer_per_group_acc()

    def configure_optimizers(self):
        optimizer = torch_optimizer.Lamb(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, verbose=True, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }