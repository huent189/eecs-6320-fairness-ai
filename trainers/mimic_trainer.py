import torch
from torch import nn
import pytorch_lightning as pl
from models.mlp_model import MLP
import torch.optim as optim
from lightning.pytorch.core.module import LightningModule
import torchmetrics
import torch_optimizer
from datasets.mimic_dataset import num_groups_per_attrb, protected_group_to_index
from utils.metrics import GroupBasedAccuracy, OptimalThresholdSelector, OptimalThresholdPerGroup, GroupBasedAccuracyVaryingThrs
import copy
from losses.gce import GeneralizedCELoss


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
        self.loss = (
            nn.BCEWithLogitsLoss()
        )  # For binary classification, adjust if necessary
        self.acc = torchmetrics.Accuracy(task="binary")
        self.auroc = torchmetrics.AUROC(task="binary")
        self.group_acc = GroupBasedAccuracy(num_groups_per_attrb)
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
        if dataloader_idx == 0:
            self.optimal_thres_selector.add_data(out, y)
            self.log_dict({"final_val_acc": test_acc, "final_val_auc": test_auc})
        else:
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


class BiasCouncilTrainer(MIMICTrainer):
    def __init__(
        self,
        model,
        learning_rate=0.001,
        end_lr_factor=1.0,
        weight_decay=0.0,
        decay_steps=1000,
    ):
        super(BiasCouncilTrainer, self).__init__(
            model, learning_rate, end_lr_factor, weight_decay, decay_steps
        )
        self.bias_councils = [copy.deepcopy(model)]
        for m in self.bias_councils:
            m.init_weights()
        self.gce = GeneralizedCELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        biased_y_hats = [m(x) for m in self.bias_councils]
        unbiased_y_hat = self.model(x)
        biased_loss = 0
        unbiased_loss = 0
        for biased_y_hat in biased_y_hats:
            biased_loss = self.gce(biased_y_hat, y).mean() + biased_loss
            biased_y_hat = biased_y_hat.detach()
            p_y_hat = torch.sigmoid(biased_y_hat)
            p_y_hat = y * p_y_hat + (1 - y) * (1 - p_y_hat)
            unbiased_loss = unbiased_loss + self.loss(unbiased_y_hat, y) * p_y_hat
            unbiased_loss = unbiased_loss + (1 - p_y_hat) * self.loss(
                unbiased_y_hat, 1 - p_y_hat
            )
        unbiased_loss = unbiased_loss.mean()
        loss = biased_loss + unbiased_loss
        self.log_dict(
            {"biased_loss": biased_loss, "unbiased_loss": unbiased_loss, "loss": loss}
        )
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
        self.group_acc = GroupBasedAccuracy(num_groups_per_attrb)
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