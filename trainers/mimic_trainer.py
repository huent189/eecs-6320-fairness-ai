import torch
from torch import nn
import torch.optim as optim
from lightning.pytorch.core.module import LightningModule
import torchmetrics
from datasets.mimic_dataset import num_groups_per_attrb, group_labels
from utils.metrics import GroupBasedAccuracy, OptimalThresholdSelector
import copy
from losses.gce import GeneralizedCELoss
import wandb
class MIMICTrainer(LightningModule):
    def __init__(self, model, learning_rate=0.001, end_lr_factor=1.0, weight_decay=0.0, decay_steps=1000):
        super(MIMICTrainer, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_steps = decay_steps
        self.end_lr_factor = end_lr_factor
        self.loss = nn.BCEWithLogitsLoss(reduction='none')  # For binary classification, adjust if necessary
        self.acc = torchmetrics.Accuracy(task='binary')
        self.auroc = torchmetrics.AUROC(task="binary")
        self.group_acc = GroupBasedAccuracy(num_groups_per_attrb)
        self.group_labels = group_labels
        self.optimal_thres_selector = OptimalThresholdSelector()
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self(x)
        loss = self.loss(y_hat, y).mean()
        self.log_dict({'train_loss': loss})
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]

        # implement your own
        out = self(x)
        loss = self.loss(out, y).mean()
        out = torch.sigmoid(out)
        # calculate acc
        val_acc = self.acc(out, y)
        val_auc = self.auroc(out, y)
        # log the outputs!
        self.log_dict({'val_loss': loss, 'val_acc': val_acc, 'val_auc' : val_auc})
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
            self.log_dict({'final_val_acc': test_acc, 'final_val_auc': test_auc})
        else:
            self.group_acc(out, y, batch[2])
            self.log_dict({'test_acc': test_acc, 'test_auc': test_auc})
    def on_test_end(self):
        super().on_test_end()
        th = self.optimal_thres_selector.compute_optimal_threshold()
        self.group_acc.set_thres(th)
        group_accs, group_gaps = self.group_acc.computer_per_group_acc()
        data = []
        for i, acc in enumerate(group_accs):
            for j, a in enumerate(acc):
                data.append([str(i) + '_' + self.group_labels[i][j], a])
        table = wandb.Table(data=data, columns=["label", "value"])
        self.logger.experiment.log({"group_acc": wandb.plot.bar(table, "label", "value", title="Group Accuracy")})
        data = []
        for i, gap in enumerate(group_gaps):
            data.append([i, gap])
        table = wandb.Table(data=data, columns=["attb", "gap"])
        self.logger.experiment.log({"group_gap": wandb.plot.bar(table, "attb", "gap", title="Group Gap")})
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5)
        return {'optimizer' : optimizer,
                'lr_scheduler' : {'scheduler' :scheduler,
                                'monitor' : 'val_loss',}}

class BiasCouncilTrainer(MIMICTrainer):
    def __init__(self, model, bias_model, num_bias_models=1, learning_rate=0.001, end_lr_factor=1.0, weight_decay=0.0, decay_steps=1000):
        super(BiasCouncilTrainer, self).__init__(model, learning_rate, end_lr_factor, weight_decay, decay_steps)
        self.bias_councils = nn.ModuleList([copy.deepcopy(bias_model) for _ in range(num_bias_models)])
        for m in self.bias_councils:
            m.init_weights()
        self.gce = GeneralizedCELoss()
        self.num_bias_models = num_bias_models
    def training_step(self, batch, batch_idx):
        if not isinstance(batch[0], list):
            batch = [batch for _ in range(self.num_bias_models)]
        biased_loss = 0
        unbiased_loss =0
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
                p_y_hat = torch.max(p_y_hat , p_by)
        unbiased_loss = unbiased_loss + self.loss(unbiased_y_hat, y) * p_y_hat 
        unbiased_loss = unbiased_loss + (1 - p_y_hat) * self.loss(unbiased_y_hat, 1 - p_y_hat)
        unbiased_loss = unbiased_loss.mean()
        loss = biased_loss * 100 + unbiased_loss
        self.log_dict({'biased_loss': biased_loss, 'unbiased_loss' : unbiased_loss, 'loss' : loss})
        return loss
    def configure_optimizers(self):
        params = list(self.model.parameters())
        for m in self.bias_councils:
            params += list(m.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay, capturable=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5)
        return {'optimizer' : optimizer,
                'lr_scheduler' : {'scheduler' :scheduler,
                                'monitor' : 'val_loss',}}
    
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
        unbiased_loss = unbiased_loss * ce_biased_loss / (ce_biased_loss + unbiased_loss.detach())
        unbiased_loss = unbiased_loss.mean()
        loss = biased_loss + unbiased_loss
        self.log_dict({'biased_loss': biased_loss, 'unbiased_loss' : unbiased_loss, 'loss' : loss})
        return loss