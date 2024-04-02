import torch
from torch import nn
import torch.optim as optim
from lightning.pytorch.core.module import LightningModule
import torchmetrics
from datasets.mimic_dataset import (
    num_groups_per_attrb,
    protected_group_to_index,
    group_labels,
    disease_labels,
    NO_FINDING_INDEX,
    ATTRB_LABELS,
)
from utils.metrics import (
    GroupBasedStats,
    OptimalThresholdSelector,
    OptimalThresholdPerGroup,
    GroupBasedStatsVaryingThrs,
)
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
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        self.acc = torchmetrics.classification.MultilabelAccuracy(
            num_labels=len(disease_labels)
        )
        self.average_auroc = torchmetrics.classification.MultilabelAUROC(
            num_labels=len(disease_labels)
        )
        self.per_label_auroc = torchmetrics.classification.MultilabelAUROC(
            num_labels=len(disease_labels), average="none"
        )
        self.group_metrics = GroupBasedStats(num_groups_per_attrb)
        self.group_labels = group_labels
        self.optimal_thres_selector = OptimalThresholdSelector()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self(x)
        loss = self.loss(y_hat, y).mean()
        self.log_dict({"train/loss": loss})
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
        self.log_dict({"val/loss": loss, "val/acc": val_acc})

    def on_validation_end(self):
        super().on_validation_end()
        self.logger.experiment.log({"val/auroc": self.average_auroc.compute()})

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
            self.group_metrics(
                out[:, NO_FINDING_INDEX, None], y[:, NO_FINDING_INDEX, None], batch[2]
            )
            self.per_label_auroc.update(out, y)
            self.log_dict({"test/acc": avg_acc})
        else:
            self.optimal_thres_selector.add_data(
                out[:, NO_FINDING_INDEX, None], y[:, NO_FINDING_INDEX, None]
            )
            self.average_auroc.update(out, y)
            self.log_dict({"val/final_acc": avg_acc})

    def on_test_end(self):
        super().on_test_end()
        self.logger.experiment.log({"val/auroc": self.average_auroc.compute()})
        test_per_label_auroc = self.per_label_auroc.compute()
        # First column is disease name, second column is auroc
        table_data = [
            [d, test_per_label_auroc[i].item()] for i, d in enumerate(disease_labels)
        ]
        table_data.append(["Average", test_per_label_auroc.mean().item()])
        self.logger.experiment.log(
            {
                "test/per_label_auroc": wandb.Table(
                    data=table_data, columns=["disease", "auroc"]
                )
            }
        )

        _, th = self.optimal_thres_selector.get_optimal_threshold()
        threshold_log = {}
        threshold_log["test/optimal_threshold"] = th
        self.logger.experiment.log(threshold_log)

        self.group_metrics.set_thres(th)

        def get_metric_mu(group_metric):
            return sum(group_metric) / len(group_metric)
        def log_metric(metric, plot_per_attb):
            data_table = wandb.Table(
                data=plot_per_attb, columns=["Label", metric]
            )
            self.logger.experiment.log(
                {
                    f"test/{attb}/{metric}": wandb.plot.bar(
                        data_table, "Label", metric, title=f"{metric} for {attb}"
                    )
                }
            )
        metrics = self.group_metrics.get_all_stats()
        table_data = []
        for protected_attb_idx in range(len(num_groups_per_attrb)):            
            attb = ATTRB_LABELS[protected_attb_idx]
            table_data.append(
                [f"Average & Gaps of {attb}", ] + [x for key in metrics.keys() for x in [get_metric_mu(metrics[key][0][protected_attb_idx]), metrics[key][1][protected_attb_idx]]]
            )
            metric_plot_per_attb = {key:[] for key in metrics.keys()}
            for j in range(num_groups_per_attrb[protected_attb_idx]):
                sub_group_label = group_labels[protected_attb_idx][j]
                table_data.append([sub_group_label] + [x for key in metrics.keys() for x in [metrics[key][0][protected_attb_idx][j], metrics[key][0][protected_attb_idx][j] - get_metric_mu(metrics[key][0][protected_attb_idx])]])
                for key in metrics.keys():
                    metric = metrics[key][0][protected_attb_idx][j]
                    metric_plot_per_attb[key].append([sub_group_label, metric])
                    log_metric(key, metric_plot_per_attb[key])
        self.logger.experiment.log(
            {
                "test/fpr_summary": wandb.Table(
                    data=table_data,
                    columns=["Attribute"] + [x for key in metrics.keys() for x in [key, "{}_gap".format(key)]],
                )
            }
        )

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
                "monitor": "val/loss",
            },
        }


class BiasCouncilTrainer(MIMICTrainer):
    def __init__(
        self,
        model,
        bias_model,
        num_bias_models=1,
        learning_rate=0.001,
        end_lr_factor=1.0,
        weight_decay=0.0,
        decay_steps=1000,
    ):
        super(BiasCouncilTrainer, self).__init__(
            model, learning_rate, end_lr_factor, weight_decay, decay_steps
        )
        self.bias_councils = nn.ModuleList(
            [copy.deepcopy(bias_model) for _ in range(num_bias_models)]
        )
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
        unbiased_loss = unbiased_loss + (1 - p_y_hat) * self.loss(
            unbiased_y_hat, 1 - p_y_hat
        )
        unbiased_loss = unbiased_loss.mean()
        loss = biased_loss * 5 + unbiased_loss
        self.log_dict(
            {"biased_loss": biased_loss, "unbiased_loss": unbiased_loss, "loss": loss}
        )
        return loss

    def configure_optimizers(self):
        params = list(self.model.parameters())
        for m in self.bias_councils:
            params += list(m.parameters())
        optimizer = torch.optim.Adam(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            capturable=True,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, verbose=True, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }


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
        unbiased_loss = (
            unbiased_loss * ce_biased_loss / (ce_biased_loss + unbiased_loss.detach())
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
        protected_attribute=None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_steps = decay_steps
        self.end_lr_factor = end_lr_factor
        self.loss = nn.BCEWithLogitsLoss(
            reduction="none"
        )  # For binary classification, adjust if necessary
        self.acc = torchmetrics.classification.MultilabelAccuracy(
            num_labels=len(disease_labels)
        )
        self.average_auroc = torchmetrics.classification.MultilabelAUROC(
            num_labels=len(disease_labels)
        )
        self.per_label_auroc = torchmetrics.classification.MultilabelAUROC(
            num_labels=len(disease_labels), average="none"
        )
        self.group_metrics = GroupBasedStats(num_groups_per_attrb)
        self.group_labels = group_labels
        self.group_stats = GroupBasedStatsVaryingThrs()
        self.optimal_thres_selector = OptimalThresholdPerGroup()
        self.protected_attribute = protected_attribute
        self.protected_attribute_idx = protected_group_to_index[
            self.protected_attribute
        ]
        self.group_stats.set_group_attrb_idx(self.protected_attribute_idx)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self(x)
        loss = self.loss(y_hat, y).mean()
        self.log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]

        # implement your own
        out = self(x)
        loss = self.loss(out, y).mean()
        out = torch.sigmoid(out)
        y = y.int()
        # calculate acc
        val_acc = self.acc(out, y)
        self.average_auroc.update(out, y)
        # log the outputs!
        self.log_dict({"val/loss": loss, "val/acc": val_acc})

    def on_validation_end(self):
        super().on_validation_end()
        self.logger.experiment.log({"val/auroc": self.average_auroc.compute()})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch[:2]
        y = y.int()
        out = self(x)
        # calculate acc
        out = torch.sigmoid(out)
        # calculate acc
        avg_acc = self.acc(out, y)
        # log the outputs!
        if dataloader_idx == 0:
            self.group_stats(
                out[:, NO_FINDING_INDEX, None],
                y[:, NO_FINDING_INDEX, None],
                batch[2],
            )
            self.group_metrics(
                out[:, NO_FINDING_INDEX, None], y[:, NO_FINDING_INDEX, None], batch[2]
            )
            self.per_label_auroc.update(out, y)
            self.log_dict({"test/acc": avg_acc})

        else:
            # dataloader idx for validation sets start from 1, since the first
            # one is the actual test set, we need the actual group index to retrive the
            # gruop name later on, so we substract idx by one to get the actual value!
            self.optimal_thres_selector.add_data(
                out[:, NO_FINDING_INDEX, None],
                y[:, NO_FINDING_INDEX, None],
                dataloader_idx - 1,
            )
            self.average_auroc.update(out, y)
            self.log_dict({"val/final_acc": avg_acc})

    def on_test_end(self):
        super().on_test_end()
        self.logger.experiment.log({"val/auroc": self.average_auroc.compute()})
        test_per_label_auroc = self.per_label_auroc.compute()
        # First column is disease name, second column is auroc
        table_data = [
            [d, test_per_label_auroc[i].item()] for i, d in enumerate(disease_labels)
        ]
        table_data.append(["Average", test_per_label_auroc.mean().item()])
        self.logger.experiment.log(
            {
                "test/per_label_auroc": wandb.Table(
                    data=table_data, columns=["disease", "auroc"]
                )
            }
        )

        ths = self.optimal_thres_selector.get_optimal_thresholds()
        threshold_log = {}
        for t in ths.keys():
            threshold_log["test/optimal_thresholds/{}".format(t)] = ths[t]
        self.logger.experiment.log(threshold_log)

        self.group_stats.set_thres(ths)
        # self.group_stats.compute_per_group_stats(self.protected_attribute_idx)
        def get_metric_mu(group_metric):
            return sum(group_metric) / len(group_metric)
        def log_metric(metric, plot_per_attb):
            data_table = wandb.Table(
                data=plot_per_attb, columns=["Label", metric]
            )
            self.logger.experiment.log(
                {
                    f"test/{attb}/{metric}": wandb.plot.bar(
                        data_table, "Label", metric, title=f"{metric} for {attb}"
                    )
                }
            )



        metrics = self.group_stats.get_all_stats()
        table_data = []
        for protected_attb_idx in range(len(num_groups_per_attrb)):            
            # First column is attb, second column is fpr, third column is fpr_gap
            # 4: accuracy, 5: accuracy gap
            attb = ATTRB_LABELS[protected_attb_idx]            
            # mu_fpr = sum(group_fpr) / len(group_fpr)
            # mu_acc = sum(group_acc) / len(group_acc)
            table_data.append(
                [f"Average & Gaps of {attb}", ] + [x for key in metrics.keys() for x in [get_metric_mu(metrics[key][0][protected_attb_idx]), metrics[key][1][protected_attb_idx]]]
            )
            # fpr_plot_per_attb = []
            metric_plot_per_attb = {key:[] for key in metrics.keys()}
            for j in range(num_groups_per_attrb[protected_attb_idx]):
                sub_group_label = group_labels[protected_attb_idx][j]
                table_data.append([sub_group_label] + [x for key in metrics.keys() for x in [metrics[key][0][protected_attb_idx][j], metrics[key][0][protected_attb_idx][j] - get_metric_mu(metrics[key][0][protected_attb_idx])]])
                for key in metrics.keys():
                    metric = metrics[key][0][protected_attb_idx][j]
                    metric_plot_per_attb[key].append([sub_group_label, metric])
                    log_metric(key, metric_plot_per_attb[key])
        self.logger.experiment.log(
            {
                "test/fpr_summary": wandb.Table(
                    data=table_data,
                    columns=["Attribute"] + [x for key in metrics.keys() for x in [key, "{}_gap".format(key)]],
                )
            }
        )

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
                "monitor": "val/loss",
            },
        }


# TODO: update logging stuff here
class SBS_MIMICTrainer(LightningModule):
    def __init__(
        self,
        model,
        learning_rate=0.001,
        end_lr_factor=1.0,
        weight_decay=0.0,
        decay_steps=1000,
        protected_attribute=None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_steps = decay_steps
        self.end_lr_factor = end_lr_factor
        self.loss = nn.BCEWithLogitsLoss(
            reduction="none"
        )  # For binary classification, adjust if necessary
        self.acc = torchmetrics.classification.MultilabelAccuracy(
            num_labels=len(disease_labels)
        )
        self.average_auroc = torchmetrics.classification.MultilabelAUROC(
            num_labels=len(disease_labels)
        )
        self.per_label_auroc = torchmetrics.classification.MultilabelAUROC(
            num_labels=len(disease_labels), average="none"
        )
        self.optimal_thres_selector = OptimalThresholdSelector()
        self.group_labels = group_labels
        self.group_metrics = GroupBasedStats(num_groups_per_attrb)
        self.protected_attribute = protected_attribute
        self.protected_attribute_idx = protected_group_to_index[
            self.protected_attribute
        ]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[:2]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log_dict({"train/loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[:2]

        out = self(x)
        loss = self.loss(out, y).mean()
        out = torch.sigmoid(out)
        y = y.int()
        # calculate acc
        val_acc = self.acc(out, y)
        self.average_auroc.update(out, y)
        # log the outputs!
        self.log_dict({"val/loss": loss, "val/acc": val_acc})

    def on_validation_end(self):
        super().on_validation_end()
        self.logger.experiment.log({"val/auroc": self.average_auroc.compute()})

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch[:2]
        y = y.int()
        out = self(x)
        # calculate acc
        out = torch.sigmoid(out)
        # calculate acc
        avg_acc = self.acc(out, y)
        # log the outputs!
        if dataloader_idx != 0:
            self.group_metrics(
                out[:, NO_FINDING_INDEX, None], y[:, NO_FINDING_INDEX, None], batch[2]
            )
            self.per_label_auroc.update(out, y)
            self.log_dict({"test/acc": avg_acc})
        else:
            self.optimal_thres_selector.add_data(
                out[:, NO_FINDING_INDEX, None], y[:, NO_FINDING_INDEX, None]
            )
            self.average_auroc.update(out, y)
            self.log_dict({"val/final_acc": avg_acc})

    def on_test_end(self):
        super().on_test_end()
        self.logger.experiment.log({"val/auroc": self.average_auroc.compute()})
        test_per_label_auroc = self.per_label_auroc.compute()
        # First column is disease name, second column is auroc
        table_data = [
            [d, test_per_label_auroc[i].item()] for i, d in enumerate(disease_labels)
        ]
        table_data.append(["Average", test_per_label_auroc.mean().item()])
        self.logger.experiment.log(
            {
                "test/per_label_auroc": wandb.Table(
                    data=table_data, columns=["disease", "auroc"]
                )
            }
        )

        _, th = self.optimal_thres_selector.get_optimal_threshold()
        threshold_log = {}
        threshold_log["test/optimal_threshold"] = th
        self.logger.experiment.log(threshold_log)

        self.group_metrics.set_thres(th)

        def get_metric_mu(group_metric):
            return sum(group_metric) / len(group_metric)
        def log_metric(metric, plot_per_attb):
            data_table = wandb.Table(
                data=plot_per_attb, columns=["Label", metric]
            )
            self.logger.experiment.log(
                {
                    f"test/{attb}/{metric}": wandb.plot.bar(
                        data_table, "Label", metric, title=f"{metric} for {attb}"
                    )
                }
            )
        metrics = self.group_metrics.get_all_stats()
        table_data = []
        for protected_attb_idx in range(len(num_groups_per_attrb)):            
            attb = ATTRB_LABELS[protected_attb_idx]
            table_data.append(
                [f"Average & Gaps of {attb}", ] + [x for key in metrics.keys() for x in [get_metric_mu(metrics[key][0][protected_attb_idx]), metrics[key][1][protected_attb_idx]]]
            )
            metric_plot_per_attb = {key:[] for key in metrics.keys()}
            for j in range(num_groups_per_attrb[protected_attb_idx]):
                sub_group_label = group_labels[protected_attb_idx][j]
                table_data.append([sub_group_label] + [x for key in metrics.keys() for x in [metrics[key][0][protected_attb_idx][j], metrics[key][0][protected_attb_idx][j] - get_metric_mu(metrics[key][0][protected_attb_idx])]])
                for key in metrics.keys():
                    metric = metrics[key][0][protected_attb_idx][j]
                    metric_plot_per_attb[key].append([sub_group_label, metric])
                    log_metric(key, metric_plot_per_attb[key])
        self.logger.experiment.log(
            {
                "test/fpr_summary": wandb.Table(
                    data=table_data,
                    columns=["Attribute"] + [x for key in metrics.keys() for x in [key, "{}_gap".format(key)]],
                )
            }
        )


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
                "monitor": "val/loss",
            },
        }
