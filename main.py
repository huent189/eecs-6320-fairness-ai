from datasets import *
from trainers import *
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.logger import Logger
import os


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--pipeline", choices=["full", "train", "test"])
        parser.add_argument("--exp_name")
        parser.add_argument("--version")
        parser.add_argument("--checkpoint")


class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer, pl_module, stage) -> None:
        if isinstance(trainer.logger, Logger):
            # Required for proper reproducibility
            config = self.parser.dump(self.config, skip_none=False)
            trainer.logger.log_hyperparams({"config": config})
# MODEL_REGISTRY.register_classes(trainers, pl.core.lightning.LightningModule)
# DATAMODULE_REGISTRY.register_classes(datasets, pl.core.LightningDataModule)


cli = CustomLightningCLI(
    subclass_mode_model=True,
    subclass_mode_data=True,
    run=False,
    trainer_defaults={
        "callbacks": [ModelCheckpoint(
            filename="{epoch:02d}",
            monitor="val/loss", mode="min",
            save_last=True,
            save_top_k=3
        ), EarlyStopping(monitor="val/loss", mode="min", patience=5)]
    }, save_config_callback=LoggerSaveConfigCallback, save_config_kwargs={"overwrite": True}
)

cli.trainer.logger = pl.loggers.wandb.WandbLogger(project='AI-debias', save_dir=os.path.join(cli.config['trainer']['default_root_dir'], cli.config['exp_name']), name=cli.config["version"],
                                                  log_model="all", entity='huent189')
if cli.config["pipeline"] == "full":
    cli.trainer.fit(cli.model, cli.datamodule,
                    ckpt_path=cli.config["checkpoint"])
    cli.trainer.test(
        cli.model,
        cli.datamodule,
        ckpt_path='best'
    )
elif cli.config["pipeline"] == "train":
    cli.trainer.fit(cli.model, cli.datamodule,
                    ckpt_path=cli.config["checkpoint"])
elif cli.config["pipeline"] == "test":
    cli.trainer.test(cli.model, cli.datamodule, cli.config["checkpoint"])
