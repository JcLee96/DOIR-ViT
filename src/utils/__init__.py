import logging
import warnings
from typing import List, Sequence
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
import pandas as pd
import math
import random
import torch
from functools import reduce
from torch.nn.functional import pairwise_distance
from torch.nn import _reduction as _Reduction
from typing import Callable, Optional
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torchmetrics.functional import pairwise_euclidean_distance
import time
import numpy as np


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "fatal",
            "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing
    """

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        log.info("Printing config tree with Rich! <config.print_config=True>")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
        config: DictConfig,
        print_order: Sequence[str] = (
                "datamodule",
                "model",
                "callbacks",
                "logger",
                "trainer",
        ),
        resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(
            f"Field '{field}' not found in config"
        )

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:  # sourcery skip: merge-dict-assign
    """Controls which config parts are saved by Lightning loggers.

    Additionaly saves:
    - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def finish(
        config: DictConfig,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
        callbacks: List[pl.Callback],
        logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


def bring_colon_dataset_csv(datatype, stage=None):
    # Directories
    path = "/colon data path"
    if stage != "fit" and stage is not None:
        return pd.read_csv(path + "test.csv")
    df_train = pd.read_csv(path + "train.csv")
    df_val = pd.read_csv(path + "valid.csv")
    return df_train, df_val




def bring_gastric_dataset_csv(stage=None):
    # Directories
    path = "your data path"

    if stage != "fit" and stage is not None:
        return pd.read_csv(f"{path}class4_step10_ds_test.csv")
    df_train = pd.read_csv(f"{path}class4_step05_ds_train.csv")
    df_val = pd.read_csv(f"{path}class4_step10_ds_valid.csv")
    return df_train, df_val

def tensor2np(data: torch.Tensor):
    """convert tensor to numpy array"""
    return data if isinstance(data, np.ndarray) else data.detach().cpu().numpy()


def dist_indexing(y, shuffle_y, y_idx_groupby, dist_matrix):
    indices = []
    for i, (yV, shuffleV) in enumerate(zip(y, shuffle_y)):
        if yV == shuffleV:
            indices.append(
                y_idx_groupby[yV][dist_matrix[i][y_idx_groupby[yV]].argmax()]
            )
        elif yV > shuffleV:
            flatten_ = reduce(lambda a, b: a + b, y_idx_groupby[:yV])
            indices.append(flatten_[dist_matrix[i][flatten_].argmin()])
        else:
            flatten_ = reduce(lambda a, b: a + b, y_idx_groupby[yV + 1:])
            indices.append(flatten_[dist_matrix[i][flatten_].argmin()])
    return indices


def params_freeze(model):
    model.blocks[22:].requires_grad_(False)
    for name, param in model.named_parameters():
        param.requires_grad = "head" in name
        param.requires_grad = "norm" in name


def get_distmat_heatmap(df, targets):
    df = pd.DataFrame(df.detach().cpu().numpy())
    plt.clf()
    plt.figure(figsize=(30, 30))
    confmat_heatmap = sns.heatmap(
        data=df,
        cmap="RdYlGn",
        annot=True,
        annot_kws={"size": 15},
        fmt=".2f",
        xticklabels=targets.detach().cpu().numpy(),
        yticklabels=targets.detach().cpu().numpy(),
        cbar=False,
    )

    confmat_heatmap.xaxis.set_label_position("top")
    plt.yticks(rotation=0)
    confmat_heatmap.tick_params(axis="x", which="both", bottom=False)

    return confmat_heatmap.get_figure()


def get_confmat(df):
    df = pd.DataFrame(df.detach().cpu().numpy())
    plt.clf()  # ADD THIS LINE
    plt.figure(figsize=(10, 10))
    confmat_heatmap = sns.heatmap(
        data=df,
        cmap="RdYlGn",
        annot=True,
        fmt=".3f",
        cbar=False,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted label")

    confmat_heatmap.xaxis.set_label_position("top")
    plt.yticks(rotation=0)
    confmat_heatmap.tick_params(axis="x", which="both", bottom=False)

    return confmat_heatmap.get_figure()


def get_feature_df(features, targets):
    cols = [f"feature_{i}" for i in range(features.shape[1])]
    df = pd.DataFrame(features.detach().cpu().numpy(), columns=cols)
    label_dict = {0: "BN_0", 1: "WD_1", 2: "MD_2", 3: "PD_3"}
    df["LABEL"] = targets.detach().cpu().numpy()
    df["LABEL"] = df["LABEL"].map(label_dict)

    return df


def get_max(lst):
    return torch.max(lst).unsqueeze(0)


def get_min(lst):
    return torch.min(lst).unsqueeze(0)


