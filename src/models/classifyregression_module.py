from typing import Any, List
import torch
import torch.nn as nn
import timm, math
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import MaxMetric, ConfusionMatrix, F1Score, CohenKappa, Accuracy
from src.utils import get_shuffled_label, get_confmat, CEOLoss, inverse_huber_loss
import wandb, numpy as np
from src.datamodules.colon_datamodule import ColonDataset, ColonDataModule
import pandas as pd
import decimal

import albumentations as A
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import cv2
import timm_change

class WeightClipper(object):
    def __init__(self, frequency=0.015):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-self.frequency, self.frequency)
            module.weight.data = w

class ClassifyregressionLitModule(LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.0005,
        t_max: int = 20,
        min_lr: int = 1e-6,
        T_0=15,
        T_mult=2,
        key='None',
        threshold=0.1,
        eta_min=1e-6,
        name="vit_base_patch16_224",
        pretrained=True,
        scheduler="CosineAnnealingLR",
        factor=0.5,
        patience=5,
        eps=1e-08,
        loss_weight=6.5,
        module_type="classifycompare",
        implementation_model=False,
        regression_loss="mse",
        regression_loss_type="exp_base",
        feature_extraction_method="sub",
        regression_loss_weight=1.0e-5,
        exp_type='each',
        berhu_para=2.0,
        class_cnt=4,
    ):

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = timm.create_model(
            self.hparams.name, pretrained=self.hparams.pretrained, num_classes=class_cnt)

        self.discriminator_layer1 = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, class_cnt),
        ) if 'net' in self.hparams.name else nn.Sequential(
            nn.Linear(self.model.head.in_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, class_cnt),
        )

        self.berhuloss = inverse_huber_loss
        self.berhu_para = berhu_para

        self.regression_loss_type = regression_loss_type
        self.exp_type = exp_type
        self.feature_extraction_method = feature_extraction_method
        self.regression_loss_weight = regression_loss_weight

        self.regression_layer = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 1),
        ) if 'net' in self.hparams.name else nn.Sequential(
            nn.Linear(self.model.head.in_features, 1),
        )

        self.regression_loss = regression_loss
        if self.regression_loss == "mse":
            self.criterion_regression = torch.nn.MSELoss(reduction='mean')
        elif self.regression_loss == "mae":
            self.criterion_regression = torch.nn.L1Loss(reduction='mean')

        self.criterion = torch.nn.CrossEntropyLoss()

        # clipper = WeightClipper(0.01)
        # self.regression_layer.apply(clipper)

        self.train_acc = Accuracy()
        self.reg_train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.reg_val_acc = Accuracy()
        self.val_f1 = F1Score(num_classes=class_cnt, average="macro")
        self.test_acc = Accuracy()
        self.reg_test_acc = Accuracy()
        self.train_acc_compare = Accuracy()
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()
        self.confusion_matrix = ConfusionMatrix(num_classes=class_cnt)
        self.f1_score = F1Score(num_classes=class_cnt, average="macro")
        self.cohen_kappa = CohenKappa(num_classes=class_cnt, weights="quadratic")

    def forward(self, x):  # 4 classification
        return self.discriminator_layer1(self.get_features(x.float()))

    def get_features(self, x):
        """get features from timm models

        Since densenet code is quite different from vit models, the extract part is different
        """
        features = self.model.global_pool(self.model.forward_features(x.float())) if 'densenet' in self.hparams.name else self.model.forward_features(x.float())
        features = features if 'densenet' in self.hparams.name else self.model.forward_head(features, pre_logits=True)
        return features

    def get_class_sub_list(self, origin):
        comparison = []
        for num, i in enumerate(origin.tolist()):
            comparison.append(origin.tolist()[0] - i)
        return torch.tensor(comparison, device=self.device)

    def get_exp_base_loss(self, pred, target, T):
        torch.set_printoptions(precision=32)

        value = torch.tensor(6, dtype=torch.float64) + torch.tensor(T, dtype=torch.float64)
        pred = pred.to(torch.float64)

        pred = torch.clamp(pred, min=-3, max=3)
        minibatch_loss = -torch.log(1 - (torch.clamp(torch.abs(target - pred), max=6) / value)).mean()

        return minibatch_loss

    def get_exp_square_loss(self, pred, target, T):

        torch.set_printoptions(precision=32)

        value = torch.tensor(6, dtype=torch.float64) + torch.tensor(T, dtype=torch.float64)
        pred = pred.to(torch.float64)

        minibatch_loss = 0
        if self.exp_type == 'each':
            pred = torch.clamp(pred, min=-3, max=3)
            minibatch_loss = -torch.log(1 - (torch.square(torch.abs(target - pred)) / value)).mean()
        elif self.exp_type == 'both':
            minibatch_loss = -torch.log(1 - (torch.clamp(torch.square(torch.abs(target - pred)), max=36) / value)).mean()

        return minibatch_loss


    def step(self, batch):
        x, y = batch
        features = self.get_features(x)
        logits_4cls = self.discriminator_layer1(features)
        loss_4cls = self.criterion(logits_4cls, y)
        preds_4cls = torch.argmax(logits_4cls, dim=1)

        sub_class_num = self.get_class_sub_list(y).to(torch.float32)

        regression_features = features - features[0].repeat(len(features), 1)
        logits_regression = self.regression_layer(regression_features).to(torch.float32)


        if self.regression_loss == 'mse' or self.regression_loss == 'mae':
            loss_regression = self.criterion_regression(logits_regression[:, 0], sub_class_num)
        else:
            loss_regression = self.get_exp_base_loss(logits_regression[:, 0], sub_class_num, self.regression_loss_weight)

        loss = loss_4cls + (loss_regression * self.hparams.loss_weight)

        return loss, preds_4cls, y, loss_4cls, loss_regression, loss_regression * self.hparams.loss_weight

    def training_step(self, batch, batch_idx):
        loss, preds_4cls, target_4cls, loss_4cls, loss_regression, weighted_regression_loss = self.step(batch)


        acc = self.train_acc(preds=preds_4cls, target=target_4cls)

        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/classificaion_loss", loss_4cls, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/regression_loss", loss_regression, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/weighted_regression_loss", weighted_regression_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log("LearningRate", self.optimizer.param_groups[0]["lr"])

        return {
            "loss": loss,
            "acc": acc,
            "preds": preds_4cls,
            "targets": target_4cls,
            "classification_loss": loss_4cls,
            "regression_loss": loss_regression,
            "weighted_regression_loss": weighted_regression_loss,
        }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        sch = self.lr_schedulers()
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["val/loss"])

    def validation_step(self, batch, batch_idx):
        loss, preds_4cls, target_4cls, loss_4cls, loss_regression, weighted_regression_loss = self.step(batch)

        acc = self.val_acc(preds_4cls, target_4cls)
        f1 = self.val_f1(preds_4cls, target_4cls)

        self.log("val/total_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/classificaion_loss", loss_4cls, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/regression_loss", loss_regression, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/weighted_regression_loss", weighted_regression_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", f1, on_step=False, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "acc": acc,
            "preds": preds_4cls,
            "targets": target_4cls,
            "f1": f1,
            "classification_loss": loss_4cls,
            "regression_loss": loss_regression,
            "weighted_regression_loss": weighted_regression_loss,
        }

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
        f1 = self.val_f1.compute()
        self.val_f1_best.update(f1)

        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)

        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
        self.log("val/f1_best", self.val_f1_best.compute(), on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        loss, preds_4cls, target_4cls, loss_4cls, loss_regression, weighted_regression_loss = self.step(batch)

        # loss, preds_4cls, target_4cls, loss_4cls, loss_regression, weighted_regression_loss, \
        #     sub_class_num, reg_pred_regression = self.step(batch)

        self.confusion_matrix(preds_4cls, target_4cls)
        self.f1_score(preds_4cls, target_4cls)
        self.cohen_kappa(preds_4cls, target_4cls)

        acc = self.test_acc(preds_4cls, target_4cls)
        # reg_acc = self.reg_test_acc(reg_pred_regression+3, sub_class_num.to(torch.int64)+3)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/classificaion_loss", loss_4cls, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/regression_loss", loss_regression, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/weighted_regression_loss", weighted_regression_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {
            "loss": loss,
            "acc": acc,
            "preds": preds_4cls,
            "targets": target_4cls,
            "classification_loss": loss_4cls,
            "regression_loss": loss_regression,
            "weighted_regression_loss": weighted_regression_loss,
        }

    def test_epoch_end(self, outputs):

        cm = self.confusion_matrix.compute()
        f1 = self.f1_score.compute()
        qwk = self.cohen_kappa.compute()
        p = get_confmat(cm)

        self.logger.experiment.log({"test/conf_matrix": wandb.Image(p)})
        self.log("test/f1_macro", f1, on_step=False, on_epoch=True)
        self.log("test/wqKappa", qwk, on_step=False, on_epoch=True)

        self.test_acc.reset()
        self.confusion_matrix.reset()
        self.f1_score.reset()
        self.cohen_kappa.reset()

    def on_epoch_end(self):
        self.train_acc.reset()
        self.val_acc.reset()
        self.test_acc.reset()


    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        self.scheduler = self.get_scheduler()
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
                "monitor": "val/loss",
            }

        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def get_scheduler(self):
        schedulers = {
            "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.hparams.factor,
                patience=self.hparams.patience,
                verbose=True,
                eps=self.hparams.eps,
            ),
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.hparams.t_max, eta_min=self.hparams.min_lr, last_epoch=-1
            ),
            "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.hparams.T_0,
                T_mult=1,
                eta_min=self.hparams.min_lr,
                last_epoch=-1,
            ),
            "StepLR": torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1),
            "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95),
        }
        if self.hparams.scheduler not in schedulers:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")

        return schedulers.get(self.hparams.scheduler, schedulers["ReduceLROnPlateau"])
