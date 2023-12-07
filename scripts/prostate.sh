#!/bin/bash


# model.name: timm model name
# datamodule: dataset_file_name in the config.datamodule

# categorical classification
python ../train.py model.name='vit_base_r50_s16_384' seed=42 model=classify4.yaml datamodule.data_ratio=1.0 datamodule=havard.yaml model.scheduler='CosineAnnealingWarmRestarts' logger.wandb.project='harvard_Analysis' logger.wandb.tags=['classify','acc_best_model'] datamodule.num_workers=8 trainer.devices=\'0,1\' datamodule.batch_size=16
python /home/compu/LJC/colon_compare/train_test.py model.name='vit_base_r50_s16_384' seed=42 model=classify4.yaml model.loss_weight=0.4 datamodule.data_ratio=1.0 datamodule=ubc.yaml model.scheduler='CosineAnnealingWarmRestarts' logger.wandb.project='harvard_Analysis' logger.wandb.tags=['test2:ubc'] ckpt_path="best_mode path" datamodule.num_workers=8 trainer.devices=[1] datamodule.batch_size=16

# categorical and regression classification
python ../train.py model.name='vit_base_r50_s16_384' seed=42 model=classifyregression.yaml model.feature_extraction_method='sub' model.regression_loss='exp_base' model.regression_loss_weight=1.0e-5 model.loss_weight=6.5 model.exp_type='each' datamodule.data_ratio=1.0 datamodule=havard.yaml model.scheduler='CosineAnnealingWarmRestarts' logger.wandb.project='harvard_Analysis' logger.wandb.tags=['classify','sub_regression','exp_base','regression_loss_weight:1.0e-5','each','loss_weight:6.5'] datamodule.num_workers=8 trainer.devices=\'2,3\' datamodule.batch_size=16
python /home/compu/LJC/colon_compare/train_test.py model.name='vit_base_r50_s16_384' seed=42 model=classifyregression.yaml model.feature_extraction_method='sub' model.regression_loss='exp_base' model.regression_loss_weight=1.0e-5 model.loss_weight=6.5 datamodule.data_ratio=1.0 datamodule=ubc.yaml model.scheduler='CosineAnnealingWarmRestarts' logger.wandb.project='harvard_Analysis' logger.wandb.tags=['classify','sub_regression','exp_base','regression_loss_weight:1.0e-5','each','loss_weight:6.5','test2: ubc'] ckpt_path="best model path" datamodule.num_workers=8 trainer.devices=\'2,3\' datamodule.batch_size=16
