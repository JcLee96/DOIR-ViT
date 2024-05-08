# DIOR-ViT

<div align="center">

# DIOR-ViT: Differential Ordinal Learning Vision Transformer for Cancer Classification in Pathology Images

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/html/Lee_Order-ViT_Order_Learning_Vision_Transformer_for_Cancer_Classification_in_Pathology_ICCVW_2023_paper.html)
[![Conference](http://img.shields.io/badge/_Conference-2023-4b44ce.svg)](https://cvamd2023.github.io/)

</div>

## Description

![DIOR-ViT](/DIOR-ViT_Model.png)

The overall model architecture is as follows. Categorical classification and sequential relationship classification problems are performed simultaneously.


## Datasets

All the models in this project were evaluated on the following datasets:

- [Colon_KBSMC](https://github.com/QuIIL/KBSMC_colon_cancer_grading_dataset) (Colon TMA from Kangbuk Samsung Hospital)
- [Colon_KBSMC](https://github.com/QuIIL/KBSMC_colon_cancer_grading_dataset) (Colon WSI from Kangbuk Samsung Hospital)
- [Gastric_KBSMC](-) (Gastric from Kangbuk Samsung Hospital)
- [Prostate_UHU](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP)(Harvard dataverse)
- [Prostate_UBC](https://gleason2019.grand-challenge.org/) (MICCAI 2019 UBC)



## How to run

Install dependencies

```bash


# [OPTIONAL] create conda environment
conda create --name [이름] python=3.
conda activate [이름]
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

pip install -r requirements.txt


```

## Repository
```
/config: data and model parameter setting
/scripts: .sh file
/src: data load and augmentation, model code
```
 
## How to training for DIOR-ViT
```
## Only Categorical classification and DIOR-ViT
# model.name = timm model name & ../train_test: Code for validating different datasets using the best model
Using /scripts/colon.sh
Using /scripts/gastric.sh
Using /scripts/prostate.sh



