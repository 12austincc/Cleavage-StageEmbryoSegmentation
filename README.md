# Cleavage-Stage Embryo Segmentation Using SAM-Based Dual Branch Pipeline: Development and Evaluation with the CleavageEmbryo Dataset

This is the official pytorch implementation of SAM-Based Dual Branch Pipeline, please refer the [paper](url) for more details.


## Introduction
### Abstract
**Motivation:** Embryo selection is one of the critical factors in determining the success of pregnancy in in vitro fertilization (IVF) techniques. Using artificial intelligence to aid in embryo selection could effectively address the current time-consuming, expensive and subjectively influenced process of embryo assessment by trained embryologists. However, current deep learning-based methods often focus on the segmentation or grading of blastocysts,  neglecting morphokinetic parameters or predicting cell development via time-lapse videos, thus lacking interpretability. Given the significance of morphokinetic and morphological evaluation of cleavage-stage embryos in predicting implantation potential, as emphasized by previous research, there is a necessity for an automated method to segment cleavage-stage embryos to improve this process.
**Results:** In this article, we introduce the SAM-based Dual Branch Segmentation Pipeline for automated segmentation of blastomeres in cleavage-stage embryos. Leveraging the powerful segmentation capability of SAM, the instance branch conducts instance segmentation of blastomeres, while the semantic branch performs semantic segmentation of fragments. Due to the lack of publicly available datasets, we constructed the CleavageEmbryo dataset, the first dataset of human cleavage-stage embryos with pixel-level annotations containing fragment information. We train and test a series of state-of-the-art segmentation algorithms on CleavageEmbryo. Our experiments demonstrate that our method outperforms existing algorithms in terms of objective metrics (mAP 0.748 on blastomere, Dice 0.694 on fragment) and visual quality, enabling more accurate segmentation of cleavage-stage embryos.


## Dataset
CleavageEmbryo Dataset.
![](https://github.com/12austincc/Cleavage-StageEmbryoSegmentation/blob/main/image/DatasetDescrip.png)
## Install
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`.
Install Segment Anything:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
Install requirements:
```
pip install -r requirements.txt
```
## QuickStart
pretrained [sam_dual]([https://whueducn-my.sharepoint.com/:u:/g/personal/2020302111430_whu_edu_cn/EdWbhvthCnhEkvgOuMUXQB8BPD62EplRApnq9qW0jjXdNA?e=yWne5S](https://whueducn-my.sharepoint.com/:u:/g/personal/2020302111430_whu_edu_cn/EVTIsyp3PAlImW-ggLkmS80BA9sBzq6QrKaCWrsi6SCHEw?e=qihhbx))
### Inference test images
```
python inference.py
```
### Evaluation with Pre-trained Models 
```
python evaluate.py
```

### Training on your own dataset
download [sam](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
modify config.py and then
```
python train_es.py
```


## Citation
```

```


## Acknowledgment
The code is built on [segment-anything-model](https://github.com/facebookresearch/segment-anything) and [lightning-sam](https://github.com/luca-medeiros/lightning-sam), many thanks for the Third Party Libs.
