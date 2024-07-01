# Cleavage-Stage Embryo Segmentation Using SAM-Based Dual Branch Pipeline: Development and Evaluation with the CleavageEmbryo Dataset

This is the official pytorch implementation of SAM-Based Dual Branch Pipeline, please refer the [paper](url) for more details.


## Introduction
### Abstract
**Motivation:** Embryo selection is one of the critical factors in determining the success of pregnancy in in vitro fertilization (IVF) techniques.Using artificial intelligence to assist in embryo selection could effectively address the current time-consuming, expensive and subjectively influenced process of embryo assessment by trained embryologists. However, current deep learning-based methods often concentrate on the segmentation or grading of blastocysts which neglects morphokinetic parameters or predicting cell development via time-lapse videos ,thus lacking interpretability. Given the importance of morphokinetic and morphological evaluation of cleavage-stage embryos in predicting implantation potential, as highlighted by some prior research, there is a need for an automated method to segment cleavage-stage embryos to enhance this process.
**Results:** In this study, we present the SAM-based Dual Branch Segmentation Pipeline for automated segmentation of blastomeres in cleavage-stage embryos. With the powerful segmentation capability of SAM, the instance branch performs instance segmentation of blastomeres and the semantic branch performs semantic segmentation of fragments. Due to the lack of publicly available datasets, we constructed the CleavageEmbryo dataset, which is the first dataset of human cleavage-stage embryos with pixel-level annotations containing fragment information. We trained and tested a series of SOTA segmentation algorithms on CleavageEmbryo, and our experiments show that our method outperforms existing algorithms in terms of objective metrics and visual quality, enabling more accurate segmentation of cleavage-stage embryos.
![](https://github.com/12austincc/Cleavage-StageEmbryoSegmentation/blob/main/image/overall.png)
![](https://github.com/12austincc/Cleavage-StageEmbryoSegmentation/blob/main/image/semantic.png)
![](https://github.com/12austincc/Cleavage-StageEmbryoSegmentation/blob/main/image/instance.png)

### Visualization
![](https://github.com/12austincc/Cleavage-StageEmbryoSegmentation/blob/main/image/object_detectionv2.png)
![](https://github.com/12austincc/Cleavage-StageEmbryoSegmentation/blob/main/image/blastomereSeg.png)
### Results
object detection
blatomere segmentation
fragment segmentation

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
### Visualization 
Download [pre-trained model](url)
```
python visualize.py
```
### Evaluation with Pre-trained Models 
```
python evaluate.py
```

### Training on your own dataset
modify config.py and then
```
python train_es.py
```


## Citation
```

```


## Acknowledgment
The code is built on [segment-anything-model](https://github.com/facebookresearch/segment-anything) and [lightning-sam](https://github.com/luca-medeiros/lightning-sam), many thanks for the Third Party Libs.
