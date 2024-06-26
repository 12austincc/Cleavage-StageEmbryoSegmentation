# Cleavage-Stage Embryo Segmentation Using SAM-Based Dual Branch Pipeline: Development and Evaluation with the CleavageEmbryo Dataset

This is the official pytorch implementation of SAM-Based Dual Branch Pipeline, please refer the [paper](url) for more details.


## Introduction
### Abstract
**Motivation:** Embryo selection is one of the critical factors in determining the success of pregnancy in in vitro fertilization (IVF) techniques.Using artificial intelligence to assist in embryo selection could effectively address the current time-consuming, expensive and subjectively influenced process of embryo assessment by trained embryologists. However, current deep learning-based methods often concentrate on the segmentation or grading of blastocysts which neglects morphokinetic parameters or predicting cell development via time-lapse videos ,thus lacking interpretability. Given the importance of morphokinetic and morphological evaluation of cleavage-stage embryos in predicting implantation potential, as highlighted by some prior research, there is a need for an automated method to segment cleavage-stage embryos to enhance this process.
**Results:** In this study, we present the SAM-based Dual Branch Segmentation Pipeline for automated segmentation of blastomeres in cleavage-stage embryos. With the powerful segmentation capability of SAM, the instance branch performs instance segmentation of blastomeres and the semantic branch performs semantic segmentation of fragments. Due to the lack of publicly available datasets, we constructed the CleavageEmbryo dataset, which is the first dataset of human cleavage-stage embryos with pixel-level annotations containing fragment information. We trained and tested a series of SOTA segmentation algorithms on CleavageEmbryo, and our experiments show that our method outperforms existing algorithms in terms of objective metrics and visual quality, enabling more accurate segmentation of cleavage-stage embryos.
![](url)


### Visualization
![](url)
### Results
object detection
blatomere segmentation
fragment segmentation

## Dataset
CleavageEmbryo Dataset.

## Install

## QuickStart
### Inference and Visualization with Pre-trained Models 
Use `tools/inference.py`
```
python tools/inference.py 
    --annTestFile <path to the test json file> 
    --imgTestFile <path to the test image file> 
    --configFile <path to the config file> 
    --outputDir <path to the output directory>
    --weightsFile <name of the weights file>
```

Download [pre-trained model](url)
### Evaluation with Pre-trained Models 
Use `tools/evaluate.py`
```
python tools/evaluate.py 
    --annTestFile <path to the test json file> 
    --imgTestFile <path to the test image file> 
    --configFile <path to the config file> 
    --outputDir <path to the output directory>
    --weightsFile <name of the weights file>
```

### Training on your own dataset
Use `tools/train.py` to train your model.
```
python train.py 
    --annTrainFile <path to the train json file>
    --imgTrainFile <path to the train image file>
    --annValFile <path to the validation json file> 
    --imgValFile <path to the validation image file> 
    --configFile <path to the config file> 
    --outputDir <path to the output directory>
```


## Citation
```

```


## Acknowledgment
The code is built on [segment-anything-model](https://github.com/facebookresearch/segment-anything) and [lightning-sam](https://github.com/luca-medeiros/lightning-sam), many thanks for the Third Party Libs.
