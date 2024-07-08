# Cleavage-Stage Embryo Segmentation Using SAM-Based Dual Branch Pipeline: Development and Evaluation with the CleavageEmbryo Dataset

This is the official pytorch implementation of SAM-Based Dual Branch Pipeline, please refer the [paper](url) for more details.


## Introduction
### Abstract
**Motivation:** Embryo selection is one of the critical factors in determining the success of pregnancy in in vitro fertilization (IVF) techniques. Using artificial intelligence to aid in embryo selection could effectively address the current time-consuming, expensive and subjectively influenced process of embryo assessment by trained embryologists. However, current deep learning-based methods often focus on the segmentation or grading of blastocysts,  neglecting morphokinetic parameters or predicting cell development via time-lapse videos, thus lacking interpretability. Given the significance of morphokinetic and morphological evaluation of cleavage-stage embryos in predicting implantation potential, as emphasized by previous research, there is a necessity for an automated method to segment cleavage-stage embryos to improve this process.
**Results:** In this article, we introduce the SAM-based Dual Branch Segmentation Pipeline for automated segmentation of blastomeres in cleavage-stage embryos. Leveraging the powerful segmentation capability of SAM, the instance branch conducts instance segmentation of blastomeres, while the semantic branch performs semantic segmentation of fragments. Due to the lack of publicly available datasets, we constructed the CleavageEmbryo dataset, the first dataset of human cleavage-stage embryos with pixel-level annotations containing fragment information. We train and test a series of state-of-the-art segmentation algorithms on CleavageEmbryo. Our experiments demonstrate that our method outperforms existing algorithms in terms of objective metrics (mAP 0.748 on blastomere, Dice 0.694 on fragment) and visual quality, enabling more accurate segmentation of cleavage-stage embryos.
![](https://github.com/12austincc/Cleavage-StageEmbryoSegmentation/blob/main/image/overall.png)
![](https://github.com/12austincc/Cleavage-StageEmbryoSegmentation/blob/main/image/semantic.png)
![](https://github.com/12austincc/Cleavage-StageEmbryoSegmentation/blob/main/image/instance.png)

### Visualization
![](https://github.com/12austincc/Cleavage-StageEmbryoSegmentation/blob/main/image/object_detection.png)
![](https://github.com/12austincc/Cleavage-StageEmbryoSegmentation/blob/main/image/blastomereSeg.png)
![](https://github.com/12austincc/Cleavage-StageEmbryoSegmentation/blob/main/image/fragment.png)
### Results
object detection

|                        |      Blastomere       |                    |                      |                 |      Fragment       |                  |                    |               |
| ---------------------- | :-------------------: | ------------------ | -------------------- | --------------- | :-----------------: | ---------------- | ------------------ | ------------- |
| Methods                |  Precision$\uparrow$  | Recall$\uparrow$   | F1-Score$\uparrow$   | mAP$\uparrow$   | Precision$\uparrow$ | Recall$\uparrow$ | F1-Score$\uparrow$ | mAP$\uparrow$ |
| FCOS                   |         0.534         | 0.554              | 0.552                | 0.680           |        0.158        | 0.213            | 0.180              | 0.138         |
| YOLOX                  |         0.576         | 0.587              | 0.643                | 0.755           |        0.208        | 0.266            | 0.214              | 0.227         |
| DETR                   |         0.594         | 0.612              | 0.667                | 0.772           |        0.160        | 0.190            | 0.250              | 0.152         |
| Dino                   |         0.631         | 0.647              | 0.547                | 0.827           |        0.197        | 0.273            | 0.192              | 0.204         |
| Co-DETR                |         0.633         | 0.645              | 0.461                | 0.834           |        0.213        | 0.270            | 0.198              | 0.229         |
| Yolov8                 |         0.634         | 0.640              | 0.697                | 0.827           |        0.245        | 0.203            | 0.280              | 0.209         |

blatomere segmentation

|                        |      Blastomere       |                    |                      |                 |                     
| --------------------   | :-------------------: | ------------------ | -------------------- | --------------- |                 
| Methods                |  Precision$\uparrow$  | Recall$\uparrow$   | F1-Score$\uparrow$   | mAP$\uparrow$   |                  
| Cascade MaskRCNN       |         0.571         | 0.579              | 0.657                | 0.747           |                   
| Mask Scoring RCNN      |         0.549         | 0.560              | 0.635                | 0.714           |                  
| YOLACT                 |         0.468         | 0.493              | 0.571                | 0.587           | 
| SOLOv2                 |         0.560         | 0.570              | 0.647                | 0.728           | 
| Mask2Former            |         0.465         | 0.472              | 0.549                | 0.592           |
| RTMDet                 |         0.526         | 0.546              | 0.591                | 0.670           |
| DONet                  |         0.505         | 0.519              | 0.599                | 0.647           |
| Our Method(Yolov8)     |         0.590         | 0.619              | 0.683                | 0.748           |
| Our Method(*)          |         0.691         | 0.704              | 0.757                | 0.913           |

fragment segmentation 

|                        |      Blastomere       |                    |                      |                 |
| --------------------   | :-------------------: | ------------------ | -------------------- | --------------- | 
| Methods                |  Precision$\uparrow$  | Recall$\uparrow$   | F1-Score$\uparrow$   | mAP$\uparrow$   | 
| Cascade MaskRCNN       |         0.571         | 0.579              | 0.657                | 0.747           |
| Mask Scoring RCNN      |         0.549         | 0.560              | 0.635                | 0.714           |
| YOLACT                 |         0.468         | 0.493              | 0.571                | 0.587           |
| SOLOv2                 |         0.560         | 0.570              | 0.647                | 0.728           |
| Mask2Former            |         0.465         | 0.472              | 0.549                | 0.592           |
| RTMDet                 |         0.526         | 0.546              | 0.591                | 0.670           |
| DONet                  |         0.505         | 0.519              | 0.599                | 0.647           |
| Our Method(Yolov8)     |         0.590         | 0.619              | 0.683                | 0.748           |
| Our Method(*)          |         0.691         | 0.704              | 0.757                | 0.913           |

## Dataset
CleavageEmbryo Dataset.
![](https://github.com/12austincc/Cleavage-StageEmbryoSegmentation/blob/main/image/DatasetDescrip.png)
## Install
The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`.

Install Segment Anything:
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
Install YOLOv8：
```
pip install ultralytics
```
Install requirements:
```
pip install -r requirements.txt
```
## QuickStart
### Inference test images
Download [pre-trained model](https://whueducn-my.sharepoint.com/:u:/g/personal/2020302111430_whu_edu_cn/EUu7pU7KIw5OuXh_AyOtexYB2WPPRNpH3F5qbTgnR1BAzw?e=xxfFiY) and [pre-trained yolo](https://whueducn-my.sharepoint.com/:u:/g/personal/2020302111430_whu_edu_cn/ETxIEO-CFKxPsNjH4348EmQBpmu5zaav-kAuBTdxhF9HLA?e=1pFp8n)
and put them into ./checkpoint
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
