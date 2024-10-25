# Cleavage-Stage Embryo Segmentation Using SAM-Based Dual Branch Pipeline: Development and Evaluation with the CleavageEmbryo Dataset

This is the official pytorch implementation of SAM-Based Dual Branch Pipeline, please refer the [paper](url) for more details.

## Dataset
CleavageEmbryo Dataset.
[]()
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
[sam_dual](https://whueducn-my.sharepoint.com/:u:/g/personal/2020302111430_whu_edu_cn/EVPYa9MhqG5FjPuOsNRNvY4Bepec4ZXdjKpZtum0Gq0uaQ?e=Rk1JNl)
[yolo](https://whueducn-my.sharepoint.com/:u:/g/personal/2020302111430_whu_edu_cn/EUaMh7yAWKpHubH-QNWtJfwBh3f7mvYfM-bVk1o7fYtfRw?e=dVINhh)
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
