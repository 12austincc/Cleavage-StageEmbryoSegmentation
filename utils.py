import json
import os
import random

import cv2
import torch
from box import Box
from dataset import COCODataset
from model import Model
from predictor import ModelPredictor
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm
import random
from medpy import metric
from resToCoco import segmentationToCocoResult


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_dice(tp, fp, fn, tn, ep=0.1):
    tp = tp.sum()
    fp = fp.sum()
    fn = fn.sum()
    tn = tn.sum()
    dice = torch.mean((2 * tp) / (2 * tp + fp + fn + ep))
    return dice


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def calc_metric(pred,gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred,gt)
        hd95 = metric.binary.hd95(pred,gt)
        return dice,hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1,0
    else:
        return 0,0

def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if boxes is not None:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if masks is not None:
        # image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
        image = draw_segmentation_masks(image, masks=masks, alpha=alpha)  # random color
    return image.numpy().transpose(1, 2, 0)


def visualize(cfg: Box):
    model = Model(cfg)
    model.setup()
    model.eval()
    model.cuda()
    dataset = COCODataset(root_dir=cfg.dataset.val.root_dir,
                          annotation_file=cfg.dataset.val.annotation_file,
                          transform=None)
    predictor = ModelPredictor(model)
    os.makedirs(cfg.out_dir, exist_ok=True)
    res_anns = []
    for image_id in tqdm(dataset.image_ids):
        image_info = dataset.coco.loadImgs(image_id)[0]
        image_path = os.path.join(dataset.root_dir, image_info['file_name'])
        image_output_path = os.path.join(cfg.out_dir, cfg.visualize_dir, image_info['file_name'])
        image_output_dir = os.path.join(cfg.out_dir, cfg.visualize_dir)
        if not os.path.exists(image_output_dir):
            os.mkdir(image_output_dir)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ann_ids = dataset.coco.getAnnIds(imgIds=image_id)
        anns = dataset.coco.loadAnns(ann_ids)
        bboxes = []
        cls = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if ann['category_id'] != 2:
                bboxes.append([x, y, x + w, y + h])
                cls.append(ann['category_id'])
        bboxes = torch.as_tensor(bboxes, device=model.sam.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(bboxes, image.shape[:2])
        predictor.set_image(image)
        masks, _, _,sem_mask = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        concat_mask = torch.cat((masks.squeeze(1) , sem_mask.squeeze(0)),dim=0)
        cls.append(2)
        score = [1] * len(cls)
        res_anns += segmentationToCocoResult(concat_mask,cls,score,image_id)
        image_output = draw_image(image, concat_mask, boxes=bboxes, labels=None)
        cv2.imwrite(image_output_path, image_output)
    with open(cfg.out_dir+'/res.json','w') as res_json:
        json.dump(res_anns,res_json,indent=1)


if __name__ == "__main__":
    from config import cfg

    visualize(cfg)
