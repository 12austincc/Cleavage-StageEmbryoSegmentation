import os

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

cls_to_id = {
    "blastomere": 1,
    "fragment": 2,
}

class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        if train: 
            image_ids = []
        else : image_ids = []
        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0
                          and image_id not in image_ids]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        ins_masks = []
        sem_masks = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            mask = self.coco.annToMask(ann)
            if ann['category_id'] == cls_to_id['fragment']:
                sem_masks.append(mask)
            else:
                # only finetune for blastomere
                bboxes.append([x, y, x + w, y + h])
                ins_masks.append(mask)
        if len(sem_masks) >= 1:
            #将fragment的实例mask合成一张语义mask
            sem_mask = torch.tensor(np.bitwise_or.reduce(sem_masks,axis=0))
        else:
            sem_mask = torch.zeros(image.shape[0],image.shape[1])
        if len(bboxes) == 0 :
            bboxes = None
            print("wrongid:",image_id)
        if self.transform:
            image, ins_masks, bboxes, sem_mask = self.transform(image, ins_masks, np.array(bboxes), sem_mask)

        bboxes = np.stack(bboxes, axis=0)
        ins_masks = np.stack(ins_masks, axis=0)
        return image, torch.tensor(bboxes), torch.tensor(ins_masks).float(),torch.tensor(sem_mask)


def collate_fn(batch):
    images, bboxes, ins_masks,sem_mask = zip(*batch)
    images = torch.stack(images)
    sem_mask = torch.stack(sem_mask)
    return images, bboxes, ins_masks,sem_mask


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, ins_masks, bboxes,sem_mask):
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        ins_masks = [torch.tensor(self.transform.apply_image(mask)) for mask in ins_masks]
        sem_mask = torch.tensor(self.transform.apply_image(sem_mask))
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        image = transforms.Pad(padding)(image)
        ins_masks = [transforms.Pad(padding)(mask) for mask in ins_masks]
        sem_mask = transforms.Pad(padding)(sem_mask)

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]

        return image, ins_masks, bboxes,sem_mask


def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = COCODataset(root_dir=cfg.dataset.train.root_dir,
                        annotation_file=cfg.dataset.train.annotation_file,
                        transform=transform)
    val = COCODataset(root_dir=cfg.dataset.val.root_dir,
                      annotation_file=cfg.dataset.val.annotation_file,
                      transform=transform, train=False)
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=True,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)
    return train_dataloader, val_dataloader


if __name__ == '__main__':
    from config import cfg
    train,val = load_datasets(cfg,1024)

