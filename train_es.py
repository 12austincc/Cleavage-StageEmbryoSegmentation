import os
import time

import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from config import cfg
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from losses import DiceLoss, FocalLoss, SemDiceLoss
from model import Model
from torch.utils.data import DataLoader
from utils import AverageMeter
from utils import calc_iou, calc_dice
from earlystop import EarlyStopping
from medpy import metric
torch.set_float32_matmul_precision('high')


def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, early_stop: EarlyStopping = None,
             epoch: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()
    recalls = AverageMeter()
    precisions = AverageMeter()
    dices = AverageMeter()
    sem_dices = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):

            images,  bboxs, ins_masks, sem_mask = data
            num_images = images.size(0)

            sem_pred_mask, pred_masks, _ = model(images, bboxs)
            sem_pred_mask = np.array((torch.argmax(torch.softmax(sem_pred_mask, dim=1), dim=1, keepdim=True) > 0).cpu())

            if not torch.all(sem_mask == 0):
                dice_sem = metric.dc(sem_pred_mask,np.array(sem_mask.cpu()))
                sem_dices.update(dice_sem,num_images)

            
            for pred_mask, gt_mask in zip(pred_masks, ins_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )

                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                batch_recall = smp.metrics.recall(*batch_stats, reduction="micro-imagewise")
                batch_precision = smp.metrics.precision(*batch_stats, reduction="micro-imagewise")
                batch_dice = calc_dice(*batch_stats)

                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
                recalls.update(batch_recall, num_images)
                precisions.update(batch_precision, num_images)
                dices.update(batch_dice, num_images)
            # fabric.print(
            #     f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            # )

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
                 f'--Dice:[{dices.avg:.4f}]--SemDice:[{sem_dices.avg: .4f}]')
    model.to('cuda')
    score = dices.avg + sem_dices.avg
    if early_stop is not None:
        early_stop(score, model, fabric, epoch)
    model.train()


def train_sam(
        cfg: Box,
        fabric: L.Fabric,
        model: Model,
        optimizer: _FabricOptimizer,
        scheduler: _FabricOptimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
):
    """The SAM training loop."""

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    sem_dice_loss = SemDiceLoss(2)
    early_stop = EarlyStopping(cfg.out_dir, cfg.patience)
    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        dice_losses_sem = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()

        if epoch > 1 and epoch % cfg.eval_interval == 0:
            validate(fabric, model, val_dataloader, early_stop, epoch)
        if early_stop.early_stop:
            fabric.print("--EarlyStop--")
            break
        for iter, data in enumerate(train_dataloader):

            data_time.update(time.time() - end)
            images, bboxs, ins_masks, sem_mask = data
            batch_size = images.size(0)
            sem_pred_mask, pred_masks, iou_predictions = model(images, bboxs)
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice_ins = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, ins_masks, iou_predictions):
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask, num_masks)
                loss_dice_ins += dice_loss(pred_mask, gt_mask, num_masks)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks


            loss_dice_sem = sem_dice_loss(sem_pred_mask, sem_mask)
            loss_total = loss_focal + loss_dice_ins + loss_iou + loss_dice_sem
            # loss_total = loss_dice_sem
            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice_ins.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            dice_losses_sem.update(loss_dice_sem.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)

            fabric.print(f'Epoch: [{epoch}][{iter + 1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                         f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                         f' | Sem Dice Loss [{dice_losses_sem.val:.4f} ({dice_losses_sem.avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')
        fabric.log_dict(
            {"focal_loss": focal_losses.avg, "dice_loss": dice_losses.avg, "sem_dice_loss": dice_losses_sem.avg,
             "iou_loss": iou_losses.avg, "total_loss": total_losses.avg},
            step=epoch
        )


def configure_opt(cfg: Box, model: Model):
    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor ** 2)

    parameters = [
        {'params': model.sam.parameters(), 'lr': cfg.opt.learning_rate_sam, 'weight_decay': cfg.opt.weight_decay},
        {'params': model.semantic_decoder.parameters(), 'lr': cfg.opt.learning_rate_sem,'weight_decay':cfg.opt.weight_decay}
    ]
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    optimizer = torch.optim.Adam(parameters)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def main(cfg: Box) -> None:
    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir, name="lightning-sam")])
    fabric.launch()
    fabric.seed_everything(1344 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    import shutil
    shutil.copy('config.py', cfg.out_dir)
    shutil.copy('semantic_decoder.py',cfg.out_dir)

    train_data, val_data = load_datasets(cfg, model.sam.image_encoder.img_size)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    validate(fabric, model, val_data, epoch=0)


if __name__ == "__main__":
    main(cfg)
