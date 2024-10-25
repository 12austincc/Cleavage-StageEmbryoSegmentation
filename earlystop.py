import glob

import numpy as np
import torch
import os


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, k=2, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.k = k

    def __call__(self, score, model, fabric, epoch):

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score,model,epoch,fabric)
            
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score,model,epoch,fabric)
            self.counter = 0

        self.saveTopK(self.k)

    def save_checkpoint(self, score, model, epoch, fabric):
        '''Saves model when validation loss decrease.'''
        fabric.print(f"best_score={self.best_score} Saving checkpoint to {self.save_path}") 
        state_dict = [model.sam.state_dict(),model.semantic_decoder.state_dict()]
        if fabric.global_rank == 0:
            torch.save(state_dict, os.path.join(self.save_path, f"epoch-{epoch:06d}-score{score:.6f}-ckpt.pth")) 
        

    def get_checkpoint_files(self):
        checkpoint_files = glob.glob(os.path.join(self.save_path, "*-ckpt.pth"))
        sorted_ckpt = sorted(checkpoint_files, key=lambda x: float(x.split('-score')[1].split('-')[0]))
        return sorted_ckpt

    def saveTopK(self, k=4):
        ckpts = self.get_checkpoint_files()
        num_ckpt = len(ckpts)
        if num_ckpt > k:
            for i in range(num_ckpt - k):
                os.remove(ckpts[i])

