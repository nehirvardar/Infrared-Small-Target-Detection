import torch.nn as nn
import numpy as np
import  torch

def SoftIoULoss( pred, target):
        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss


def TverskyLoss(pred, target, alpha=0.3, beta=0.7, smooth=1.0):
    pred = torch.sigmoid(pred)

    pred = pred.view(-1)
    target = target.view(-1)

    TP = (pred * target).sum()
    FP = ((1 - target) * pred).sum()
    FN = (target * (1 - pred)).sum()

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - Tversky

class AverageMeter(object):
    """Computes and stores the average and current value"""

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


