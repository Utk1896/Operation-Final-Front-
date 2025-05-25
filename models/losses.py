import torch
import torch.nn as nn

class MaskedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target, mask=None):
        loss = self.bce(pred, target)
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum()
        return loss.mean()
