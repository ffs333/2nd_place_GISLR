import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def get_loss_func(cfg):
    if cfg.loss == 'ce':
        if cfg.use_loss_wgt:
            bad_wgt = np.load(cfg.base_path + 'bad_weights.npy')
            common_wgt = np.load(cfg.base_path + 'common_weights.npy')

            full_wgt = torch.tensor(bad_wgt ** cfg.pw_bad + common_wgt ** cfg.pw_com, dtype=torch.float32)
            full_wgt = full_wgt.to(cfg.device)
        else:
            full_wgt = None

        loss_fn = nn.CrossEntropyLoss(weight=full_wgt, label_smoothing=cfg.label_smooth)
    elif cfg.loss == 'focal':
        loss_fn = FocalLoss(alpha=cfg.alpha, gamma=cfg.gamma)

    else:
        raise ValueError('Error in "get_loss_func" function:',
                         f'Wrong loss name. Choose one from ["ce"] ')

    return loss_fn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

