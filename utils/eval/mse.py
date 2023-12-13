#!/usr/bin/env python3

from torch import nn

def mse_loss(output, target):
    criterion = nn.MSELoss(reduction='sum')
    loss = criterion(output, target)
    return loss.item() / output.numel()  # Average over all elements