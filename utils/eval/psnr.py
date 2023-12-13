#!/usr/bin/env python3

import math
from torch import nn

def psnr(output, target):
    mse = nn.functional.mse_loss(output, target, reduction='mean').item()
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr