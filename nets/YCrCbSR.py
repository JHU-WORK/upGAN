#!/usr/bin/env python3

import torch.nn as nn
import cv2 as cv
import torch
from ..utils.process import colors

class YCrCbSR(nn.Module):
    def __init__(self, scale):
        super(YCrCbSR, self).__init__()

        # SRCNN layers
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.scale = scale

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        '''x = colors.rgb2ycbcr(x)
        #x = cv.cvtColor(x.numpy(force=True), cv.COLOR_RGB2YCrCb)
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=True)
        
        Y = (x.clone()[:, 0, :, :]).unsqueeze(1)
        #print(x.size())
        #print(.size())
        Y = self.relu(self.conv1(Y))
        Y = self.relu(self.conv2(Y))
        Y = self.conv3(Y)
        x[:, 0, :, :] = Y[:, 0, :, :]
        return colors.ycbcr2rgb(x)'''
        x = colors.rgb2ycbcr(x)
        #x = cv.cvtColor(x.numpy(force=True), cv.COLOR_RGB2YCrCb)
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=True)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return colors.ycbcr2rgb(x)