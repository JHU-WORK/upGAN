#!/usr/bin/env python3

import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()

        # Ensure reduction ratio does not lead to zero output channels
        if in_channels // reduction_ratio == 0: self.reduced_channels = 1
        else: self.reduced_channels = in_channels // reduction_ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, self.reduced_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(self.reduced_channels, in_channels, 1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out)  # This will be a tensor with shape [batch_size, channels, 1, 1]
        out = x * scale  # Element-wise multiplication to apply attention weights across channels
        return out

class AttnFSRCNN(nn.Module):
    def __init__(self, scale_factor=2, num_channels=3, layers=4, dropout_rate=0.5):
        """
        Enhanced FSRCNN network with ReLU, Batch Normalization, and Dropout.
        :param scale_factor: Upscaling factor.
        :param num_channels: Number of channels of the input image.
        :param layers: Number of mapping layers.
        :param dropout_rate: Dropout rate for the Dropout layers.
        """
        super(AttnFSRCNN, self).__init__()

        # Feature extraction layer
        self.first_part = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=56, kernel_size=5, padding=2),
            nn.BatchNorm2d(56),
            nn.ReLU()
        )

        # Shrinking layer
        self.shrink = nn.Sequential(
            nn.Conv2d(in_channels=56, out_channels=12, kernel_size=1),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )

        # Mapping layers with ReLU, BatchNorm and Dropout
        self.map = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1),
                nn.BatchNorm2d(12),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                ChannelAttention(12)  # Channel Attention after each mapping layer
            ) for _ in range(layers)])

        # Expanding layer
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=56, kernel_size=1),
            nn.BatchNorm2d(56),
            nn.ReLU()
        )

        # Deconvolution layer for upscaling
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=56, out_channels=num_channels, kernel_size=9, stride=scale_factor, padding=4, output_padding=scale_factor - 1),
        )

    def forward(self, x):
        x = self.first_part(x)
        x = self.shrink(x)
        for mapping_layer in self.map:
            x += mapping_layer(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x
    