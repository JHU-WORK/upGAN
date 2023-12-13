import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels=12, dropout_rate=0.5):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual  # Adding input to the output
        out = self.relu(out)
        return out



class ResFSRCNN(nn.Module):
    def __init__(self, scale_factor=2, num_channels=3, layers=4, dropout_rate=0.5):
        """
        Enhanced FSRCNN network with ReLU, Batch Normalization, and Dropout.
        :param scale_factor: Upscaling factor.
        :param num_channels: Number of channels of the input image.
        :param layers: Number of mapping layers.
        :param dropout_rate: Dropout rate for the Dropout layers.
        """
        super(ResFSRCNN, self).__init__()

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

        self.map = nn.Sequential(*[ResidualBlock(channels=12, dropout_rate=dropout_rate) for _ in range(layers)])

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
        x = self.map(x) 
        x = self.expand(x)
        x = self.deconv(x)
        return x