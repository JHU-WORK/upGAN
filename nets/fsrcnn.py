import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FSRCNN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, d=56, s=12, m=4):
        """
        Initialize the FSRCNN model.
        :param scale_factor: Upscaling factor.
        :param num_channels: Number of channels of the input image.
        :param d: Number of feature maps in the first and last part.
        :param s: Number of feature maps in the shrinking and expanding layers.
        :param m: Number of mapping layers.
        """
        super(FSRCNN, self).__init__()

        # Feature extraction layer
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=2),
            nn.PReLU(d)
        )

        # Shrinking layer
        self.shrink = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s)
        )

        # Non-linear Mapping layers
        self.map = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.PReLU(s)
            ) for _ in range(m)])

        # Expanding layer
        self.expand = nn.Sequential(
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d)
        )

        # Deconvolution layer for upscaling
        self.deconv = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=4, output_padding=scale_factor - 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for the convolutional layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.first_part(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x