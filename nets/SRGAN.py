import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, doubling_number, B):
        super(Generator, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4), 
            nn.PReLU()
        )
         
        self.block2 = nn.Sequential(*[ResidualBlock(64) for i in range(B)])
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64)
        )
        
        self.block4 = nn.Sequential(*[UpsampleBlock(64, 2) for i in range(doubling_number)])
        
        self.block5 = nn.Conv2d(64, 3, kernel_size=9, padding=4)
        

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2) + out1
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        
        return (torch.tanh(out5) + 1) / 2   #Range [0, 1]


class ResidualBlock(nn.Module):
    def __init__(self, channels): #channels will be 64
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.batchNorm1(y)
        y = self.prelu(y)
        y = self.conv2(y)
        y = self.batchNorm2(y)
        return x + y
    

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (up_scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.leaky = nn.LeakyReLU(0.2)
        
        self.blocks = nn.Sequential(
            ConvBlock(64, 64, 2),
            ConvBlock(64, 128, 1),
            ConvBlock(128, 128, 2),
            ConvBlock(128, 256, 1),
            ConvBlock(256, 256, 2),
            ConvBlock(256, 512, 1),
            ConvBlock(512, 512, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(512, 1024, kernel_size=1), 
            nn.LeakyReLU(0.2), 
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky(x)
        x = self.blocks(x)
        x = self.classifier(x)
        
        batch_size = x.size(0)
        return torch.sigmoid(x.view(batch_size))

    
class ConvBlock(nn.Module):
    def __init__(self, channelsIn, channelsOut, strideLength):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(channelsIn, channelsOut, kernel_size=3, stride=strideLength, padding=1)
        self.batchNorm = nn.BatchNorm2d(channelsOut)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.leaky(x)
        return x