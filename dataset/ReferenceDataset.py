#!/usr/bin/env python3

import os
from skimage import io
from skimage.transform import rescale
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ReferenceDataset(Dataset):
    def __init__(self, data_dir, transform=transforms.Compose([transforms.ToTensor(),]), scale_factor=2, imageSize=800):
        self.data_dir = data_dir
        self.transform = transform
        self.scale_factor = scale_factor
        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        self.length = len(files)
        self.toTensor = transforms.ToTensor()
        self.imageSize = imageSize

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fileName = "i" + str(idx).zfill(6) + ".jpg"
        img_path = os.path.join(self.data_dir, fileName)
        
        high_res = io.imread(img_path)

        offset = int((high_res.shape[1] - self.imageSize) / 2)
        high_res = high_res[:self.imageSize, offset : offset + self.imageSize]  #Cropping to center top

        extraHeight = high_res.shape[0] % self.scale_factor
        extraWidth = high_res.shape[1] % self.scale_factor
        high_res = high_res[:high_res.shape[0] - extraHeight, :high_res.shape[1] - extraWidth] #Make sure final size is multiple of scale factor for clean scaling
        
        # TODO: JPEG Png
        low_res = rescale(high_res.copy(), 1 / self.scale_factor, anti_aliasing=True, channel_axis=2)
        return self.transform(low_res).float(), self.toTensor(high_res).float()
       