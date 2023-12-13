#!/usr/bin/env python3

import os
from skimage import io
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class InferenceDataset(Dataset):
    def __init__(self, data_dir, file_list, transform=transforms.Compose([transforms.ToTensor(),])):
        self.data_dir = data_dir
        self.transform = transform
        with open(file_list) as file:
            self.files = [line.rstrip() for line in file]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        img_path = os.path.join(self.data_dir, file)
        high_res = io.imread(img_path)

        return self.transform(high_res)