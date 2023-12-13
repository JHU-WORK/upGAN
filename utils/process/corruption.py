import random
import numpy as np

class Corruption(object):
    def __init__(self):
        self.blackout_probability = 1
        self.max_blackout_size = 0.2
        
        self.noise_probability = 1
        self.noise_std = 0.02

    def __call__(self, img):
        if random.random() < self.noise_probability:
            img = img + np.random.normal(0, self.noise_std, img.shape)
            img = np.clip(img, 0, 255)
        
        if random.random() < self.blackout_probability:
            h, w = img.shape[:2]
            blackout_size = int(min(h, w) * self.max_blackout_size)

            # Randomly select a region to blackout
            top = random.randint(0, h - blackout_size)
            left = random.randint(0, w - blackout_size)

            # Blackout the selected region
            img[top:top + blackout_size, left:left + blackout_size, :] = 0
        return img