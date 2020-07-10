import os
import cv2
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy

class CustomDataset(Dataset):
    def __init__(self, data_dir, mask_dir, img_size, transform=None):
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.imgs = os.listdir(data_dir)
        self.masks = os.listdir(mask_dir)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_dir = os.path.join(self.data_dir, self.imgs[idx])
        mask_dir = os.path.join(self.mask_dir, self.masks[random.randint(0, 999)])

        img = cv2.cvtColor(cv2.imread(img_dir), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_dir), cv2.COLOR_BGR2RGB)
        
        h, w = self.img_size
        img_2x = cv2.resize(img, (h, w))
        img = cv2.resize(img, self.img_size)
        mask_2x = cv2.resize(mask, (h, w))
        mask = cv2.resize(mask, self.img_size)

        masked = deepcopy(img)
        mask = mask // 230
        mask_2x = mask_2x // 230
        masked[mask == 0] = 255

        img_2x = img_2x * 1./255
        img_2x = np.transpose(img_2x, (2, 0, 1))
        img = img * 1./255
        img = np.transpose(img, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        mask_2x = np.transpose(mask_2x, (2, 0, 1))
        masked = masked * 1./255
        masked = np.transpose(masked, (2, 0, 1))

        if self.transform:
            img_2x = self.transform(img_2x)
            img = self.transform(img)
            mask = self.transform(mask)
            mask_2x = self.transform(mask_2x)
        return img, mask, masked, img_2x, mask_2x
