#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 14:26:55 2021

@author: kkhan
"""

import torch
import os
import numpy as np
import time
from skimage import io
import torchvision.transforms as tforms
from torch.utils.data import Dataset

class CelebMainDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root, transforms=None, ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.training_files_names = os.listdir(root)
        self.root_dir = root
        self.transform = transforms
        

    def __len__(self):
        return len(self.training_files_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.training_files_names[idx])
        image = io.imread(img_name)
        
        
        if self.transform:
            image = self.transform(image)
        image = np.array(image)
        image = np.moveaxis(image, 2, 0)
        return image, np.zeros_like(image)
        

if __name__ == "__main__":
    if not os.path.exists("/HPS/CNF/work/ffjord-rnode/data/CelebAMask-HQ/training_sets/"):
        os.mkdir("/HPS/CNF/work/ffjord-rnode/data/CelebAMask-HQ/training_sets/")
    if not os.path.exists("/HPS/CNF/work/ffjord-rnode/data/CelebAMask-HQ/test_sets/"):
        os.mkdir("/HPS/CNF/work/ffjord-rnode/data/CelebAMask-HQ/test_sets/")
    train_set = CelebMainDataset("/HPS/CNF/work/ffjord-rnode/data/CelebAMask-HQ/training/",
                             transform=tforms.Compose([
                                 tforms.ToPILImage(),
                                 tforms.Resize(32),
                                 tforms.RandomHorizontalFlip(),
                                 tforms.ToTensor()
                             ]))
    for i in range(train_set.len()):
        if i % 100 == 0:
            if i> 0 and i%1000:
                time.sleep(1)
            print(i)
        x = train_set.getitem(i)

