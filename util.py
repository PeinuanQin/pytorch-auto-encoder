"""
 @file: util.py
 @Time    : 2023/1/11
 @Author  : Peinuan qin
 """
import random
import torch
from torch.utils.data import Dataset, Subset
import numpy as np


class MyDataset(Dataset):
    def __init__(self, dataset, ratio=0.2, add_noise=True):
        self.dataset = dataset
        self.add_noise = add_noise
        if ratio:
            random_indexs = random.sample(range(len(dataset)), int(ratio * len(dataset)))
            self.dataset = Subset(dataset, random_indexs)
            print(f"using a small dataset with ratio: {ratio}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # noise image as the encoder-decoder input, and the clean image as the groundtruth label
        if self.add_noise:
            return self.make_noise(self.dataset[item][0]), self.dataset[item][0]
        else:
            return self.dataset[item][0], self.dataset[item][0]

    def make_noise(self, x):
        """
        generate gaussian noise to make noised data for encoder
        :param x:
        :return:
        """
        noise = np.random.normal(0, 1, size=x.size())
        noise = torch.from_numpy(noise)
        x += noise
        return x

