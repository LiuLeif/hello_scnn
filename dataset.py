#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 2021-12-30 19:14
import json
import os

import pickle
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import util
import config
from collections import defaultdict

from data_augment import flip


class Tusimple(Dataset):
    def __init__(self, mode):
        super(Tusimple, self).__init__()
        self.mode = mode
        self.load_data()
        self.cache = {}

    def load_data(self):
        label_file = os.path.join("label", "{}.dat".format(self.mode))
        with open(label_file, "rb") as f:
            self.label_data = pickle.load(f)

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        label_data = self.label_data[idx]
        image_file = os.path.join(config.DATA_PATH, label_data[0])
        label_file = label_data[1]

        exist = label_data[2]
        exist = torch.from_numpy(np.array(exist)).type(torch.float32)

        img = self.cache.get(image_file)
        if img is None:
            img = cv2.imread(image_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = util.normalize(
                util.to_tensor(util.resize(img, (config.IMAGE_W, config.IMAGE_H))),
                config.MEAN,
                config.STD,
            )
            self.cache[image_file] = img

        label = self.cache.get(label_file)
        if label is None:
            label = cv2.imread(label_file)
            label = label[:, :, 0]
            label = util.resize(label, (config.IMAGE_W, config.IMAGE_H))
            label = torch.from_numpy(label).type(torch.long)
            self.cache[label_file] = label

        sample = {
            "img": img,
            "label": label,
            "exist": exist,
        }
        if self.mode == "test":
            return sample
        return (sample, flip(sample))


if __name__ == "__main__":
    data = Tusimple("train")
    print(data[1])
