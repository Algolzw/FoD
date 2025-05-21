import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.utils.data as data

from . import util


class LQGTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, dataroot_LQ, dataroot_GT, size=64, phase='train'):
        super().__init__()
        self.LR_paths, self.GT_paths = None, None
        self.size = size
        self.phase = phase

        # read image list from image files
        self.LR_paths = util.get_image_paths('img', dataroot_LQ)  # LR list
        self.GT_paths = util.get_image_paths('img', dataroot_GT)  # GT list
        
        assert self.GT_paths, "Error: GT paths are empty."
        if self.LR_paths and self.GT_paths:
            assert len(self.LR_paths) == len(
                self.GT_paths
            ), "GT and LR datasets have different number of images - {}, {}.".format(
                len(self.LR_paths), len(self.GT_paths)
            )

    def __getitem__(self, index):

        GT_path, LR_path = None, None
        size = self.size

        # get GT image
        GT_path = self.GT_paths[index]
        img_GT = util.read_img(None, GT_path, None)  # return: Numpy float32, HWC, BGR, [-1,1]

        # get LR image
        LR_path = self.LR_paths[index]
        img_LR = util.read_img(None, LR_path, None)

        if self.phase == "train":
            H, W, C = img_LR.shape
            assert img_GT.shape[0] == img_LR.shape[0], "GT size does not match LR size"

            # randomly crop
            rnd_h = random.randint(0, max(0, H - size))
            rnd_w = random.randint(0, max(0, W - size))
            img_GT = img_GT[rnd_h : rnd_h + size, rnd_w : rnd_w + size, :]
            img_LR = img_LR[rnd_h : rnd_h + size, rnd_w : rnd_w + size, :]

            # augmentation - flip, rotate
            img_LR, img_GT = util.augment(
                [img_LR, img_GT], hflip=True, rot=False, mode='LQGT')

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LR = img_LR[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        ).float()
        img_LR = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_LR, (2, 0, 1)))
        ).float()

        return {"LQ": img_LR, "GT": img_GT, "LQ_path": LR_path, "GT_path": GT_path}

    def __len__(self):
        return len(self.GT_paths)
