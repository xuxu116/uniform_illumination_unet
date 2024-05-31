import logging
import os
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import glob
import cv2


class BasicDataset(Dataset):
    def __init__(self, images_dir="./result", size=(768,768)):
        images = [img for ext in ("*.png", "*.jpg", "*.jpeg", "*.gif")
                for img in glob.glob(os.path.join(images_dir, "**", ext), recursive=True)]
        self.images = [img for img in images if  "matting" not in img and "no_light" not in img]
        logging.info("Target lighting imgs %d from total file %d"%(len(images), len(self.images)))
        self.size = size

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        source_path = self.images[idx]
        mask_path = os.path.join(os.path.dirname(source_path), "matting.png")
        gt_path = os.path.join(os.path.dirname(source_path), "no_light.jpg")

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        source = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)

        gt = cv2.resize(gt, self.size)
        mask = cv2.resize(mask, self.size)
        source = cv2.resize(source, self.size)

        mask = mask[:,:,None].astype("float32") / 255
        gt_lab = cv2.cvtColor(gt, cv2.COLOR_BGR2LAB).astype("float32") / 255
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32") / 255

        l_mean_gt, a_mean_gt, b_mean_gt = np.sum(mask * gt_lab, axis=(0,1)) / np.sum(mask)
        l_std_gt, a_std_gt, b_std_gt = np.sqrt(np.sum(mask * (gt_lab - [l_mean_gt, a_mean_gt, b_mean_gt])**2, axis=(0,1)) / np.sum(mask))

        l_mean_source, a_mean_source, b_mean_source = np.sum(mask * source_lab, axis=(0,1)) / np.sum(mask)
        l_std_source, a_std_source, b_std_source = np.sqrt(np.sum(mask * (source_lab - [l_mean_source, a_mean_source, b_mean_source])**2, axis=(0,1)) / np.sum(mask))

        gt2source = (gt_lab - [l_mean_gt, a_mean_gt, b_mean_gt]) * [l_std_source/l_std_gt, a_std_source/a_std_gt, b_std_source/b_std_gt] + [l_mean_source, a_mean_source, b_mean_source]

        offset = gt2source - source_lab

        source_lab = source_lab.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        offset = offset.transpose((2, 0, 1))
        gt_lab = gt_lab.transpose((2, 0, 1))

        return torch.as_tensor(source_lab.copy()).float().contiguous(), torch.as_tensor(mask.copy()).float().contiguous(), torch.as_tensor(offset.copy()).float().contiguous(), torch.as_tensor(gt_lab.copy()).float().contiguous()

if __name__ == '__main__':
    dataset = BasicDataset()
    source_lab, mask, offset, gt = dataset[0]

    source_lab_np, mask_np, offset_np, gt_np = source_lab.numpy(), mask.numpy(), offset.numpy(), gt.numpy()
    target = source_lab_np + offset_np

    source_cv = cv2.cvtColor(np.clip(source_lab_np*255, 0 ,255).astype('uint8'), cv2.COLOR_Lab2BGR)
    target_cv = cv2.cvtColor(np.clip(target*255, 0 ,255).astype('uint8'), cv2.COLOR_Lab2BGR)
    gt_cv = cv2.cvtColor(np.clip(gt_np*255, 0 ,255).astype('uint8'), cv2.COLOR_Lab2BGR)

    cv2.imwrite("source.jpg", source_cv)
    cv2.imwrite("target.jpg", target_cv)
    cv2.imwrite("gt.jpg", gt_cv)
