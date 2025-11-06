import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import random
from data.base_dataset import BaseDataset

from torch.utils.data import Dataset
from data.utils import MaskToTensor, get_params, affine_transform
from data.image_folder import make_dataset

class SpermDataset(BaseDataset):
    """A dataset class for patch-based crack dataset training."""

    def __init__(self, opt, patch_size=256, stride=128, min_label_ratio=0.00): #min_label_ratio=0.01
        self.opt = opt
        self.patch_size = patch_size
        self.stride = stride
        self.min_label_ratio = min_label_ratio  # Keep patches with >1% label pixels

        self.img_paths = make_dataset(os.path.join(opt.dataroot, f'{opt.phase}_img'))
        self.lab_dir = os.path.join(opt.dataroot, f'{opt.phase}_lab')

        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.lab_transform = MaskToTensor()

        self.patches = self.extract_all_patches()

    def extract_all_patches(self):
        all_patches = []
        for img_path in self.img_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            lab_path = os.path.join(self.lab_dir, os.path.basename(img_path).split('.')[0] + '.png')
            lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)
            if lab is None:
                continue
            if len(lab.shape) == 3:
                lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)

            img_h, img_w = lab.shape
            for y in range(0, img_h, self.stride):
                for x in range(0, img_w, self.stride):
                    # Adjust patch coordinates to not go out of bounds
                    y_end = min(y + self.patch_size, img_h)
                    x_end = min(x + self.patch_size, img_w)
                    y_start = y_end - self.patch_size
                    x_start = x_end - self.patch_size
                    if y_start < 0 or x_start < 0:
                        continue
                    patch_lab = lab[y_start:y_end, x_start:x_end]
                    label_ratio = np.mean(patch_lab > 127)
                    if label_ratio >= self.min_label_ratio:
                        all_patches.append((img_path, lab_path, x_start, y_start))
        return all_patches

    def __getitem__(self, index):
        img_path, lab_path, x, y = self.patches[index]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)

        patch_img = img[y:y+self.patch_size, x:x+self.patch_size]
        patch_lab = lab[y:y+self.patch_size, x:x+self.patch_size]

        # Binary mask
        _, patch_lab = cv2.threshold(patch_lab, 127, 255, cv2.THRESH_BINARY)

        # Apply flip
        if (not self.opt.no_flip) and random.random() > 0.5:
            if random.random() > 0.5:
                patch_img = np.fliplr(patch_img)
                patch_lab = np.fliplr(patch_lab)
            else:
                patch_img = np.flipud(patch_img)
                patch_lab = np.flipud(patch_lab)

        # Apply affine transform
        if self.opt.use_augment and random.random() > 0.5:
            angle, scale, shift = get_params()
            patch_img = affine_transform(patch_img, angle, scale, shift, self.patch_size, self.patch_size)
            patch_lab = affine_transform(patch_lab, angle, scale, shift, self.patch_size, self.patch_size)

        # 🔆 Apply random brightness adjustment
        if self.opt.use_augment and random.random() > 0.5:
            brightness_factor = random.uniform(0.7, 1.3)  # 30% darker or brighter
            pil_img = Image.fromarray(patch_img)
            patch_img = transforms.functional.adjust_brightness(pil_img, brightness_factor)
        else:
            patch_img = Image.fromarray(patch_img)

        # Convert mask to binary and tensor
        _, patch_lab = cv2.threshold(patch_lab, 127, 1, cv2.THRESH_BINARY)
        patch_lab = self.lab_transform(patch_lab.copy()).unsqueeze(0).float()
        patch_img = self.img_transforms(patch_img)

        return {'image': patch_img, 'label': patch_lab, 'A_paths': img_path, 'B_paths': lab_path, 'coord': (x, y)}

    def __len__(self):
        return len(self.patches)
