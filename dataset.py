# Import libraries: --->
import os
import h5py
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

# Import constants: --->
from const import GRAY, GT_DENSITY_MAP_ROOT

class CrowdDataset(Dataset):
    def __init__(self, root_dir, split = 'train'):
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.density_map_dir = os.path.join(root_dir, split, GT_DENSITY_MAP_ROOT)

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.density_map_files = sorted([f for f in os.listdir(self.density_map_dir) if f.endswith('.h5')])

        # Check whether equal number of images and corresponding density map files are available or not: --->
        assert len(self.image_files) == len(self.density_map_files), "Mismatch in dataset size"

        # Using only ToTensor as resizing and color conversion is handled by OpenCV
        self.transform = transforms.ToTensor() # Converts numpy array (H, W, C) in range [0, 255] to a torch.FloatTensor of shape (C, H, W) in range [0.0, 1.0]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and preprocess image: --->
        img_path = os.path.join(self.image_dir, self.image_files[idx])

        if GRAY:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        elif not GRAY:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        else:
            print(f"The value of GRAY = {GRAY} is invalid!")
            quit()

        image = self.transform(image)           # shape: [1, 224, 224] if GRAY; else shape: [3, 224, 224]

        # Load density map (.h5): --->
        base_name = os.path.splitext(self.image_files[idx])[0]
        gt_path = os.path.join(self.density_map_dir, base_name + '.h5')

        with h5py.File(gt_path, 'r') as f:
            density = np.asarray(f['density'])

        # Convert the density map from .h5 format to monochrome image of size 224 x 224 x 1: --->
        density = cv2.resize(density, (224, 224), interpolation=cv2.INTER_LINEAR)
        density = density.astype(np.float32)
        density = torch.from_numpy(density).unsqueeze(0)    #[1, H, W]

        return image, density
