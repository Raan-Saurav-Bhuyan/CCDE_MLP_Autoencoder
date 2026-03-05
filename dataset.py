# Import libraries: --->
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
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

        if GRAY:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels = 1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),                                       # Converts to [C, H, W] and scales [0, 255] → [0, 1]
            ])
        elif not GRAY:
            self.transform = transforms.Compose([
                # transforms.Grayscale(num_output_channels = 1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),                                       # Converts to [C, H, W] and scales [0, 255] → [0, 1]
            ])
        else:
            print(f"The value of GRAY = {GRAY} is invalid!")
            quit(0)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and preprocess image: --->
        img_path = os.path.join(self.image_dir, self.image_files[idx])

        if GRAY:
            image = Image.open(img_path).convert('L')
        elif not GRAY:
            image = Image.open(img_path).convert('RGB')
        else:
            print(f"The value of GRAY = {GRAY} is invalid!")
            quit()

        image = self.transform(image)           # shape: [1, 224, 224] if L; else shape: [3, 224, 224]

        # Load density map (.h5): --->
        base_name = os.path.splitext(self.image_files[idx])[0]
        gt_path = os.path.join(self.density_map_dir, base_name + '.h5')

        with h5py.File(gt_path, 'r') as f:
            density = np.asarray(f['density'])

        # Convert the density map from .h5 format to monochrome image of size 224 x 224 x 1: --->
        density = Image.fromarray(density).resize((224, 224), resample=Image.BILINEAR)
        density = np.array(density).astype(np.float32)
        density = torch.from_numpy(density).unsqueeze(0)    #[1, H, W]

        return image, density
