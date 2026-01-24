import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np


class LiTSSliceDataset(Dataset):
    def __init__(self, X_paths, y_paths, low=-100, high=400):
        self.X_paths = X_paths
        self.y_paths = y_paths
        self.low = low
        self.high = high

    def __len__(self):
        return len(self.X_paths)

    def _windowing(self, img):
        img = np.clip(img, self.low, self.high)
        img = (img - self.low) / (self.high - self.low)  # -> [0,1]
        return img

    def __getitem__(self, index):
        ct = nib.load(self.X_paths[index]).get_fdata()
        seg = nib.load(self.y_paths[index]).get_fdata()

        z = ct.shape[2] // 2
        ct2d = ct[:, :, z]
        seg2d = seg[:, :, z]

        ct2d = self._windowing(ct2d)


        tumor2d = (seg2d == 2).astype(np.float32)

        x = torch.from_numpy(ct2d).float().unsqueeze(0)  
        y = torch.from_numpy(tumor2d).float()                    
        return x, y