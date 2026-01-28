from torch.utils.data import Dataset


class LiTSPatchDataset(Dataset):
    def __init__(self, X_paths, y_paths, patch_size=256, low=-100, high=400):
        self.X_paths = X_paths
        self.y_paths = y_paths
        self.patch_size = patch_size
        self.low = low
        self.high = high

        self.samples = []  # (vol_path, seg_path, z, cx, cy)

        for vol_path, seg_path in zip(X_paths, y_paths):
            seg_nii = nib.load(seg_path)
            seg = seg_nii.dataobj  # lazy

            for z in range(seg.shape[2]):
                mask = (seg[..., z] == 2)
                if not np.any(mask):
                    continue

                ys, xs = np.where(mask)
                cy, cx = ys.mean().astype(int), xs.mean().astype(int)

                self.samples.append((vol_path, seg_path, z, cx, cy))

    def __len__(self):
        return len(self.samples)

    def _windowing(self, img):
        img = np.clip(img, self.low, self.high)
        img = (img - self.low) / (self.high - self.low)
        return img

    def __getitem__(self, idx):
        vol_path, seg_path, z, cx, cy = self.samples[idx]

        ct = nib.load(vol_path).dataobj[..., z]
        seg = nib.load(seg_path).dataobj[..., z]

        ps = self.patch_size // 2
        H, W = ct.shape

        x1 = max(cx - ps, 0)
        y1 = max(cy - ps, 0)
        x2 = min(cx + ps, W)
        y2 = min(cy + ps, H)

        ct_patch = ct[y1:y2, x1:x2]
        seg_patch = (seg[y1:y2, x1:x2] == 2).astype(np.float32)

        # pad si on est au bord
        ct_patch = np.pad(ct_patch,
                          ((0, self.patch_size - ct_patch.shape[0]),
                           (0, self.patch_size - ct_patch.shape[1])),
                          mode="constant")
        seg_patch = np.pad(seg_patch,
                           ((0, self.patch_size - seg_patch.shape[0]),
                            (0, self.patch_size - seg_patch.shape[1])),
                           mode="constant")

        ct_patch = self._windowing(ct_patch)

        x = torch.from_numpy(ct_patch).float().unsqueeze(0)
        y = torch.from_numpy(seg_patch).float().unsqueeze(0)

        return x, y
