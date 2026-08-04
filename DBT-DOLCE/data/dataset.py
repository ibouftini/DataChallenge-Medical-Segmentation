"""
PyTorch Dataset for DBT breast phantom slices.

Each sample is an HDF5 file produced by data/preprocess.py containing:
  arr_img512   - ground truth attenuation slice, normalised to [0, 1], shape (1, H, W)
  arr_la_rls   - SIRT conditioning (from limited-angle sinogram),  shape (1, H, W)
Both are stored as float16 (upcast to float32 on load) to save disk. The DOLCE
"fbp" conditioning slot is identical to "rls", so it is not stored separately;
the loader reuses arr_la_rls for it (older files with arr_la_fbp still load).

The Dataset returns:
  gt           - torch.Tensor (1, H, W)  ground truth
  model_kwargs - dict with keys "condition_rls" and "condition_fbp", each a
                 single-channel tensor (1, H, W).  This matches DOLCE's
                 ConditionalModel.forward(x, t, condition_fbp, condition_rls),
                 which internally concatenates [x, cond, cond] -> 3 channels.
                 We only have a SIRT reconstruction, so it is used for both
                 conditioning slots (with use_condtion='rls' only the rls slot
                 is actually consumed by the model).
"""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data.augmentation import TrainingAugmentation


def _minmax_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """DOLCE conditioning normalisation: clip negatives, then min-max to [0,1]."""
    x = np.clip(x, 0.0, np.inf)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + eps)


# Dataset

class DBTSliceDataset(Dataset):
    """
    Dataset of (ground_truth, conditioning) pairs stored as HDF5 files.

    Parameters
    ----------
    file_list : list[str] or str
        Either a Python list of HDF5 file paths, or a path to a .txt file
        containing one HDF5 path per line.
    augment : bool
        Whether to apply online augmentation (flips + zoom). Use True for
        training, False for validation / test.
    image_size : int
        Expected spatial size (used for shape assertion).
    deterministic : bool
        If True, disable any randomness (useful for evaluation).
    """

    def __init__(
        self,
        file_list,
        augment: bool = False,
        image_size: int = 512,
        deterministic: bool = False,
        data_range: str = "-11",
    ):
        if isinstance(file_list, (str, os.PathLike)):
            with open(file_list, "r") as f:
                self.files = [l.strip() for l in f if l.strip()]
        else:
            self.files = list(file_list)

        self.augment      = augment
        self.image_size   = image_size
        self.deterministic = deterministic
        # DOLCE's native image range. "-11" maps GT and conditioning to [-1,1]
        # (the range model512_all.pt was trained in); "01" keeps [0,1].
        self.data_range   = data_range
        self.transform = TrainingAugmentation() if augment else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with h5py.File(path, "r") as hf:
            # Stored as float16 to save disk; upcast to float32 for processing.
            gt   = hf["arr_img512"][:].astype(np.float32)   # (1, H, W)
            rls  = hf["arr_la_rls"][:].astype(np.float32)   # (1, H, W)
            # The fbp conditioning slot is identical to rls and is no longer
            # stored separately; reuse rls. Fall back to a stored arr_la_fbp
            # only if an older file still contains one.
            if "arr_la_fbp" in hf:
                fbp = hf["arr_la_fbp"][:].astype(np.float32)
            else:
                fbp = rls.copy()

        gt  = np.flipud(gt[0])[None].copy()     # vertical flip (DOLCE convention)
        rls = np.flipud(rls[0])[None].copy()
        fbp = np.flipud(fbp[0])[None].copy()

        # Match DOLCE image_datasets.py exactly so the conditioning statistics
        # are in-distribution for model512_all.pt:
        #   GT  : clip to [0, 1]
        #   cond: clip(0, inf) then per-image min-max normalisation to [0, 1]
        gt  = np.clip(gt, 0, 1).astype(np.float32)
        rls = _minmax_norm(rls).astype(np.float32)
        fbp = _minmax_norm(fbp).astype(np.float32)

        gt_t  = torch.from_numpy(gt)    # (1, H, W)
        rls_t = torch.from_numpy(rls)
        fbp_t = torch.from_numpy(fbp)

        if self.transform is not None and not self.deterministic:
            # Apply the IDENTICAL geometric transform to gt, rls and fbp by
            # stacking the two conditioning channels, transforming once, then
            # splitting them back into single-channel tensors.
            cond2 = torch.cat([rls_t, fbp_t], dim=0)   # (2, H, W)
            gt_t, cond2 = self.transform(gt_t, cond2)
            rls_t = cond2[0:1]
            fbp_t = cond2[1:2]

        # Map [0,1] -> [-1,1] AFTER augmentation, so zoom zero-padding stays at
        # the background value (0 -> -1) rather than the midpoint.
        if self.data_range == "-11":
            gt_t  = gt_t * 2 - 1
            rls_t = rls_t * 2 - 1
            fbp_t = fbp_t * 2 - 1

        # DOLCE ConditionalModel expects named single-channel conditions.
        model_kwargs = {"condition_rls": rls_t, "condition_fbp": fbp_t}
        return gt_t, model_kwargs


# DataLoader factories

def build_loader(
    file_list,
    batch_size: int = 4,
    num_workers: int = 4,
    augment: bool = False,
    image_size: int = 512,
    deterministic: bool = False,
    rank: int = 0,
    world_size: int = 1,
    data_range: str = "-11",
) -> DataLoader:
    dataset = DBTSliceDataset(
        file_list,
        augment=augment,
        image_size=image_size,
        deterministic=deterministic,
        data_range=data_range,
    )

    sampler = None
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=(not deterministic)
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and not deterministic),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(not deterministic),
    )
    return loader


def infinite_loader(loader: DataLoader):
    """Yield batches indefinitely (for step-based training loops)."""
    while True:
        yield from loader
