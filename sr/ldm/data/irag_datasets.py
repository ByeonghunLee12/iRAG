"""iRAG paired image datasets.

Disk layout (paired, same basename across folders):

    root_path/
      ├── gt/          <name>.png   HR ground truth
      ├── lr/          <name>.png   native LR (HR/scale)   -- online only
      ├── sr_bicubic/  <name>.png   bicubic upsample of LR to HR size
      └── ref/         <name>.png   HR reference            -- online only
      └── inter/       <name>.png   precomputed I_inter     -- precomp only

The diffusion-side LR seen by the diffusion model (== self.lq, the input to the
condition encoder ε_φ) is `sr_bicubic`. The native `lr` is only used by
TTSR's MainNet as the actual SR backbone input.

Output (all tensors): RGB, CHW, float32, range [0, 1].
LatentDiffusionSRiRAG.get_input normalizes to [-1, 1].
"""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')


def _list_basenames(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dataset folder does not exist: {folder}")
    names = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(_IMG_EXTS)
    )
    if not names:
        raise RuntimeError(f"No images found under {folder}")
    return names


def _read_img(path):
    """Load image with cv2 -> RGB float32 [0, 1] HWC."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _to_tensor_chw(img_hwc):
    """HWC numpy [0,1] -> CHW torch tensor [0,1]."""
    return torch.from_numpy(np.ascontiguousarray(img_hwc.transpose(2, 0, 1)))


class IRAGPairedDataset(Dataset):
    """Paired dataset for iRAG training/eval.

    Modes:
      - 'online'  : returns {'gt', 'lq', 'lr', 'ref'} (no Real-ESRGAN kernels)
      - 'precomp': returns {'gt', 'lq', 'inter'}

    Args:
        root_path: directory containing 'gt', 'sr_bicubic', plus mode-specific
            subfolders ('lr','ref' for online; 'inter' for precomp).
        mode: 'online' or 'precomp'.
        gt_size: HR crop size. The LR crop is gt_size // scale. If None,
            uses the full image (caller must ensure batchable sizes).
        scale: integer SR scale (default 4).
        use_hflip: random horizontal flip during training.
        use_rot: random 90-degree rotation during training (also vertical flip).
        phase: 'train' enables augmentations; otherwise deterministic.
        file_list: optional list of basenames to subset. If None, all files
            in `root_path/gt/` are used.
    """

    def __init__(
        self,
        opt=None,
        mode='online',
        gt_size=512,
        scale=4,
        use_hflip=True,
        use_rot=False,
        phase='train',
        file_list=None,
        root_path=None,
        inter_dir_name='inter',
    ):
        super().__init__()
        # Two calling conventions are supported:
        #  (a) diffusion-model/basicsr: a single `opt` mapping (this is how train.py's
        #      DataModuleFromConfig -> instantiate_from_config_sr constructs
        #      datasets, passing config.params positionally).
        #  (b) direct kwargs: IRAGPairedDataset(root_path=..., mode=..., ...).
        if opt is not None and not isinstance(opt, str):
            # opt is a mapping (dict / OmegaConf DictConfig). Pull fields from it,
            # letting any explicit kwargs act as defaults only.
            o = dict(opt)
            root_path = o.get('root_path', root_path)
            mode = o.get('mode', mode)
            gt_size = o.get('gt_size', gt_size)
            scale = o.get('scale', scale)
            use_hflip = o.get('use_hflip', use_hflip)
            use_rot = o.get('use_rot', use_rot)
            phase = o.get('phase', phase)
            file_list = o.get('file_list', file_list)
            inter_dir_name = o.get('inter_dir_name', inter_dir_name)
        elif isinstance(opt, str) and root_path is None:
            # opt given positionally as the root path.
            root_path = opt
        if root_path is None:
            raise ValueError("IRAGPairedDataset requires `root_path` (either in "
                             "the opt mapping or as a keyword argument).")
        if mode not in ('online', 'precomp'):
            raise ValueError(f"mode must be 'online' or 'precomp', got {mode!r}")
        self.root_path = root_path
        self.mode = mode
        self.gt_size = gt_size
        self.scale = int(scale)
        self.use_hflip = bool(use_hflip)
        self.use_rot = bool(use_rot)
        self.phase = phase

        self.gt_dir = os.path.join(root_path, 'gt')
        self.lq_dir = os.path.join(root_path, 'sr_bicubic')
        if mode == 'online':
            self.lr_dir = os.path.join(root_path, 'lr')
            self.ref_dir = os.path.join(root_path, 'ref')
            self.inter_dir = None
        else:
            self.lr_dir = None
            self.ref_dir = None
            self.inter_dir = os.path.join(root_path, inter_dir_name)

        self.basenames = list(file_list) if file_list is not None else _list_basenames(self.gt_dir)

    def __len__(self):
        return len(self.basenames)

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------
    def _paired_crop(self, hr_imgs, lr_img):
        """Random crop. hr_imgs: list of HR-sized HWC arrays; lr_img: LR-sized
        HWC array (matched by self.scale). LR crop is at the corresponding
        LR-space position. Returns (cropped_hr_list, cropped_lr or None)."""
        if self.gt_size is None:
            return hr_imgs, lr_img

        h_hr, w_hr = hr_imgs[0].shape[:2]
        gt = self.gt_size
        if h_hr < gt or w_hr < gt:
            raise RuntimeError(
                f"Image smaller than gt_size: {h_hr}x{w_hr} < {gt}. "
                f"basenames around this index need padding or resizing."
            )

        if self.phase == 'train':
            top = random.randint(0, h_hr - gt)
            left = random.randint(0, w_hr - gt)
        else:
            top = (h_hr - gt) // 2
            left = (w_hr - gt) // 2

        cropped_hr = [im[top:top + gt, left:left + gt, :] for im in hr_imgs]

        cropped_lr = None
        if lr_img is not None:
            lr_top = top // self.scale
            lr_left = left // self.scale
            lr_size = gt // self.scale
            cropped_lr = lr_img[lr_top:lr_top + lr_size, lr_left:lr_left + lr_size, :]

        return cropped_hr, cropped_lr

    def _augment(self, hr_imgs, lr_img):
        if self.phase != 'train':
            return hr_imgs, lr_img

        # horizontal flip
        if self.use_hflip and random.random() < 0.5:
            hr_imgs = [im[:, ::-1, :].copy() for im in hr_imgs]
            if lr_img is not None:
                lr_img = lr_img[:, ::-1, :].copy()

        # 90-deg rotation (vertical flip + transpose; full 8-way symmetry)
        if self.use_rot:
            if random.random() < 0.5:  # vertical flip
                hr_imgs = [im[::-1, :, :].copy() for im in hr_imgs]
                if lr_img is not None:
                    lr_img = lr_img[::-1, :, :].copy()
            if random.random() < 0.5:  # transpose
                hr_imgs = [im.transpose(1, 0, 2).copy() for im in hr_imgs]
                if lr_img is not None:
                    lr_img = lr_img.transpose(1, 0, 2).copy()

        return hr_imgs, lr_img

    # ------------------------------------------------------------------
    # Item access
    # ------------------------------------------------------------------
    def _load_online(self, name):
        gt = _read_img(os.path.join(self.gt_dir, name))
        lq = _read_img(os.path.join(self.lq_dir, name))
        ref = _read_img(os.path.join(self.ref_dir, name))
        lr = _read_img(os.path.join(self.lr_dir, name))

        # Sanity-check shapes.
        if gt.shape != lq.shape:
            raise RuntimeError(
                f"gt/lq shape mismatch for {name}: {gt.shape} vs {lq.shape}"
            )
        if gt.shape != ref.shape:
            raise RuntimeError(
                f"gt/ref shape mismatch for {name}: {gt.shape} vs {ref.shape}"
            )
        exp_lr = (gt.shape[0] // self.scale, gt.shape[1] // self.scale, 3)
        if lr.shape != exp_lr:
            raise RuntimeError(
                f"lr shape mismatch for {name}: got {lr.shape}, expected {exp_lr} (scale={self.scale})"
            )

        (gt, lq, ref), lr = self._paired_crop([gt, lq, ref], lr)
        (gt, lq, ref), lr = self._augment([gt, lq, ref], lr)

        return {
            'gt': _to_tensor_chw(gt),
            'lq': _to_tensor_chw(lq),
            'ref': _to_tensor_chw(ref),
            'lr': _to_tensor_chw(lr),
            'name': name,
        }

    def _load_precomp(self, name):
        gt = _read_img(os.path.join(self.gt_dir, name))
        lq = _read_img(os.path.join(self.lq_dir, name))
        inter = _read_img(os.path.join(self.inter_dir, name))

        if gt.shape != lq.shape:
            raise RuntimeError(
                f"gt/lq shape mismatch for {name}: {gt.shape} vs {lq.shape}"
            )
        if gt.shape != inter.shape:
            raise RuntimeError(
                f"gt/inter shape mismatch for {name}: {gt.shape} vs {inter.shape}"
            )

        (gt, lq, inter), _ = self._paired_crop([gt, lq, inter], None)
        (gt, lq, inter), _ = self._augment([gt, lq, inter], None)

        return {
            'gt': _to_tensor_chw(gt),
            'lq': _to_tensor_chw(lq),
            'inter': _to_tensor_chw(inter),
            'name': name,
        }

    def __getitem__(self, index):
        name = self.basenames[index]
        if self.mode == 'online':
            return self._load_online(name)
        return self._load_precomp(name)
