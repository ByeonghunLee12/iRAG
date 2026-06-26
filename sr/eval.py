"""Evaluate an iRAG checkpoint on a validation set.

For each checkpoint, runs DDIM sampling over the validation images and reports
PSNR / SSIM / LPIPS / CLIP-IQA / MUSIQ. The intermediate image is produced
on the fly by the model's TTSR restoration module from the bicubic LQ, the
reference, and the native LR. An optional color fix (AdaIN or wavelet, using the
bicubic LQ as the color reference) can be applied to the SR output before
scoring; `--colorfix all` scores none/adain/wavelet together in one pass.

The validation root must contain `gt/`, `sr_bicubic/`, `lr/` and `ref/`
subfolders sharing the same basenames.

Usage:
  python eval.py \\
    --config  configs/irag.yaml \\
    --ckpt    path/to/iRAG.ckpt \\        # a single ckpt or a directory of ckpts
    --val-dir PATH/TO/valset \\
    --ddim-steps 50 --n-val 500 --colorfix all
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, space_timesteps
from scripts.wavelet_color_fix import (
    wavelet_reconstruction, adaptive_instance_normalization)


def apply_colorfix(sr01, ref01, kind):
    """Color-correct the SR output (content) toward the LQ reference (style).

    Tensors are 4D in [0, 1]; the output is clamped back to [0, 1]. `none`
    returns the SR unchanged.
    """
    if kind == 'none':
        return sr01
    if kind == 'adain':
        out = adaptive_instance_normalization(sr01, ref01)
    elif kind == 'wavelet':
        out = wavelet_reconstruction(sr01, ref01)
    else:
        raise ValueError(f'unknown colorfix type: {kind}')
    return out.clamp(0.0, 1.0)


def build_model(cfg, ckpt_path):
    cfg.model.params.ckpt_path = ckpt_path
    model = instantiate_from_config(cfg.model)
    model.configs = cfg
    model = model.cuda().eval()
    if getattr(model, 'restoration_module', None) is None:
        raise RuntimeError("model.restoration_module is None; this script needs a "
                           "combined checkpoint (use_restoration: true).")
    model.register_schedule(given_betas=None, beta_schedule="linear",
                            timesteps=1000, linear_start=0.00085,
                            linear_end=0.0120, cosine_s=8e-3)
    model.num_timesteps = 1000
    # register_schedule adds new buffers (alphas_cumprod, ...) on CPU; move the
    # whole model back to GPU so they match the input tensors.
    model = model.cuda()
    return model


@torch.no_grad()
def evaluate(model, loader, lpips_model, clipiqa, musiq,
             ddim_steps, n_val, colorfix_types):
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
    ts_arr = np.array(sorted(set(space_timesteps(1000, [ddim_steps]))))
    text_cond = model.cond_stage_model([''])

    keys = ['psnr', 'ssim', 'lpips', 'clipiqa', 'musiq']
    acc = {cf: {k: [] for k in keys} for cf in colorfix_types}
    for i, batch in enumerate(loader):
        if i >= n_val:
            break
        gt = batch['gt'].cuda().float() * 2 - 1
        lq = batch['lq'].cuda().float() * 2 - 1     # bicubic LQ
        lr = batch['lr'].cuda().float() * 2 - 1     # native LR
        ref = batch['ref'].cuda().float() * 2 - 1   # HR reference

        # TTSR produces the intermediate; encode both LR and intermediate.
        i_inter = torch.clamp(model.restoration_module(lq, ref, lr_pixel=lr),
                              -1.0, 1.0)
        z_lr = model.get_first_stage_encoding(model.encode_first_stage(lq))
        z_inter = model.get_first_stage_encoding(model.encode_first_stage(i_inter))

        # Deterministic noise per image -> identical comparison across ckpts.
        torch.manual_seed(1000 + i)
        noise = torch.randn_like(z_lr)
        t = torch.full((1,), 999, device='cuda', dtype=torch.long)
        x_T = model.q_sample(z_lr, t, noise=noise)

        samples, _ = sampler.ddim_sampling_sr_t(
            cond=text_cond, struct_cond=z_lr, struct_cond_ref=z_inter,
            shape=z_lr.shape, x_T=x_T, timesteps=ts_arr,
        )
        sr = torch.clamp(model.decode_first_stage(samples), -1.0, 1.0)
        sr01 = (sr + 1) / 2
        gt01 = (gt + 1) / 2
        lq01 = (lq + 1) / 2          # bicubic LQ = color reference
        gt_np = gt01[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)

        for cf in colorfix_types:
            f01 = apply_colorfix(sr01, lq01, cf)
            f_np = f01[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            acc[cf]['psnr'].append(psnr_fn(gt_np, f_np, data_range=1.0))
            acc[cf]['ssim'].append(ssim_fn(gt_np, f_np, data_range=1.0,
                                           channel_axis=-1))
            acc[cf]['lpips'].append(float(lpips_model(gt, f01 * 2 - 1).item()))
            acc[cf]['clipiqa'].append(float(clipiqa(f01).item()))
            acc[cf]['musiq'].append(float(musiq(f01).item()))

    return {cf: {k: float(np.mean(acc[cf][k])) for k in keys}
            for cf in colorfix_types}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True,
                        help='a single .ckpt file or a directory of .ckpt files')
    parser.add_argument('--val-dir', required=True,
                        help='validation root with gt/ sr_bicubic/ lr/ ref/ subfolders')
    parser.add_argument('--ddim-steps', type=int, default=50)
    parser.add_argument('--n-val', type=int, default=500)
    parser.add_argument('--colorfix', default='none',
                        choices=['none', 'adain', 'wavelet', 'all'])
    parser.add_argument('--out', default=None, help='optional JSON to save results')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    colorfix_types = (['none', 'adain', 'wavelet'] if args.colorfix == 'all'
                      else [args.colorfix])

    cfg = OmegaConf.load(args.config)

    import lpips
    import pyiqa
    from ldm.data.irag_datasets import IRAGPairedDataset
    lpips_model = lpips.LPIPS(net='alex').cuda().eval()
    clipiqa = pyiqa.create_metric('clipiqa').cuda().eval()
    musiq = pyiqa.create_metric('musiq').cuda().eval()

    ds = IRAGPairedDataset(args.val_dir, mode='online', gt_size=None, scale=4,
                           phase='val')
    loader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=False)
    print(f'val dataset: {len(ds)} images')

    if os.path.isdir(args.ckpt):
        ckpts = sorted(glob.glob(os.path.join(args.ckpt, '*.ckpt')))
    else:
        ckpts = [args.ckpt]

    results = {}
    for ckpt in ckpts:
        name = os.path.basename(ckpt)
        model = build_model(cfg, ckpt)
        metrics = evaluate(model, loader, lpips_model, clipiqa, musiq,
                           args.ddim_steps, args.n_val, colorfix_types)
        results[name] = metrics
        for cf in colorfix_types:
            m = metrics[cf]
            print(f'{name} [{cf:7s}] PSNR={m["psnr"]:.3f} SSIM={m["ssim"]:.4f} '
                  f'LPIPS={m["lpips"]:.4f} CLIP-IQA={m["clipiqa"]:.4f} '
                  f'MUSIQ={m["musiq"]:.2f}')
        del model
        torch.cuda.empty_cache()

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'saved results to {args.out}')


if __name__ == '__main__':
    main()
