"""iRAG wrapper around TTSR (researchmm/TTSR, MIT).

The iRAG `LatentDiffusionSRiRAG` model calls
    restoration_module(lq_pixel, ref_pixel) -> i_inter_pixel
with both inputs at HR resolution (the base SR model bicubically upsamples the
degraded LR back to HR size, see ddpm.py:get_input `resize_lq`). TTSR's
own forward expects (lr, lrsr, ref, refsr) with lr at native LR size
(=HR/4) and the others at HR size, so this wrapper handles the resizing.

All image tensors are in [-1, 1] throughout, matching the base model's
self.lq / self.gt normalization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ttsr_core import TTSRCore, Vgg19


class TTSR(nn.Module):
    """iRAG-facing restoration module.

    Args:
        scale: integer downsampling factor used to derive TTSR's `lr` from
            the HR-sized `lq_pixel`. Default 4 (TTSR's native ×4 SR setting).
        num_res_blocks, n_feats, res_scale: forwarded to TTSRCore.
        return_aux: if True, forward returns (i_inter, aux) where aux is a
            dict with S, T_lv1/2/3 — used by the transferal perceptual loss.
            Default False (only the image is returned, matching the simple
            `(lq, ref) -> i_inter` contract LatentDiffusionSRiRAG expects).

    Inputs / output ranges: all in [-1, 1].
    """

    def __init__(
        self,
        scale=4,
        num_res_blocks=(16, 16, 8, 4),
        n_feats=64,
        res_scale=1.0,
        return_aux=False,
    ):
        super().__init__()
        self.scale = int(scale)
        self.return_aux = return_aux
        self.core = TTSRCore(
            num_res_blocks=num_res_blocks,
            n_feats=n_feats,
            res_scale=res_scale,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _bicubic(x, size=None, scale_factor=None):
        # PyTorch warns about align_corners default; pin it for stability.
        return F.interpolate(
            x, size=size, scale_factor=scale_factor,
            mode='bicubic', align_corners=False,
        )

    def _build_refsr(self, ref_pixel):
        """Simulate the LR-style version of `ref` at HR size: ref ↓s ↑s."""
        h, w = ref_pixel.shape[-2:]
        ref_down = self._bicubic(ref_pixel, scale_factor=1.0 / self.scale)
        return self._bicubic(ref_down, size=(h, w))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, lq_pixel, ref_pixel, lr_pixel=None):
        """
        lq_pixel : [B, 3, H, W]   in [-1, 1]; degraded LR bicubically upsampled
                   to HR size (== the base model's self.lq, equivalently sr_bicubic).
                   Used by TTSR's LTE for content matching against the ref.
        ref_pixel: [B, 3, H, W]   in [-1, 1]; HR reference, same H, W as lq.
        lr_pixel : [B, 3, H/s, W/s] in [-1, 1] or None. Native-resolution LR
                   image (HR / scale). This is fed to TTSR's MainNet (SFE) as
                   the actual SR input. When None, it is synthesized as
                   bicubic_down(lq_pixel, 1/scale) -- usable but suboptimal
                   because it round-trips through bicubic. Prefer passing the
                   on-disk LR when available.
        Returns I_inter at the HR size, in [-1, 1].
        """
        if lq_pixel.shape[-2:] != ref_pixel.shape[-2:]:
            raise ValueError(
                f"TTSR wrapper expects lq and ref at same H, W "
                f"(got {tuple(lq_pixel.shape[-2:])} vs {tuple(ref_pixel.shape[-2:])})."
            )

        # TTSR's `lr` is at native LR size; everything else is at HR size.
        if lr_pixel is None:
            lr = self._bicubic(lq_pixel, scale_factor=1.0 / self.scale)
        else:
            lr = lr_pixel
            # Sanity-check that user-provided lr has expected size relative to lq.
            exp_h = lq_pixel.shape[-2] // self.scale
            exp_w = lq_pixel.shape[-1] // self.scale
            if lr.shape[-2:] != (exp_h, exp_w):
                raise ValueError(
                    f"lr_pixel size {tuple(lr.shape[-2:])} does not match "
                    f"lq/scale = ({exp_h}, {exp_w}) with scale={self.scale}."
                )
        lrsr = lq_pixel
        ref = ref_pixel
        refsr = self._build_refsr(ref_pixel)

        sr_out, S, T_lv3, T_lv2, T_lv1 = self.core(lr=lr, lrsr=lrsr, ref=ref, refsr=refsr)

        if self.return_aux:
            return sr_out, {'S': S, 'T_lv3': T_lv3, 'T_lv2': T_lv2, 'T_lv1': T_lv1}
        return sr_out

    # Convenience: expose transferal-perceptual-loss pass-through.
    def lte_features(self, sr):
        """Return (sr_lv1, sr_lv2, sr_lv3) via LTE_copy for transferal
        perceptual loss. `sr` in [-1, 1]."""
        return self.core(sr=sr)


# Re-export for config convenience.
__all__ = ['TTSR', 'Vgg19']
