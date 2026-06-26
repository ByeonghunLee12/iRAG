# Ported from researchmm/TTSR/model/TTSR.py (MIT License).
# Renamed to TTSRCore to avoid colliding with our top-level iRAG wrapper TTSR.
import torch
import torch.nn as nn

from .lte import LTE
from .mainnet import MainNet
from .search_transfer import SearchTransfer


class TTSRCore(nn.Module):
    """Original TTSR architecture (×4 SR).

    forward(lr, lrsr, ref, refsr, sr=None) returns:
      - (sr, S, T_lv3, T_lv2, T_lv1) for normal SR pass
      - (sr_lv1, sr_lv2, sr_lv3) when `sr` is given (transferal perceptual loss)

    Conventions (all tensors in [-1, 1] for image inputs; LTE internally
    shifts to [0, 1] via +1./2.):
      lr    : LR image at native LR size (= HR / 4)
      lrsr  : LR bicubically upsampled to HR size
      ref   : HR reference image (same H, W as HR / lrsr)
      refsr : ref bicubically down-and-up sampled at HR size
              (simulates LR-style content)
      sr    : optional, an SR prediction fed back through LTE_copy
              to provide transferal perceptual loss targets.
    """

    def __init__(self, num_res_blocks=(16, 16, 8, 4), n_feats=64, res_scale=1.0):
        super(TTSRCore, self).__init__()
        # Allow either an iterable or a "+"-separated string for back-compat
        # with the original `args.num_res_blocks = "16+16+8+4"` convention.
        if isinstance(num_res_blocks, str):
            num_res_blocks = list(map(int, num_res_blocks.split('+')))
        self.num_res_blocks = list(num_res_blocks)
        self.n_feats = n_feats
        self.res_scale = res_scale

        self.MainNet = MainNet(num_res_blocks=self.num_res_blocks,
                               n_feats=n_feats, res_scale=res_scale)
        self.LTE = LTE(requires_grad=True)
        self.LTE_copy = LTE(requires_grad=False)  ### used in transferal perceptual loss
        self.SearchTransfer = SearchTransfer()

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None):
        if sr is not None:
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        _, _, lrsr_lv3 = self.LTE((lrsr.detach() + 1.) / 2.)
        _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)

        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)

        S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3)

        sr_out = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)

        return sr_out, S, T_lv3, T_lv2, T_lv1
