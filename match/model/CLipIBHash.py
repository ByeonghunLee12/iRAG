import clip 

import torch
import torch.nn as nn

from model.CIBHash import CIBHash
from loss.NtXent import NtXentLoss

class CLipIBHash(CIBHash):
    """
    CLipIBHash model that uses a CLIP-ViT-L/14 as a feature extractor,
    an encoder network, and contrastive loss with KL divergence regularization.
    """
    def define_parameters(self):
        _clip_model = 'ViT-L/14'  # or 'ViT-B/32'
        self.clip_model, _ = clip.load(_clip_model,
                                       device=self.hparams.device,
                                       jit=False)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        clip_feature_out = self.clip_model.visual.output_dim
        feature_hidden   = 1024
        self.encoder = nn.Sequential(
            nn.Linear(clip_feature_out, feature_hidden),
            nn.ReLU(),
            nn.Linear(feature_hidden, self.hparams.encode_length)
        )

        # Define the contrastive loss (NT-Xent Loss).
        self.criterion = NtXentLoss(batch_size=self.hparams.batch_size,
                                    temperature=self.hparams.temperature)

    # CLIP Feature Extraction 
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure that x is preprocessed according to CLIP's requirements.
        with torch.no_grad():
            features = self.clip_model.encode_image(x.half()).to(torch.float32)
        return features
