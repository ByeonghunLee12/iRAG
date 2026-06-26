import torchvision

import torch
import argparse
import torch.nn as nn

from model.base_model import Base_Model
from loss.NtXent import NtXentLoss
from model.hash.hash_fn import cont_layer, hash_layer

class CIBHash(Base_Model):
    """
    CIBHash model that uses a truncated VGG16 as a feature extractor,
    an encoder network, and contrastive loss with KL divergence regularization.
    """
    def __init__(self, hparams):
        super().__init__(hparams=hparams)
        self.hash_layer = hash_layer
        self.cont_layer = cont_layer

    def define_parameters(self):
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        for param in self.vgg.parameters():
            param.requires_grad = False

        vgg_feature_out = 4096 
        feature_hidden  = 1024
        self.encoder = nn.Sequential(
            nn.Linear(vgg_feature_out, feature_hidden),
            nn.ReLU(),
            nn.Linear(feature_hidden, self.hparams.encode_length)
        )

        # Define the contrastive loss (NT-Xent Loss).
        self.criterion = NtXentLoss(batch_size=self.hparams.batch_size,
                                    temperature=self.hparams.temperature)

    # VGG Feature Extraction 
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        return x

    def forward(self, imgi: torch.Tensor, imgj: torch.Tensor) -> dict:
        # Remove the explicit 'device' parameter; use the device of the inputs.
        def _process_image(img: torch.Tensor) -> torch.Tensor:
            feat = self.extract_features(img)
            prob = torch.sigmoid(self.encoder(feat))
            return prob

        # Process the first image.
        prob_i = _process_image(imgi)
        z_i    = self.hash_layer(prob_i - 0.5)

        # Process the second image.
        prob_j = _process_image(imgj)
        z_j    = self.hash_layer(prob_j - 0.5)

        # Compute symmetric KL divergence between the two probability distributions.
        kl_loss = (self.compute_kl(prob_i, prob_j) + self.compute_kl(prob_j, prob_i)) / 2

        # Compute contrastive (NT-Xent) loss.
        contra_loss = self.criterion(z_i, z_j)

        # Total loss is the sum of contrastive loss and weighted KL divergence.
        loss = contra_loss + self.hparams.weight * kl_loss

        return {'loss': loss, 'contra_loss': contra_loss, 'kl_loss': kl_loss}

    def encode_discrete(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image to its discrete hash code.
        """
        feat = self.extract_features(x)
        prob = torch.sigmoid(self.encoder(feat))
        z    = self.hash_layer(prob - 0.5)
        return z

    def encode_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image to its continuous hash code.
        """
        feat = self.extract_features(x)
        prob = torch.sigmoid(self.encoder(feat))
        z    = self.cont_layer(prob - 0.5)
        return z 

    def compute_kl(self, prob: torch.Tensor, prob_v: torch.Tensor) -> torch.Tensor:
        """
        Compute the KL divergence between two probability distributions.
        """
        prob_v = prob_v.detach()
        kl_div = prob * (torch.log(prob + 1e-8) - torch.log(prob_v + 1e-8)) \
                 + (1 - prob) * (torch.log(1 - prob + 1e-8) - torch.log(1 - prob_v + 1e-8))
        kl_div = torch.mean(torch.sum(kl_div, dim=1))
        return kl_div

    def configure_optimizers(self):
        """
        Configure the optimizer (only optimizing the encoder parameters).
        """
        return torch.optim.Adam(self.encoder.parameters(), lr=self.hparams.lr)

    def get_hparams_grid(self) -> dict:
        """
        Get a grid of hyperparameters for tuning.
        """
        grid = Base_Model.get_general_hparams_grid()
        grid.update({
            'temperature': [0.2, 0.3, 0.4],
            'weight': [0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001]
        })
        return grid

    @staticmethod
    def get_model_specific_argparser() -> argparse.ArgumentParser:
        """
        Get an argument parser with model-specific hyperparameters.
        """
        parser = Base_Model.get_general_argparser()
        parser.add_argument("-t", "--temperature", default=0.3, type=float,
                            help="Temperature (default: %(default)f)")
        parser.add_argument("-w", "--weight", default=0.001, type=float,
                            help="Weight of I(x,z) (default: %(default)f)")
        return parser
