import torch
import torch.nn as nn

#### Contrastive Loss ####
class NtXentLoss(nn.Module):
    """
    Normalized temperature-scaled cross entropy (NT-Xent) loss.
    """
    def __init__(self, batch_size: int, temperature: float):
        super(NtXentLoss, self).__init__()
        self.temperature = temperature
        self.similarity_function = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.batch_size = batch_size

    def mask_correlated_samples(self) -> torch.Tensor:
        """
        Create a mask that removes the positive pairs from the negatives.
        """
        N = 2 * self.batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask.fill_diagonal_(False)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = False
            mask[self.batch_size + i, i] = False
        return mask

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute the NT-Xent loss.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_function(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)

        mask = self.mask_correlated_samples().to(z.device)
        negative_samples = sim[mask].view(N, -1)

        labels = torch.zeros(N, device=z.device, dtype=torch.long)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
