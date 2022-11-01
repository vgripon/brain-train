import torch
import torch.nn as nn

class SupConLoss(nn.Module): # inspired by : https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def __init__(self, temperature=0.07, base_temperature=0.07, device="cpu"):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, proj1, proj2, labels=None):

        features = torch.cat([proj1.unsqueeze(1), proj2.unsqueeze(1)], dim=1)
        batch_size = features.shape[0]

        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * contrast_count).view(-1, 1).to(self.device), 0)
        # or simply : logits_mask = torch.ones_like(mask) - torch.eye(50)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()

        return loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss(nn.Module):
    def __init__(self, projector_out=128, lambd=0.0051):
        super().__init__()

        self.bn = nn.BatchNorm1d(projector_out, affine=False)
        self.lambd = lambd

    def forward(self, z1, z2):

        c = self.bn(z1).T @ self.bn(z2) #normalization outside the loss function
        c = z1.T @ z2
        c.div_(z1.shape[0])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss