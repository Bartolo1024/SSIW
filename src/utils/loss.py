import torch
from torch import nn


def nxn_cos_sim(A, B, dim=1, eps=1e-8):
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)


class HDLoss(nn.Module):
    def __init__(self, embeddigns: torch.Tensor):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.embeddings = embeddigns

    def __call__(self, output: torch.Tensor, target: torch.Tensor, one_hot: torch.Tensor):
        bs, channels, h, w = output.shape
        output = output.permute(0, 2, 3, 1).view(-1, channels)
        cos_sim = nxn_cos_sim(output, self.embeddings)
        one_hot = one_hot.permute(0, 2, 3, 1).reshape(-1, one_hot.shape[1])
        loss = self.log_softmax(cos_sim)
        loss = loss * one_hot
        loss = -loss.mean()
        return loss
