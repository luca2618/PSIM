import torch
import torch.nn.functional as F

class frobeniusLoss(torch.nn.Module):
    def __init__(self, x: torch.tensor):
        super().__init__()
        self.X = x

        self.denominator = torch.linalg.matrix_norm(self.X, ord='fro')**2
    
    def forward(self, input):
        numerator = torch.linalg.matrix_norm(self.X - input, ord='fro')**2
        return numerator/self.denominator


class ShiftNMFLoss(torch.nn.Module):
    def __init__(self, x: torch.tensor):
        super().__init__()
        _, self.M = x.shape
        self.X = x
        # self.norm = torch.linalg.matrix_norm(self.X, ord="fro")**2

    def forward(self, input):
        return 1/(2*self.M) * torch.linalg.matrix_norm(self.X - input, ord='fro')**2 # /self.norm


class VolLoss(torch.nn.Module):
    def __init__(self, x: torch.tensor, alpha):
        super().__init__()
        self.X = x
        self.alpha = alpha
        self.norm = torch.linalg.matrix_norm(self.X, ord="fro")**2

    def forward(self, x, h):
        reg = self.alpha*torch.linalg.det(torch.matmul(h, h.T))
        loss = torch.linalg.matrix_norm(self.X - x, ord='fro')**2 / self.norm
        return reg + loss

# Sparseness measure of the H-matrix

class MVR_ShiftNMF_Loss(torch.nn.Module):
    def __init__(self, x: torch.tensor, lamb=0.01):
        super().__init__()
        self.N, self.M = x.shape
        self.Xf = x
        self.lamb = lamb
        self.eps = 1e-9

    def forward(self, inp, H): # Loss function must take the reconstruction and H.
        loss = 1 / (2 * self.M) * torch.linalg.matrix_norm(self.Xf - inp, ord='fro')**2
        vol_H = torch.log(torch.det(torch.matmul(H, H.T) + self.eps))
        reg = self.lamb * vol_H
        return loss + reg