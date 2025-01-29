import pytest
from psim.data import ArtificialDataset
from psim.models.shiftNMF import ShiftNMF
import torch
from torchrl.modules.utils import inv_softplus

    

def test_shiftNMF_fit():
    """Test the ShiftNMF fit method and reconstruction."""
    dataset = ArtificialDataset()
    model = ShiftNMF(dataset.X,3, lr=0.1, std=True)
    assert model is not None
    model.H = torch.nn.Parameter(inv_softplus(torch.tensor(dataset.H, dtype=torch.double)))
    model.W = torch.nn.Parameter(torch.tensor(dataset.W, dtype=torch.double))
    model.tau = torch.tensor(dataset.TAU)
    output = model.forward()
    #rescale to match std
    output = output /torch.std(torch.tensor(dataset.X))
    #test reconstruction
    assert model.lossfn(output).detach().cpu().numpy() < 0.001
    model.fit()
    output = model.forward()
    #rescale to match std
    output = output /torch.std(torch.tensor(dataset.X))
    #test reconstruction
    assert model.lossfn(output).detach().cpu().numpy() < 0.001

