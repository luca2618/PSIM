import pytest
from torch.utils.data import Dataset
from psim.data import UrineDataset, OilDataset, WineDataset, AlkoDataset, ArtificialDataset, standard_path

def test_urine_dataset():
    """Test the UrineDataset class."""
    dataset = UrineDataset()
    assert isinstance(dataset, Dataset)
    assert len(dataset) > 0
    assert dataset[0] is not None

def test_oil_dataset():
    """Test the OilDataset class."""
    dataset = OilDataset()
    assert isinstance(dataset, Dataset)
    assert len(dataset) > 0
    assert dataset[0] is not None

def test_wine_dataset():
    """Test the WineDataset class."""
    dataset = WineDataset()
    assert isinstance(dataset, Dataset)
    assert len(dataset) > 0
    assert dataset[0] is not None

def test_alko_dataset():
    """Test the AlkoDataset class."""
    dataset = AlkoDataset()
    assert isinstance(dataset, Dataset)
    assert len(dataset) > 0
    assert dataset[0] is not None

def test_artificial_dataset():
    """Test the ArtificialDataset class."""
    dataset = ArtificialDataset(standard_path)
    assert isinstance(dataset, Dataset)
    assert len(dataset) > 0
    assert dataset[0] is not None

if __name__ == "__main__":
    pytest.main()
