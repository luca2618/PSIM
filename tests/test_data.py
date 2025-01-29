from torch.utils.data import Dataset

from psim.data import MyDataset, standard_path


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset(standard_path)
    assert isinstance(dataset, Dataset)
