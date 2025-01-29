from pathlib import Path
import scipy.io
import numpy as np
import typer
from torch.utils.data import Dataset


standard_path = Path("data/raw")

class NMRDataset(Dataset):
    """NMR dataset"""
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.X)  # Adjust this as needed
    def __shape__(self) -> tuple:
        """Return the shape of the dataset."""
        return self.X.shape

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.X[index]  # Adjust this as needed
    
    def downscale(self, factor: int) -> np.ndarray:
        """Return a downsampled version of the dataset."""
        self.X = self.X[:, ::factor]


class UrineDataset(NMRDataset):
    def __init__(self, raw_data_path: Path = standard_path) -> None:
        self.data_path = raw_data_path
        self.X, self.Y = self.load_urine_data()

    def load_urine_data(self):
        mat = scipy.io.loadmat(self.data_path / 'nmrdata.mat')
        mat = mat.get('nmrdata')
        X_URINE = mat[0][0][0]
        labels_URINE = mat[0][0][1]
        return X_URINE, labels_URINE

class OilDataset(NMRDataset):
    def __init__(self, raw_data_path: Path = standard_path) -> None:
        self.data_path = raw_data_path
        self.X, self.Y = self.load_oil_data()


    def load_oil_data(self):
        mat = scipy.io.loadmat(self.data_path / 'nmrdata_Oil_group3.mat')
        mat = mat.get('nmrdata_Oil_group3')
        X_OIL = mat[0][0][0]
        OIL_labels = mat[0][0][1]
        return X_OIL, OIL_labels

class WineDataset(NMRDataset):
    def __init__(self, raw_data_path: Path = standard_path) -> None:
        self.data_path = raw_data_path
        self.X, self.WINE_PARAMETERS, self.PPM_WINE, self.Y = self.load_wine_data()

    def load_wine_data(self):
        mat = scipy.io.loadmat(self.data_path / 'NMR_40wines.mat')
        X_WINE = mat.get('X')
        WINE_PARAMETERS = mat.get('Y')
        PPM_WINE = mat.get('ppm')[0]
        labels = mat.get('Label')
        WINE_labels = [x[0] for x in labels[0]]
        return X_WINE, WINE_PARAMETERS, PPM_WINE, WINE_labels

class AlkoDataset(NMRDataset):
    def __init__(self, raw_data_path: Path = standard_path) -> None:
        self.data_path = raw_data_path
        self.X, self.Y, self.ALKO_labels, self.axis = self.load_alko_data()

    def load_alko_data(self):
        mat = scipy.io.loadmat(self.data_path / 'NMR_mix_DoE.mat')
        X_ALKO = mat.get('xData')
        Y_ALKO = mat.get('yData')
        ALKO_labels = mat.get('yLabels')
        axis = mat.get("Axis")
        return X_ALKO, Y_ALKO, ALKO_labels, axis

class ArtificialDataset(NMRDataset):
    def __init__(self, raw_data_path: Path = standard_path) -> None:
        self.data_path = raw_data_path
        self.X, self.X_ART, self.H, self.W, self.TAU = self.generate_artificial_data()
    

    def generate_artificial_data(self):
        N, M, d = 30, 20000, 3
        t = np.arange(0, M, 1)
        np.random.seed(42)
        W = np.random.dirichlet(np.ones(d), N)
        shift = 1
        tau = np.random.randint(-shift, shift, size=(N, d))
        tau = np.zeros((N, d))
        tau = np.random.randint(-1000, 1000, size=(N, d))
        H = np.zeros((d, M))
        from psim.helpers.generators import multiplet
        H[0] = multiplet(t, 3, 6000, 110 * 2, 900) + multiplet(t, 1, 12000, 160 * 2, 0)
        H[1] = multiplet(t, 2, 2000, 150 * 2, 800) + multiplet(t, 2, 14000, 240 * 2, 1200)
        H[2] = multiplet(t, 3, 18000, 300 * 2, 1300) + multiplet(t, 4, 12000, 120 * 2, 800)
        X_ART = self.shift_dataset(W, H, tau)
        NOISE_ART = np.random.normal(0, 5e-6, X_ART.shape)
        X_ART_NOISY = X_ART + NOISE_ART
        return X_ART_NOISY, X_ART, H, W, tau

    def shift_dataset(self, W, H, tau):
        Nf = H.shape[1] // 2 + 1
        Hf = np.fft.fft(H, axis=1)
        Hf = Hf[:, :Nf]
        Hf_reverse = np.fliplr(Hf[:, 1:Nf - 1])
        Hft = np.concatenate((Hf, np.conj(Hf_reverse)), axis=1)
        f = np.arange(0, H.shape[1]) / H.shape[1]
        omega = np.exp(-1j * 2 * np.pi * np.einsum('Nd,M->NdM', tau, f))
        Wf = np.einsum('Nd,NdM->NdM', W, omega)
        Vf = np.einsum('NdM,dM->NM', Wf, Hft)
        V = np.fft.ifft(Vf)
        return V.real

if __name__ == "__main__":
    #typer.run(preprocess)
    print(ArtificialDataset()[0])
