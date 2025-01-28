from pathlib import Path
import scipy.io
import numpy as np
import typer
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path
        self.X_URINE, self.labels_URINE = self.load_urine_data()
        self.X_OIL, self.OIL_labels = self.load_oil_data()
        self.X_WINE, self.WINE_PARAMETERS, self.PPM_WINE, self.WINE_labels = self.load_wine_data()
        self.X_ALKO, self.Y_ALKO, self.ALKO_labels, self.axis = self.load_alko_data()
        self.X_ART_NOISY, self.H_ART, self.W_ART, self.TAU_ART = self.generate_artificial_data()

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.X_URINE)  # Adjust this as needed

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.X_URINE[index]  # Adjust this as needed

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        # Implement any preprocessing steps here
        pass

    def load_urine_data(self):
        mat = scipy.io.loadmat(self.data_path / 'nmrdata.mat')
        mat = mat.get('nmrdata')
        X_URINE = mat[0][0][0]
        labels_URINE = mat[0][0][1]
        return X_URINE, labels_URINE

    def load_oil_data(self):
        mat = scipy.io.loadmat(self.data_path / 'nmrdata_Oil_group3.mat')
        mat = mat.get('nmrdata_Oil_group3')
        X_OIL = mat[0][0][0]
        OIL_labels = mat[0][0][1]
        return X_OIL, OIL_labels

    def load_wine_data(self):
        mat = scipy.io.loadmat(self.data_path / 'NMR_40wines.mat')
        X_WINE = mat.get('X')
        WINE_PARAMETERS = mat.get('Y')
        PPM_WINE = mat.get('ppm')[0]
        labels = mat.get('Label')
        WINE_labels = [x[0] for x in labels[0]]
        return X_WINE, WINE_PARAMETERS, PPM_WINE, WINE_labels

    def load_alko_data(self):
        mat = scipy.io.loadmat(self.data_path / 'NMR_mix_DoE.mat')
        X_ALKO = mat.get('xData')
        Y_ALKO = mat.get('yData')
        ALKO_labels = mat.get('yLabels')
        axis = mat.get("Axis")
        return X_ALKO, Y_ALKO, ALKO_labels, axis

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
        from helpers.generators import multiplet
        H[0] = multiplet(t, 3, 6000, 110 * 2, 900) + multiplet(t, 1, 12000, 160 * 2, 0)
        H[1] = multiplet(t, 2, 2000, 150 * 2, 800) + multiplet(t, 2, 14000, 240 * 2, 1200)
        H[2] = multiplet(t, 3, 18000, 300 * 2, 1300) + multiplet(t, 4, 12000, 120 * 2, 800)
        X_ART = self.shift_dataset(W, H, tau)
        NOISE_ART = np.random.normal(0, 5e-6, X_ART.shape)
        X_ART_NOISY = X_ART + NOISE_ART
        return X_ART_NOISY, H, W, tau

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

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)

if __name__ == "__main__":
    typer.run(preprocess)
