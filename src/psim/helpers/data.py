import scipy.io
import numpy as np



# import matplotlib.pyplot as plt
#load data from .MAT file
mat = scipy.io.loadmat('helpers/data/nmrdata.mat')
#Get X and Labels. Probably different for the other dataset, but i didn't check :)
mat = mat.get('nmrdata')
X_URINE = mat[0][0][0]
labels_URINE = mat[0][0][1]


#OIL DATA
mat = scipy.io.loadmat('helpers/data/nmrdata_Oil_group3.mat')
#Get X and Labels. Probably different for the other dataset, but i didn't check :)
mat = mat.get('nmrdata_Oil_group3')
X_OIL = mat[0][0][0]
OIL_labels = mat[0][0][1]

#WINE DATA
mat = scipy.io.loadmat('helpers/data/NMR_40wines.mat')
#40 wines times 8712 length spectrum
X_WINE = mat.get('X')
WINE_PARAMETERS = mat.get('Y')
#ppm is the scale of the x-axis.
PPM_WINE = mat.get('ppm')[0]

labels = mat.get('Label')
# #try to uncover mixings
WINE_labels = [x[0] for x in labels[0]]

# load data from .MAT file
mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')
# Get X and Labels. Probably different for the other dataset, but i didn't check :)
X_ALKO = mat.get('xData')
Y_ALKO = mat.get('yData')
ALKO_labels = mat.get('yLabels')
axis = mat.get("Axis")



#functions for normalizing, and inversing the normalization of data
def normalize_data(target):
    # return (target - np.mean(target))/np.std(target)
    # don't subtract mean, resulting values would be negative
    # and not reproducible by a positive matrix
    return target/np.std(target)

def inv_normalize_data(target, std):
    # return target * std + mean
    #same as above
    return target * std

### ARTIFICIAL DATASET


N, M, d = 30, 20000, 3

t = np.arange(0, M, 1)

def shift_dataset(W, H, tau):
    # Get half the frequencies
    Nf = H.shape[1] // 2 + 1
    # Fourier transform of S along the second dimension
    Hf = np.fft.fft(H, axis=1)
    # Keep only the first Nf[1] elements of the Fourier transform of S
    Hf = Hf[:, :Nf]
    # Construct the shifted Fourier transform of S
    Hf_reverse = np.fliplr(Hf[:, 1:Nf - 1])
    # Concatenate the original columns with the reversed columns along the second dimension
    Hft = np.concatenate((Hf, np.conj(Hf_reverse)), axis=1)
    f = np.arange(0, M) / M
    omega = np.exp(-1j * 2 * np.pi * np.einsum('Nd,M->NdM', tau, f))
    Wf = np.einsum('Nd,NdM->NdM', W, omega)
    # Broadcast Wf and H together
    Vf = np.einsum('NdM,dM->NM', Wf, Hft)
    V = np.fft.ifft(Vf)
    return V.real

np.random.seed(42)

# Random mixings:
W = np.random.dirichlet(np.ones(d), N)
# W = np.append(W, [[1,0,0]], axis=0)
# W = np.append(W, [[0,1,0]], axis=0)
# W = np.append(W, [[0,0,1]], axis=0)
# N = N+3

#W = np.random.rand(N, d)
shift = 1
# Random gaussian shifts
tau = np.random.randint(-shift, shift, size=(N, d))
tau = np.zeros((N,d))
tau = np.random.randint(-1000, 1000, size=(N, d))
# Purely positive underlying signals. I define them as 3 gaussian peaks with random mean and std.
H = np.zeros((d,M))
from helpers.generators import *
H[0] = multiplet(t, 3, 6000, 110*2, 900)+multiplet(t, 1, 12000, 160*2, 0)
H[1] = multiplet(t, 2, 2000, 150*2, 800)+multiplet(t, 2, 14000, 240*2, 1200)
H[2] = multiplet(t, 3, 18000, 300*2, 1300)+multiplet(t, 4, 12000, 120*2, 800)
H_ART = H
W_ART = W
TAU_ART = tau
X_ART = shift_dataset(W, H, tau)
NOISE_ART = np.random.normal(0, 5e-6, X_ART.shape)

X_ART_NOISY = X_ART + NOISE_ART





