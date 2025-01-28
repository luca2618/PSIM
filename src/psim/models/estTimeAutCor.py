import numpy as np 
import matplotlib.pyplot as plt
from scipy.fft import ifft
# from scipy.optimize import nnls
from nnls_l1 import nnls


def estTimeAutCor(Xf, A, Sf, krSf, krf, Tau, Nf, N, w, TauW):
    TauW = generateTauWMatrix(TauW,N[1])

    Xf = np.expand_dims(Xf, axis=0)
    A = np.expand_dims(A, axis=0)
    #make A complex
    A = np.array(A, dtype=np.complex128)
    noc = A.shape[1]
    if N[1] % 2 == 0:
        sSf = 2 * Nf[1] - 2
    else:
        sSf = 2 * Nf[1] - 1
    t1 = np.random.permutation(A.shape[0])
    t2 = np.random.permutation(noc)
    for k in t1:
        Resf = Xf[k, :] - np.dot(A[k, :], (krSf * np.exp(np.conj(Tau[k]).T * krf)))
        for d in t2:
            if np.sum(TauW[d, :]) > 0:
                Resfud = Resf + A[k, d] * (krSf[d, :] * np.exp(Tau[d] * krf))
                Xft = Resfud
                Xd = Xft
                
                C = Xd * np.conj(Sf[d, :])
                if N[1] % 2 == 0:
                    C = np.concatenate((C, np.conj(C[-2:0:-1])))
                else:
                    C = np.concatenate((C, np.conj(C[:0:-1])))
                C = np.fft.ifft(C, axis=0)
                C = C * TauW[d, :]
                
                ind = np.argmax(C)
                
                Tau[d] = ind - sSf - 1
                
                # A[k, d] = C[ind] / (np.sum(w * (krSf[d, :] * np.conj(krSf[d, :]))) / sSf + Lambda[d])
                #A[k, d] = C[ind] / (np.sum(w * (krSf[d, :] * np.conj(krSf[d, :]))) / sSf)
                
                if abs(Tau[d]) > (sSf / 2):
                    if Tau[d] > 0:
                        Tau[d] = Tau[d] - sSf
                    else:
                        Tau[d] = Tau[d] + sSf
                Resf = Resfud - A[k,d] * (krSf[d, :] * np.exp(Tau[d] * krf))
    return Tau.real, A.real



def generateTauWMatrix(TauW, N2):
    TauWMatrix = np.zeros((TauW.shape[0], N2))

    for d in range(TauW.shape[0]):
        TauWMatrix[d, 0:int(TauW[d, 1])] = 1
        TauWMatrix[d, N2+int(TauW[d, 0]):N2] = 1

    return TauWMatrix



def estT(X,W,H, Tau=None, Lambda=0):
    N = [*X.shape,1]
    Xf = np.fft.fft(X)
    Xf = np.ascontiguousarray(Xf[:,:int(np.floor(Xf.shape[1]/2))+1])
    Nf = np.array(Xf.shape)
    # A = np.copy(W)
    A = W
    
    noc = A.shape[1]
    Sf = np.ascontiguousarray(np.fft.fft(H)[:,:Nf[1]])
    krSf = np.conj(Sf)
    krf = (-1j*2*np.pi * np.arange(0,N[1])/N[1])[:Nf[1]]
    # Tau = np.zeros((N[0],noc))
    if Tau is None:
        Tau = np.zeros((N[0],noc))
    N = np.array(N)
    w = np.ones(Xf.shape[1])
    #TauW = np.column_stack((np.ones((3,1)) * -N[1]*2 / 2, np.ones(3) * N[1] / 2))
    TauW = np.ones((noc, 1))*np.array([-1000,1000])
    SST = np.sum(np.square(X))
    
    miss_ind = np.isnan(X)
    SNR = 1
    sigma_sq = SST / ((1 + 10 ** (SNR / 10)*np.prod(N)))

    
    # print(sigma_sq)
    # exit()
    # Lambda = np.ones(noc)*sigma_sq.real
    # Lambda *= 0.5
    for i in range(N[0]):
        Tau[i], _ = estTimeAutCor(Xf[i],A[i],Sf,krSf,krf,Tau[i],Nf,N,w,TauW)
    
    # Tau = np.array(Tau,dtype=np.float64)
    
    #subtract mean of tau in each column
    # Tau = Tau - np.mean(Tau, axis=0)
    
    #update A by nnls (non negative least square) REMOVE to estimate with estTimeAutCor
    H_shifted = np.zeros_like(H)
    for i in range(N[0]):
        for j in range(H.shape[0]):
            H_shifted[j] = np.roll(H[j], int(Tau[i,j]))
    
        A[i] = nnls(H_shifted.T, X[i], Lambda, A[i])
    #     print(H_shifted.T.real.shape, X[i].real.shape, A[i].real.shape)

    
    # Hf = np.fft.fft(H)
    # f = np.arange(0, H.shape[1]) / H.shape[1]
    # omega = np.exp(-1j * 2 * np.pi * np.einsum('Nd,M->NdM', Tau, f))
    # for i in range(N[0]):
    #     H_shifted = np.einsum('dM,dM->dM', omega[i], Hf).real
    #     A[i] = nnls(H_shifted.T, X[i], Lambda, A[i])
    #     # print(H_shifted.T.real.shape, X[i].real.shape, A[i].real.shape)
    
    return Tau, A



if __name__ == "__main__":
    np.random.seed(45)
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
        return V

    N, M, d = 5, 10000, 3
    Fs = 1000  # The sampling frequency we use for the simulation
    t0 = 10    # The half-time interval we look at
    t = np.arange(-t0, t0, 1/Fs)  # the time samples
    f = np.arange(-Fs/2, Fs/2, Fs/len(t))  # the corresponding frequency samples

    def gauss(mu, s, time):
        return 1/(s*np.sqrt(2*np.pi))*np.exp(-1/2*((time-mu)/s)**2)

    W = np.random.dirichlet(np.ones(d), N)

    shift = 400
    # Random gaussian shifts
    tau = np.random.randint(-shift, shift, size=(N, d))

    mean = [1500, 5000, 8500]
    std = [30, 40, 50]
    t = np.arange(0, 10000, 1)

    H = np.array([gauss(m, s, t) for m, s in list(zip(mean, std))])

    X = shift_dataset(W, H, tau)
    W_before = np.copy(W)
    W=np.zeros_like(W)
    tau_est, A = estT(X,W,H)
    # print(W)
    
    W = A
    plt.subplot(1, 2, 1)
    plt.imshow(W_before)
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(W)
    plt.colorbar()
    plt.show()


    plt.subplot(2,1,1)
    plt.plot(shift_dataset(W_before, H, tau).real.T)
    plt.subplot(2,1,2)
    plt.plot(shift_dataset(W, H, tau_est).real.T)

    plt.show()
    