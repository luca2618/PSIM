import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
# from TimeCor import estT
from psim.models.estTimeAutCor import estT
from psim.helpers.callbacks import ChangeStopper, ImprovementStopper
from psim.helpers.losses import frobeniusLoss

# import matplotlib.pyplot as plt


class ShiftNMF(torch.nn.Module):
    def __init__(self, X, rank, lr=0.2, alpha=1e-4, factor=1, min_imp=1e-3, patience=10):
        super().__init__()

        self.rank = rank
        self.X = torch.tensor(X)
        self.std = torch.std(self.X)
        self.X_MAX = torch.max(self.X)
        # self.X = self.X / self.X_MAX  #self.std
        self.X = self.X / self.std
        
        self.N, self.M = X.shape

        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus()
        #scale applied to self.H
        self.scale = lambda x : self.softplus(x)
        self.lossfn = frobeniusLoss(torch.fft.fft(self.X))
        
        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.randn(self.N, rank, requires_grad=True, dtype=torch.double))
        self.H = torch.nn.Parameter(torch.randn(rank, self.M, requires_grad=True, dtype=torch.double))
        #self.H = torch.tensor(PCA_init(X.T.real, rank).real.T, requires_grad=True, dtype=torch.double)
        #self.H = torch.nn.Parameter(inv_softplus(self.H))
        self.tau = torch.zeros(self.N, self.rank,dtype=torch.double)
        
        self.stopper = ChangeStopper(alpha=alpha, patience=patience)
        
        self.optimizer = Adam([self.H], lr=lr)
        self.full_optimizer = Adam([self.W, self.H], lr=lr)
        self.improvement_stopper = ImprovementStopper(min_improvement=min_imp, patience=patience)
        
        if factor < 1:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience)
        else:
            self.scheduler = None

    def forward(self):
        # Get half of the frequencies
        Nf = self.M // 2 + 1
        # Fourier transform of H along the second dimension
        Hf = torch.fft.fft(self.scale(self.H), dim=1)[:, :Nf]
        # Keep only the first Nf[1] elements of the Fourier transform of H
        # Construct the shifted Fourier transform of H
        Hf_reverse = torch.flip(Hf[:, 1:Nf-1], dims=[1])
        # Concatenate the original columns with the reversed columns along the second dimension
        Hft = torch.cat((Hf, torch.conj(Hf_reverse)), dim=1)
        f = torch.arange(0, self.M) / self.M
        omega = torch.exp(-1j * 2 * torch.pi * torch.einsum('Nd,M->NdM', self.tau, f))
        Wf = torch.einsum('Nd,NdM->NdM', self.W, omega)
        # Wf = torch.einsum('Nd,NdM->NdM', self.softplus(self.W), omega)
        # Broadcast Wf and H together
        V = torch.einsum('NdM,dM->NM', Wf, Hft)
        return V
    
    def fit_tau(self, update_T = True, update_W = True):
        X = np.array(self.X.detach().numpy().real, dtype=np.double)
        
        # W = np.array(self.softplus(self.W).detach().numpy(), dtype=np.complex128)
        H = np.array(self.scale(self.H).detach().numpy(), dtype=np.double)
        
        tau = np.array(self.tau.detach().numpy(), dtype=np.double)
        
        # W = np.zeros((self.N, self.rank))
        W = np.array(self.W.detach().numpy(), dtype=np.double)

        T, A = estT(X,W,H, Lambda = self.Lambda)
        # T, A = estT(X, W, H, tau, Lambda = self.Lambda)
        # W = inv_softplus(W.real)

        if update_T:
            self.tau = torch.tensor(T, dtype=torch.double)
        # self.tau = torch.tensor(T, dtype=torch.cdouble)

        # self.W = torch.nn.Parameter(W)
        if update_W:
            self.W = torch.nn.Parameter(torch.tensor(A, dtype=torch.double, requires_grad=True))
    
    def center_tau(self):
        tau = self.tau.detach().numpy()
        mean_tau = np.mean(tau, axis=0)
        mean_tau = np.round(mean_tau)
        #convert means to tuple of ints
        mean_tau = tuple(mean_tau.astype(int))

        H_roll = torch.zeros_like(self.H, dtype = torch.double)
        for i in range(self.rank):
            H_roll[i] = torch.roll(self.H[i],
                                mean_tau[i])
            
        # H_roll = torch.roll(self.H, shifts = mean_tau, dims = 3)
        
        tau = tau - mean_tau
        
        self.tau = torch.tensor(tau, dtype=torch.double)
        self.H = torch.nn.Parameter(H_roll)
    def fit_grad(self, grad):
        stopper = ChangeStopper(alpha=1e-5, patience=5)
        improvement_stopper = ImprovementStopper(min_improvement=1e-5, patience=5)

        while not stopper.trigger() and not improvement_stopper.trigger():
            grad.zero_grad()
            output = self.forward()
            loss = self.lossfn.forward(output)
            print(f"Loss: {loss.item()}", end='\r')
            loss.backward()
            grad.step()
            stopper.track_loss(loss)
            improvement_stopper.track_loss(loss)
    
    def fit(self, verbose=False, return_loss=False, max_iter = 15000, tau_iter=100, Lambda=0):
        self.Lambda = Lambda
        running_loss = []
        self.iters = 0
        self.tau_iter = tau_iter
        while not self.stopper.trigger() and self.iters < max_iter and not self.improvement_stopper.trigger():
        #while self.iters < max_iter:
            self.iters += 1
            # zero optimizer gradient
            self.optimizer.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn(output)
            
            loss.backward()

            # Update H
            self.optimizer.step()

            # Update W and tau
            if (self.iters%20) == 0 and self.iters > tau_iter:
                #print loss before updating tau
                # output = self.forward()
                # loss_pre = self.lossfn(output)
                # print(f"epoch: {len(running_loss)}, Loss: {loss_pre.item()}, Tau: {torch.norm(self.tau)}")
                self.fit_tau(update_T = True, update_W = True)
                #print loss after updating tau
                # output = self.forward()
                # loss_pos = self.lossfn(output)
                # print(f"epoch: {len(running_loss)}, Loss: {loss_pos.item()}, Tau: {torch.norm(self.tau)}")
                # if loss_pos > loss_pre:
                #     print("HERE")
                # print('-'*50)

            if self.scheduler != None:
                self.scheduler.step(loss)
            
            running_loss.append(loss.item())
            self.stopper.track_loss(loss)
            self.improvement_stopper.track_loss(loss)
            
            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}, Tau: {torch.norm(self.tau)}", end='\r')
        if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}, Tau: {torch.norm(self.tau)}")
        #self.fit_grad(self.full_optimizer)
        self.center_tau()
        
        W = self.W.detach().numpy()
        H = (self.scale(self.H)*self.std).detach().numpy()
        tau = self.tau.detach().numpy()
        tau = np.array(tau, dtype=np.int32)

        output = self.forward()
        self.recon = torch.fft.ifft(output)*self.std

        if return_loss:
            return W, H, tau, running_loss
        else:
            return W, H, tau

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    # from TimeCor import *
    
    #seed random
    np.random.seed(45)
    torch.manual_seed(45)
    
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

    N, M, d = 5, 10000, 3
    Fs = 1000  # The sampling frequency we use for the simulation
    t0 = 10    # The half-time interval we look at
    t = np.arange(-t0, t0, 1/Fs)  # the time samples
    f = np.arange(-Fs/2, Fs/2, Fs/len(t))  # the corresponding frequency samples

    def gauss(mu, s, time):
        return 1/(s*np.sqrt(2*np.pi))*np.exp(-1/2*((time-mu)/s)**2)

    W = np.random.dirichlet(np.ones(d), N)

    shift = 500
    # Random gaussian shifts
    tau = np.random.randint(-shift, shift, size=(N, d))
    # tau = np.random.randn(N, d)*shift
    tau = np.array(tau, dtype=np.int32)

    mean = [1500, 5000, 8500]
    std = [30, 40, 50]
    t = np.arange(0, 10000, 1)

    H = np.array([gauss(m, s, t) for m, s in list(zip(mean, std))])

    X = shift_dataset(W, H, tau)
        
    # X  = pd.read_csv("X_duplet.csv").to_numpy()

    # X = np.pad(X, ((0, 0), (1000, 1000)), 'edge')
    
    # plt.plot(X.T)
    # plt.show()
    # exit()
    
    alpha = 1e-5
    nmf = ShiftNMF(X, 3, lr=0.1, alpha=1e-6, patience=1000, min_imp=0)
    W_est, H_est, tau_est = nmf.fit(verbose=1, max_iter=100, tau_iter=0, Lambda=0)
    
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(tau)
    ax[1].imshow(tau_est)
    plt.show()
    
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(W)
    ax[1].imshow(W_est)
    
    plt.show()
    
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(np.dot(W, H).T)
    ax[1].plot(np.dot(W_est, H_est).T)
    
    # ax[1].plot(inv_softplus(H).T)
    
    plt.show()
    
    # fig, ax = plt.subplots(3, 1)
    # ax[0].plot(X.T)
    # ax[1].plot(nmf.recon.detach().numpy().T)
    # ax[2].plot(np.dot(W_est, H_est).T)
    # plt.show()
    
    # plt.imshow(tau.real)
    # plt.imshow(W)
    # plt.show()