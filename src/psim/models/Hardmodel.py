import torch
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from helpers.callbacks import ChangeStopper, ImprovementStopper
from helpers.losses import frobeniusLoss, VolLoss
import scipy
import numpy as np
from nnls_l1 import nnls
from nlars import calc_scoring, get_optimal_W


torch.manual_seed(3)

torch.autograd.set_detect_anomaly(True)

class Hard_Model(torch.nn.Module):
    def __init__(self, X, H, peak_means, peak_sigmas, peak_n,):
        super().__init__()
        means = []
        heights = []
        mult = []
        sigma = []
        J_coup = []
        n = []
        print("hypothesises:")
        print(H)
        self.H = H

        #Number of single peaks
        self.nr_peaks = len(peak_means)

        for hyp in H:
            if len(hyp) > 1:
                if peak_means[hyp[1]]-peak_means[hyp[0]]<2000:
                    means.append(np.mean([peak_means[i] for i in hyp]))
                    mult.append(len(hyp))
                    sigma.append(np.mean([peak_sigmas[i] for i in hyp]))
                    n.append(np.mean([peak_n[i] for i in hyp]))
                    if len(hyp) > 1:
                        J_coup.append(peak_means[hyp[1]]-peak_means[hyp[0]])
                    else:
                        J_coup.append(0)
            else:
                means.append(np.mean([peak_means[i] for i in hyp]))
                mult.append(len(hyp))
                sigma.append(np.mean([peak_sigmas[i] for i in hyp]))
                n.append(np.mean([peak_n[i] for i in hyp]))
                J_coup.append(0)


        #get the heights from the X data
        # print(X.shape)
        # print(means)
        for i in range(len(means)):
            height = X[int(means[i])]
            heights.append(height)
        
        # print('Means')
        # print(means)
        # print('Heights')
        # print(heights)
        
        rank = len(means)
        self.X = torch.tensor(X)
        if len(X.shape) == 1:
            self.X = torch.unsqueeze(self.X,dim=0)
            n_row = 1
            n_col = X.shape[0]
        else:
            n_row, n_col = X.shape
        
        self.softplus = torch.nn.Softplus()
        # self.softmax = torch.nn.Softmax()
        self.n_row = n_row # nr of samples
        self.n_col = n_col
        self.rank = rank
        
        
        self.std = torch.std(self.X)
        self.X = self.X/self.std
        
        self.lossfn = frobeniusLoss(self.X.clone().detach())
        
        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        #Weights of each component
        self.W = torch.nn.Parameter(torch.rand(n_row, rank, requires_grad=True))
        self.sigma = torch.nn.Parameter(torch.tensor(sigma, requires_grad=False,dtype=torch.float32))
        self.spacing = torch.nn.Parameter(torch.tensor(J_coup, requires_grad=False,dtype=torch.float32))
        self.means = torch.tensor(means,dtype = torch.float32, requires_grad=False)
        self.N = torch.tensor(n, dtype=torch.float32, requires_grad=False)
        

        self.multiplicity = torch.tensor(mult,dtype=torch.int32, requires_grad=False)
        self.forward()
        

    def pascal(self, x):
        triangle = torch.zeros((x, x))
        for i in range(x):
            triangle[i, 0] = 1
            triangle[i, i] = 1
            if i > 0:
                for j in range(1, i):
                    triangle[i, j] = triangle[i - 1, j - 1] + triangle[i - 1, j]
        return triangle[-1]

    def lorentzian(self, x, mean, variance):
        #print(variance) nan problem was the variance
        return 1 / (torch.pi * variance * (1 + ((x - mean) / variance) ** 2))
    def gauss(self, x, mean, variance):
        return 1/(variance*(2*torch.pi)**(1/2))*torch.exp(-1/2*((x-mean)/variance)**2)
    def voigt(self, x, mean, fwhm, n):
        return n*self.lorentzian(x,mean,fwhm/2)+(1-n)*self.gauss(x,mean,fwhm/(2*torch.sqrt(2*torch.log(torch.tensor(2)))))

    def multiplet(self, x, mult, mean, sigma, spacing, n):
        triangle = self.pascal(mult)
        t_max = torch.max(triangle)
        triangle = triangle/t_max
        
        #find the largest value from the mean in H and multiply it with the multiplicity
        # height = self.H[torch.argmax(self.H[:,mean])]*mult
        if (mult%2 == 0):
            triangle = triangle*self.X[0, int(mean + spacing/2)]
        else:
            triangle = triangle*self.X[0, int(mean)]
        
        y = torch.zeros(len(x),dtype=float)

        if mult%2 == 0:
            space = -1*mult/2*spacing+spacing/2
        else:
            space = -1*(mult-1)/2*spacing
        for i,size in enumerate(triangle):
                y += self.voigt(x, mean+space, sigma, n)*size
                space +=  spacing
        return y

    def forward(self):
        time = torch.linspace(0,self.n_col,self.n_col)
        self.C = torch.zeros((self.rank, self.n_col),requires_grad=False)
        
        for i in range(self.rank):
            self.C[i] += self.multiplet(time,
                                    self.multiplicity[i],
                                    self.means[i],
                                    torch.clamp(self.sigma[i],1),
                                    self.softplus(self.spacing[i]),
                                    torch.sigmoid(self.N[i]))
            C_new = self.C[i].detach()#/torch.max(self.C[i].detach())
            self.C[i] = C_new
        
        # WC = torch.matmul(self.softplus(self.W), self.C) #self.softplus(self.C))
        WC = torch.matmul(self.W, self.C)
        return WC
    
    def fit_grad(self, grad):
        stopper = ChangeStopper(alpha=1e-3, patience=3)
        improvement_stopper = ImprovementStopper(min_improvement=1e-3, patience=5)

        while not stopper.trigger() and not improvement_stopper.trigger():
            grad.zero_grad()
            output = self.forward()
            loss = self.lossfn.forward(output)
            print(f"Loss: {loss.item()}", end='\r')
            loss.backward()
            grad.step()
            stopper.track_loss(loss)
            improvement_stopper.track_loss(loss)
        print(f"Loss: {loss.item()}")

    def fit_W(self, threshold=0.25):
        # W, path, lambdas = calc_scoring(self.X.detach().numpy(), self.C.detach().numpy(), inc_path=True, maxK=self.nr_peaks)
        W, path, lambdas = get_optimal_W(self.X.detach().numpy(), self.C.detach().numpy(), threshold)

        losses = [np.linalg.norm(self.X.detach().numpy() - w @ self.C.detach().numpy(), 'fro')/np.linalg.norm(self.X.detach().numpy(), 'fro') for w in path[0].T]
        W = torch.tensor(W, dtype=torch.float32)
        self.W = torch.nn.Parameter(W)
        return path, lambdas, losses

    def return_values(self):
        W = self.W.detach().numpy()
        active_values = [w>0 for w in W[0]]
        index_filter = [i for i, x in enumerate(active_values) if x]
        means = self.means.detach().numpy()
        sigma = self.sigma.detach().numpy()
        j_coup = self.spacing.detach().numpy()
        mult = self.multiplicity.detach().numpy()
        n = torch.sigmoid(self.N).detach().numpy()
        return means[index_filter], sigma[index_filter], j_coup[index_filter], mult[index_filter], n[index_filter], W[0][index_filter]

    
    def fit(self, verbose=False, return_loss=False, threshold=0.25):
        path, lambdas, losses = self.fit_W(threshold=threshold)

        # # forward
        output = self.forward()

        #loss calc
        loss = self.lossfn.forward(output)


        # print loss
        if verbose:
            print(f"Loss: {loss.item()}")
            # print(f"epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

        # W = self.softplus(self.W).detach().numpy()
        W = self.W.detach().numpy()
        C = self.C.detach().numpy()
        if return_loss:
            return W, C, loss, path, lambdas, losses
        else:
            return W, C



if __name__ == "__main__":
    from helpers.callbacks import explained_variance
    #from helpers.data import X_clean
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')
    
    # X = mat.get('xData')
    # targets = mat.get('yData')
    # target_labels = mat.get('yLabels')
    # axis = mat.get("Axis")
    X  = pd.read_csv("X_duplet.csv").to_numpy()
    alpha = 1e-3
    model = Hard_Model(X, [1000, 4000, 8000.], lr=10, alpha = alpha, factor=1, patience=1, min_imp=0.01) # min_imp=1e-3)
    model.forward()
    C_ini = model.C.detach().numpy()
    for c in C_ini:
        plt.plot(c)
    plt.show()

    W, C = model.fit(verbose=True)
    #print(W)
    print(scipy.special.softmax(model.multiplicity.detach().numpy(), axis=1))
    #print(f"Explained variance HardModel: {explained_variance(model.X.detach().numpy(), np.matmul(W, C))}")
    #print(np.matmul(W, C).shape) 
    plt.figure()
    for vec in C:
        plt.plot(vec)
    plt.title("C")
    plt.show()

    plt.plot(model.X.detach().numpy()[0])
    plt.plot(np.matmul(W,C)[0])
    plt.show()

    # for xt in model.X.detach().numpy():
    #     plt.plot(xt)
    # plt.show()
