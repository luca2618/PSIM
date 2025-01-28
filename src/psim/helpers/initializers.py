import numpy as np
from numpy.matlib import repmat

try:
    from helpers.losses import frobeniusLoss
except:
    from losses import frobeniusLoss

def FurthestSum(K, noc, i, exclude=[]):
    """
    Python implementation of Morten Mørup's Furthest Sum algorithm

    FurthestSum algorithm to efficiently generate initial seeds/archetypes

    Input:
    K           Either a data matrix or a kernel matrix
    noc         number of candidate archetypes to extract
    i           inital observation used for to generate the FurthestSum
    exclude     Entries in K that can not be used as candidates

    Output:
    i           The extracted candidate archetypes

    """
    I, J = K.shape
    index = np.ones(J, dtype=bool)
    index[exclude] = False
    index[i] = False
    ind_t = i
    sum_dist = np.zeros(J)
    noc = noc-1 if noc > 1 else 1 # prevent error when noc = 1, subscript of int
    if J > noc * I:
        # Fast implementation for large scale number of observations. Can be improved by reusing calculations
        Kt = K.copy()
        Kt2 = np.sum(Kt**2, axis=0)
        for k in range(noc + 10):
            if k > noc - 1: #remove initial seed
                Kq = Kt[:, i[0]].conj().T @ Kt
                sum_dist = sum_dist - np.emath.sqrt(Kt2 - 2*Kq + Kt2[i[0]])
                index[i[0]] = True
                i = i[1:]
            t = np.where(index)[0]
            Kq = Kt[:, ind_t].conj().T @ Kt
            sum_dist = sum_dist + np.emath.sqrt(Kt2 - 2*Kq + Kt2[ind_t])
            ind = np.argmax(sum_dist[t])
            ind_t = t[ind]
            i = np.append(i, ind_t)
            index[ind_t] = False
    else:
        # Generate kernel if K not a kernel matrix
        if I != J or np.sum(K - K.conj().T) != 0:
            Kt = K.copy()
            K = Kt.conj().T @ Kt
            K = np.lib.scimath.sqrt(
                repmat(np.diag(K), J, 1) - 2 * K + \
                repmat(np.mat(np.diag(K)).T, 1, J)
            )
        Kt2 = np.diag(K).conj().T
        for k in range(noc + 11):
            if k > noc - 1:
                sum_dist = sum_dist -np.lib.scimath.sqrt(Kt2 - 2*K[i[0], :] + Kt2[i[0]])
                index[i[0]] = True
                i = i[1:]
            t = np.where(index)[0]
            sum_dist = sum_dist + np.lib.scimath.sqrt(Kt2 - 2*K[ind_t, :] + Kt2[ind_t])
            ind = np.argmax(sum_dist[t])
            ind_t = t[ind]
            i = np.append(i, ind_t)
            index[ind_t] = False
    return i

import torch

class S_fit(torch.nn.Module):
    def __init__(self, X, C):
        super(S_fit, self).__init__()
        self.noc = C.shape[0]
        
        self.softmax = torch.nn.Softmax(dim=1)
        
        self.C_tilde = torch.nn.Parameter(torch.Tensor(C), requires_grad=False)
        self.X = torch.tensor(X, dtype=torch.float32)
        # self.C_tilde = torch.tensor(C, dtype=torch.float32)
        self.S_tilde = torch.nn.Parameter(torch.zeros(X.shape[0], self.noc, requires_grad=True))

        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.3)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.9)
        
        self.loss_fn = frobeniusLoss(self.X)
        
    def forward(self):
        CX  = torch.matmul(self.softmax(self.C_tilde), self.X)
        
        SCX = torch.matmul(self.softmax(self.S_tilde), CX)
        
        return SCX

    def fit(self, epochs=100, return_tilde=False):
        for _ in range(epochs):
            self.optimizer.zero_grad()
            
            output = self.forward()
            
            loss = self.loss_fn.forward(output)
            loss.backward()
            
            #print gradients
            # self.C_tilde.grad = self.C_tilde.grad * 0
            # print(np.linalg.norm(self.S_tilde.grad))
            
            self.optimizer.step()
            # self.scheduler.step(loss)
            
            # print(loss.item())
        
        if return_tilde:
            return self.C_tilde.detach().numpy(), self.S_tilde.detach().numpy()
        else:
            return self.softmax(self.C_tilde).detach().numpy(), self.softmax(self.S_tilde).detach().numpy()


def init_C_S(X, rank, epochs=50, return_tilde=False):
    
    cols = FurthestSum(X.T, rank, 0)
    
    n_row, n_col = X.shape
    
    C = np.zeros((rank, n_row))
    
    for i,ele in enumerate(cols):
        C[i][ele] = 1
        
    model = S_fit(X, C)
    return model.fit(epochs=epochs)


if __name__ == "__main__":
    print("This is a helper file, import it to use it.")
    import pandas as pd
    import scipy.io
    import matplotlib.pyplot as plt
    mat = scipy.io.loadmat('data/NMR_mix_DoE.mat')
    X = mat.get('xData')
    
    rank = 3
    
    print(FurthestSum(X.T, rank, 0))
    
    # n_row, n_col = X.shape
    
    # cols = FurthestSum(X.T, rank, 0)
        
    # C = np.zeros((rank, n_row))
    
    # for i,ele in enumerate(cols):
    #     C[i][ele] = 1
    
    # C = np.random.rand(rank, n_row)
    # plt.plot(X.T)
    
    # S, C = fit_s(X, C, epochs=100)
    # rec = np.dot(X, S)
    # A = np.dot(C, X)
    # rec = np.dot(S, A)
    
    C, S = init_C_S(X, rank, epochs=100)
    A = np.dot(C, X)
    rec = np.dot(S, A)
    
    # plt.imshow(C)
    # plt.imshow(S, aspect='auto', interpolation="none")
    # plt.show()
    plt.plot(A.T)
    plt.show()
    plt.plot(X.T)
    plt.show()
    

def PCA_init(X, noc):
    #uses PCA to initialize the H of the NMF
    from sklearn.decomposition import PCA
    pca = PCA(n_components=noc)
    pca.fit(X)
    return pca.transform(X)