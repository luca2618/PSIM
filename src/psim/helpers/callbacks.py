import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_data(X, title=""):
    plt.figure()
    for signal in X:
        plt.plot(signal)
    plt.title(title)
    plt.show()

def NMI(S, Sp):
    """
    Function t compute the normalized mutul information between two probability distributions. The probability
    distributions must be represented as matrices N x M such that each row is a realization of M random variables
    forming a joint probability distribution. We consider the case where we want to comapre an estimate to the
    ground truth
    :param mixing: The estimate of the probability distribution computed by some factorization model
    :param org_mixing: The ground truth of the dsitribution
    :return: A number in [0, 1] which is 1 if mixing and org_mixing are identical and 0 if they share no information.
    """
    N, M = S.shape

    def I(A, B):
        pdd = 1 / N * np.matmul(A.T, B)
        pd = 1 / N * np.sum(A, axis=0)
        pdp = 1 / N * np.sum(B, axis=0)
        info = 0
        for i in range(M):
            for j in range(M):
                info += pdd[i, j]*np.log(pdd[i, j]/(pd[i]*pdp[j]))

        return info

    NMI = 2*I(S, Sp) / (I(S, S) + I(Sp, Sp))
    return NMI

def explained_variance(original_data, reconstructed_data):
    """
    Calculate the explained variance between original and reconstructed data.

    Args:
        original_data (numpy.ndarray): The original dataset.
        reconstructed_data (numpy.ndarray): The reconstructed dataset.

    Returns:
        float: The explained variance score.
    """
    numerator = np.sum(np.square(original_data - reconstructed_data))
    denominator = np.sum(np.square(original_data - np.mean(original_data)))
    explained_variance = 1 - (numerator / denominator)

    return explained_variance


def train_n_times(number, object, data, components, **kwargs):
    """
    Trains "number" versions on Instance and return the parameters of the one with the lowest loss
    :param object: An algorithm class, which returns a loss.
    :param data: The dataset to fit the algorithm on
    :param components: The number of latent components to use
    :param number: The number of versions to train
    :param **kwargs: The keyword arguments that you want to pass to the class instance upon initialisation.
    These are the hyperparameters for the model, learning rate, patience, alpha etc.
    :return: The parameters and running loss of the best model
    """
    params = []
    losses = []
    for i in range(number):
        print(f"Training model {i+1}/{number}")
        model = object(data, components, **kwargs)
        returns = model.fit(return_loss=True)
        losses.append(returns[-1][-1]) # Loss is always the last element returned
        params.append(returns[:len(returns)-1])
    best_params = params[np.argmin(losses)]
    return best_params, np.min(losses)

#Superclass for stopping criteria
class Stopper:
    def __init__(self) -> None:
        pass

    # Function for tracking loss - to be implemented in subclasses
    def track_loss(self):
        pass
    
    # Function for triggering stop - to be implemented in subclasses
    def trigger(self):
        pass
    
    # Function for resetting stopper - to be implemented in subclasses
    def reset(self):
        pass

class ChangeStopper(Stopper):
    def __init__(self, alpha=1e-8, patience=5):
        self.alpha = alpha
        self.ploss = None
        self.loss = None
        
        self.patience = patience
        self.counter = 0

    def track_loss(self, loss):
        if self.loss is None:
            self.loss = loss
        else:
            if self.ploss is not None:
                if self.loss<self.ploss:
                    self.ploss = self.loss
            self.loss = loss
        
        if self.ploss is not None:
            if abs((self.ploss - self.loss)/self.ploss) < self.alpha:
                self.counter += 1
            else:
                self.counter = 0

    def trigger(self):
        return self.counter >= self.patience

    def reset(self):
        self.ploss = None
        self.loss = None
        self.counter = 0

class ImprovementStopper(Stopper):
    def __init__(self, patience=10, min_improvement=1e-4):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.min_improvement = min_improvement
    
    def track_loss(self, loss):
        if loss < self.best_loss - self.min_improvement:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
    
    def trigger(self):
        return self.counter >= self.patience
    
    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')
            

class ConvergenceCriterion:
    def __init__(self, x, convergence_threshold, num_epochs_convergence):
        self.norm = torch.linalg.matrix_norm(x, ord="fro")**2
        self.convergence_threshold = convergence_threshold
        self.num_epochs_convergence = num_epochs_convergence
        self.previous_loss = float('inf')
        self.epochs_since_last_improvement = 0

    def trigger(self, current_loss):
        current_loss /= self.norm
        self.previous_loss /= self.norm
        if torch.abs(self.previous_loss - current_loss)/self.previous_loss < self.convergence_threshold:
            self.epochs_since_last_improvement += 1
        else:
            self.epochs_since_last_improvement = 0
        print(f"Loss difference {torch.abs(self.previous_loss - current_loss)/self.previous_loss}")
        self.previous_loss = current_loss

        return self.epochs_since_last_improvement >= self.num_epochs_convergence
