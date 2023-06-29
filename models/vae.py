import numpy as np
from torch.autograd import Variable
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch import distributions
import os
torch.set_default_dtype(torch.float32)
import gpytorch
from math import log, pi, exp
from scipy import linalg as la
        
class betaVAE(torch.nn.Module):
    def __init__(self,
                 in_dim: int,
                 latent_dim: int,
                 **kwargs) -> None:
        super(betaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.pcs_dim = in_dim

        modules = []
        #hidden_dims = [1024, 768, 512, 256, 256, 128, 128]   # old
        #hidden_dims = [128, 64]  
        hidden_dims = [256, 128, 64]
        #hidden_dims = [512, 256, 256, 128, 128]
        # Build Encoder
        #for h_dim in hidden_dims:
        for i in range(len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True)
                    )
                )
            in_dim = hidden_dims[i]
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.LeakyReLU(negative_slope=0.01, inplace=True)
                    )
                )
            in_dim = hidden_dims[i+1]

        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
                    #nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                    nn.Linear(hidden_dims[-1], self.pcs_dim)
                    #nn.BatchNorm1d(self.pcs_dim)
                    #nn.Tanh() # removed as pcs not scaled
                    )
            

    def encode(self, input: torch.Tensor):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: torch.Tensor):
        result = self.decoder_input(z)
        #result = result.view(-1, 128, 2, 2, 1)
        result = self.decoder(result)
        result = self.final_layer(result)
        
        return result

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset

        recons_loss = torch.sum(F.mse_loss(recons, input, reduction='none'), dim = 1)
        kld_loss = torch.sum(-0.5*(1 + log_var - mu ** 2 - log_var.exp()), dim = 1)
        loss = recons_loss.sum() + kld_weight * kld_loss.sum()
        loss = loss.sum()
        
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.sum().detach(), 'kld':-kld_loss.sum().detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):

        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor, **kwargs):

        return self.forward(x)[0]
        

def schedule_KL_annealing(start, stop, n_epochs, n_cycle=4, ratio=0.5):
    """
    Custom function for multiple annealing scheduling: Monotonic and cyclical_annealing
    Given number of epochs, it returns the value of the KL weight at each epoch as a list.
    Based on from: https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
    """

    weights = torch.ones(n_epochs)
    period = n_epochs/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epochs):
            weights[int(i+c*period)] = v
            v += step
            i += 1

    return weights  