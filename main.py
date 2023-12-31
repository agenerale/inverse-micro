import numpy as np
import torch
import scipy.io as sio
from scipy.spatial import Delaunay
import os
import h5py
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
torch.set_default_dtype(torch.float32)
import torch.optim as optim
import gpytorch
import matplotlib.pyplot as plt
from pickle import load
import argparse

from models.gmm import GaussianMixture
from models.mogp import MultitaskGPModel
from models.realnvp import RealNVP
from models.vae import betaVAE
from post import pcaposterior, twopsposterior

parser = argparse.ArgumentParser(description="Deep Probabilistic Inverse Microstructure UQ")
parser.add_argument("--train", action='store_false', help="train (True) cuda")
parser.add_argument("--load", action='store_true', help="load pretrained model")
parser.add_argument("--train_gmm", action='store_true', help="train GMM prior (True)")
parser.add_argument("--use_gmm", action='store_true', help="use GMM prior (True) or N(0,1) (False)")
parser.add_argument("--micro", default=0, type=int, help="[0, 1, 2, 3]")
parser.add_argument("--n_flow", default=48, type=int, help="number of flows in RealNVP")
parser.add_argument("--n_epoch", default=40000, type=int, help="number of epochs for training RealNVP")
parser.add_argument("--lr_init", default=1e-3, type=float, help="init. learning rate")
parser.add_argument("--lr_end", default=1e-10, type=float, help="end learning rate")
parser.add_argument("--clip", default=1e-5, type=float, help="gradient clip for neural network training")
parser.add_argument("--n_samples", default=1024, type=int, help="minibatch size")
parser.add_argument("--vae_latent", default=64, type=int, help="VAE latend dimension")
parser.add_argument("--vae_input", default=1024, type=int, help="# PCs input to VAE")
parser.add_argument("--pcs_gp", default=16, type=int, help="# PCs for GP reduced-order model")
parser.add_argument("--logprior_weight", default=1.0, type=float, help="weight on prior")
parser.add_argument("--logdet_weight", default=1.0, type=float, help="diversity loss weight")
parser.add_argument("--n_gmm", default=1, type=int, help="# GMM components")
parser.add_argument("--beta", default=100, type=int, help="beta-VAE hyperparameter")
args = parser.parse_args()

device = torch.device("cuda")

microindx_array = [1556,16,1373]
microindx = microindx_array[args.micro]

###############################################################################
# load target thermal conductivity and latent space of VAE
with h5py.File("inputs.h5", "r") as f:
    print("Keys: %s" % f.keys())
    pcstot = f['pcs'][()]
    output = f['k'][()]
    ztot = f['z'][()]
    f.close()

pcstot = torch.from_numpy(pcstot).float().to(device)
output = torch.from_numpy(output).float().to(device)
ztot = torch.from_numpy(ztot).float().to(device)

pcstot_min = pcstot.min(0)[0].to(device)
pcstot_max = pcstot.max(0)[0].to(device)

Yexpmean = output[microindx,:]
print(f"Target Properties - k11: {Yexpmean[0].item():.5f}, k22: {Yexpmean[1].item():.5f}, k33: {Yexpmean[2].item():.5f}")

###############################################################################
# Create prior by GMM or N(0,1)
if all([args.use_gmm,args.train_gmm]):
    gmmmodel = GaussianMixture(args.n_gmm, args.vae_latent, covariance_type="full", eps=1e-6).to(device)
    gmmmodel.fit(ztot)
    torch.save(gmmmodel.state_dict(), 'gmm.pth')
elif args.use_gmm:
    state_dict_gmm = torch.load('gmm.pth', map_location=device)
    gmmmodel = GaussianMixture(args.n_gmm, args.vae_latent, covariance_type="full", eps=1e-6).to(device)
    gmmmodel.load_state_dict(state_dict_gmm)
else: 
    prior_z = MultivariateNormal(torch.zeros(args.vae_latent).to(device),torch.eye(args.vae_latent).to(device)) 
    
###############################################################################
# Load in SV-MOGP       
num_latents = 2
num_tasks = 3
num_inducing = int(0.02*2500)
input_dims = args.pcs_gp

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)
model = MultitaskGPModel(num_latents,num_tasks,num_inducing,input_dims).to(device)

lik_params = sum(p.numel() for p in likelihood.parameters() if p.requires_grad)
model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
params_gp = lik_params + model_params

state_dict_model = torch.load('mogp_model_state.pth', map_location=device)
state_dict_likelihood = torch.load('mogp_likelihood_state.pth', map_location=device)
model.load_state_dict(state_dict_model)
likelihood.load_state_dict(state_dict_likelihood)

# Load in VAE
vae = betaVAE(args.vae_input, args.vae_latent).to(device)
state_dict = torch.load('vae.pth', map_location=device)
vae.load_state_dict(state_dict)
vae.eval()

###############################################################################
# define prior and likelihood
def predictK(pcs,device):
    """forward predictions of property set given input PC scores"""
    train_predictions = likelihood(model(pcs))
    train_mean = train_predictions.mean
    
    Km = train_predictions.lazy_covariance_matrix.diag()
    K = Km.reshape([pcs.shape[0], 3])
    
    return train_mean, K

def log_likelihood(pcs, device):
    """liklihood given target property set with main diagonal of covariance from GP"""
    Ymodel, K = predictK(pcs, device)
    Yexp = Yexpmean.to(device = device)
    
    # no noise
    log_likelihood = ((Yexp - Ymodel)**2)/K + torch.log(2*torch.pi*K)   
    log_likelihood = -0.5*torch.sum(log_likelihood,-1)
   
    return log_likelihood, Ymodel, K

def log_prior(z, device):
    """specification of prior distribution either using GMM or N(0,1)"""
    if args.use_gmm:
        log_prior = gmmmodel.score_samples(z)
    else:
        log_prior = prior_z.log_prob(z)
    
    return log_prior

###############################################################################
# train normalizing flow
affine = True
logdet_weight = 1 # weight of the diversity loss
beta_init = 1e3
tau = 200
tanhf = 5

generator = RealNVP(args.vae_latent, args.n_flow, affine=affine, seqfrac=1/8).to(device)
optimizer = optim.Adam(generator.parameters(), lr = args.lr_init)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.n_epoch,args.lr_end)
print('Total Parameters: ' + str(sum(p.numel() for p in generator.parameters() if p.requires_grad)))
print('Beta: ' + str(args.beta))

if args.load:
    state_dict = torch.load('normflow_k_vae_' + str(microindx) + '_' + str(args.logprior_weight) + '_' + str(args.n_flow) + '.pth', map_location=device)
    generator.load_state_dict(state_dict)

if args.train:
    loss_list = []
    sample_list = []
    for k in range(args.n_epoch):
        z_sample = torch.randn((args.n_samples, args.vae_latent)).to(device)
        x, logdet = generator.reverse(z_sample)
        logdet_rec = logdet
        
        logprior = args.logprior_weight*log_prior(x,device)
        pcs = vae.decode(x)     
        pcs = 2 * (pcs[:,:args.pcs_gp] - pcstot_min)/(pcstot_max - pcstot_min) - 1
        
        loglik, kt, var = log_likelihood(pcs,device)
        kmean = torch.mean(kt,0)
        varmean = torch.mean(var,0)**0.5
        
        beta = np.max((1.0,beta_init*np.exp(-k/tau)))
        beta = 1
        logprob = (loglik + logprior)
        lossdet = (1/beta)*logdet_weight*logdet
    
        loss = torch.mean(-logprob - lossdet)
        loss_list.append(loss.detach().cpu().numpy())
    
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), args.clip)
        optimizer.step()
        scheduler.step()

        if (k + 1) % 10 == 0:
            print(f"epoch: {k+1:}, loss: {loss.item():.5f}, k: {kmean[2]:.5f}, kstd: {varmean[2]:.5f}, -loglik: {torch.mean(-loglik):.5f}, -logprior: {torch.mean(-logprior):.5f}, -logdet: {torch.mean(-logdet_rec):.5f}, beta: {beta:.5f}")
            sample_list.append(x.detach().cpu().numpy())
            
        if (k + 1) % 2000 == 0:
            torch.save(generator.state_dict(), 'normflow_k_vae_' + str(microindx) + '_' + str(args.logprior_weight) + '_' + str(args.args.n_flow) + '.pth')

###############################################################################
# post-processing results
###############################################################################
output = output.detach().cpu().numpy()
pcstot = pcstot.detach().cpu().numpy()
ztot = ztot.detach().cpu().numpy()

pcstot_hull = pcaposterior(args, predictK, vae, generator,
                 output, pcstot, ztot, microindx,
                 microindx_array, device)

#twopsposterior(args, vae, generator,
#                 output, pcstot, ztot, microindx,
#                 microindx_array, pcstot_hull, device)
    
