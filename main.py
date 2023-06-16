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

parser = argparse.ArgumentParser(description="Deep Probabilistic Inverse Microstructure UQ")
parser.add_argument("--train", action='store_true', help="train (True) cuda")
parser.add_argument("--load", action='store_false', help="load pretrained model")
parser.add_argument("--train_gmm", action='store_true', help="train GMM prior (True)")
parser.add_argument("--use_gmm", action='store_true', help="use GMM prior (True) or N(0,1) (False)")
parser.add_argument("--micro", default=0, type=int, help="[0, 1, 2, 3]")
parser.add_argument("--n_flow", default=48, type=int, help="number of flows in RealNVP")
parser.add_argument("--n_epoch", default=40000, type=int, help="number of epochs for training RealNVP")
parser.add_argument("--lr_init", default=1e-3, type=float, help="init. learning rate")
parser.add_argument("--lr_end", default=1e-10, type=float, help="end learning rate")
parser.add_argument("--clip", default=1e-5, type=float, help="gradient clip for neural network training")
parser.add_argument("--n_samples", default=256, type=int, help="minibatch size")
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
state_dict = torch.load('vae_'+str(args.vae_input)+'d_'+str(args.vae_latent)+'_beta' + str(args.beta)+'.pth', map_location=device)
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
generator.eval()
model.eval()
likelihood.eval()

output = output.detach().cpu().numpy()
pcstot = pcstot.detach().cpu().numpy()
ztot = ztot.detach().cpu().numpy()

clr_dict = {
    16 : 'red',
    1373 : 'lime',
    1556 : 'aqua'
    }   

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
font = {'family' : 'serif','weight' : 'normal','size'   : 20}
plt.rc('font', **font)
plt.rc('font', family='serif')
fig, axes = plt.subplots(1,2, figsize=(20, 8), sharex=False)
axes[0].scatter(output[:,0],output[:,1],s=10,color='black')
axes[1].scatter(output[:,1],output[:,2],s=10,color='black')

for i in range(len(microindx_array)):
    axes[0].scatter(output[microindx_array[i],0],output[microindx_array[i],1],s=200,
           facecolor=clr_dict[microindx_array[i]],edgecolor='black',label=microindx_array[i]+1)
    axes[1].scatter(output[microindx_array[i],0],output[microindx_array[i],2],s=200,
           facecolor=clr_dict[microindx_array[i]],edgecolor='black',label=microindx_array[i]+1)
axes[0].legend()
axes[0].set_xlabel(r'$k_{11}$ (W/mK)')
axes[0].set_ylabel(r'$k_{22}$ (W/mK)')
axes[1].set_xlabel(r'$k_{11}$ (W/mK)')
axes[1].set_ylabel(r'$k_{33}$ (W/mK)')
plt.savefig('./images/inference/k_ensemble_paper.png', bbox_inches='tight')

plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)
font = {'family' : 'serif','weight' : 'normal','size'   : 20}
plt.rc('font', **font)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.plot(pcstot[:,0],pcstot[:,1],pcstot[:,2],'.',markersize=5,color='black',alpha=0.5)
for j in range(len(microindx_array)):
    ax.plot(pcstot[microindx_array[j],0],pcstot[microindx_array[j],1],pcstot[microindx_array[j],2],
               '.',markersize=30,markeredgecolor='black',color=clr_dict[microindx_array[j]],label=microindx_array[j]+1)
# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.set_xlabel(r'$\alpha_{0}$')
ax.set_ylabel(r'$\alpha_{1}$')
ax.set_zlabel(r'$\alpha_{2}$')
ax.legend()
plt.show()
plt.savefig('./images/inference/pc_ensemble_3d.png', bbox_inches='tight')

# Plot posterior distribution
n_samples=5000
z_sample = torch.randn((n_samples, args.vae_latent)).to(device)
z, logdet = generator.reverse(z_sample)
pcs = vae.decode(z)#.detach().cpu().numpy()    
z = z.detach().cpu().numpy()     

nplot = 8
import corner
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
font = {'family' : 'serif','weight' : 'normal','size'   : 30}
plt.rc('font', **font)
lbl = []
for i in range(nplot):
    lbl.append(r'$z_{'+str(i)+'}$')
fig = corner.corner(z[:,:nplot],
                    labels=lbl,
                    hist_bin_factor=2,
                    smooth=False,
                    truths=ztot[microindx,:nplot])
plt.savefig('./images/inference/corner_z_' + str(microindx) + '_' + str(args.logprior_weight) + '_' + str(args.n_flow) + '.png', bbox_inches='tight') 

nplot = 8
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
font = {'family' : 'serif','weight' : 'normal','size'   : 30}
plt.rc('font', **font)
lbl = []
for i in range(nplot):
    lbl.append(r'$\alpha_{'+str(i)+'}$')
fig = corner.corner(pcs[:,:nplot].detach().cpu().numpy(),
                    labels=lbl,
                    hist_bin_factor=2,
                    smooth=False,
                    color = "tab:blue",
                    truths=pcstot[microindx,:nplot],
                    truth_color='tab:red')
corner.overplot_points(fig, pcstot[:,:nplot], color="black", alpha=0.02, markersize = 3)

# Extract the axes
axes = np.array(fig.axes).reshape((nplot, nplot))

# Loop over the diagonal
for i in range(nplot):
    ax = axes[i, i]
    ax.set_xlim([pcstot[:,i].min(),pcstot[:,i].max()])

# Loop over the histograms
for yi in range(nplot):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.set_xlim([pcstot[:,xi].min(),pcstot[:,xi].max()])
        ax.set_ylim([pcstot[:,yi].min(),pcstot[:,yi].max()])

plt.savefig('./images/inference/corner_pcs_' + str(microindx) + '_' + str(args.logprior_weight) + '_' + str(args.n_flow) + '.png', bbox_inches='tight') 

# Plot passes through forward model alongside conditioning experimental results
Ymodel = torch.zeros((5000,3))
Kmodel = torch.zeros((5000,3))
for j in range(5000):
    pcsvec = pcs[j,:args.pcs_gp][None,...]
    pcsvec = 2 * (pcsvec - pcstot_min)/(pcstot_max - pcstot_min) - 1
    Ymodel[j,:], Kmodel[j,:] = predictK(pcsvec,device)
    if (j + 1) % 100 == 0:
        print(j + 1)

Ymodel = Ymodel.detach().cpu().numpy()
Kmodel = Kmodel.detach().cpu().numpy()

lowerbound = Ymodel - 2*Kmodel**0.5
upperbound = Ymodel + 2*Kmodel**0.5

# Identify real microstructures within posterior
pcs = pcs.detach().cpu().numpy()
pcs_mean = np.mean(pcs,axis=0)

# find hull of posterior
args.vae_latent_hull = 5
hull = Delaunay(pcs[:,:args.vae_latent_hull])   
        
def in_hull(p, hull):
    ind = abs((hull.find_simplex(p)>=0).astype('int'))
    return ind

ind_hull = in_hull(pcstot[:,:args.vae_latent_hull],hull).astype('bool')
pcstot_hull = pcstot[ind_hull,:]
k_hull = output[ind_hull,:]

# Location of both point clouds in PC space
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
font = {'family' : 'serif','weight' : 'normal','size'   : 20}
plt.rc('font', **font)
fig, axes = plt.subplots(1,3, figsize=(30, 8), sharex=False)
for i in range(3):
    ax = axes[i]
    ax.scatter(pcstot[:,i],pcstot[:,i+1],s=10,color='tab:gray',label=r'$p(\alpha)$')
    ax.scatter(pcs[:,i],pcs[:,i+1],s=30,color='tab:blue',alpha=0.1,label=r'$p(\alpha|k^{*})$')
    #ax.scatter(pcs_mean[i],pcs_mean[i+1],s=100,color='tab:red',alpha=1.0,label=r'$E[p(\alpha|k^{*})]$')
    ax.scatter(pcstot_hull[:,i],pcstot_hull[:,i+1],s=50,color='black',alpha=1.0,label=r'$\alpha \in H_{0:4}$')
    ax.set_xlabel(r'$\alpha_{'+str(int(i))+'}$')
    ax.set_ylabel(r'$\alpha_{'+str(int(i+1))+'}$')
    
    ax.scatter(pcstot[microindx,i],pcstot[microindx,i+1],s=200,
                   facecolor=clr_dict[microindx],edgecolor='black',label=microindx+1)
axes[0].legend()
plt.savefig('./images/inference/pc_clouds_' + str(microindx) + '_' + str(args.logprior_weight) + '_' + str(args.n_flow) + '.png', bbox_inches='tight') 

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
font = {'family' : 'serif','weight' : 'normal','size'   : 20}
plt.rc('font', **font)
fig, axes = plt.subplots(1,3, figsize=(30, 8), sharex=False)
for i in range(3):
    ax = axes[i]
    ax.hist(Ymodel[:,i], rwidth=0.9, density=True, bins=20,label=r'$E[k_m]$')
    ax.hist(lowerbound[:,i], rwidth=0.9, density=True, bins=20,alpha=0.5,color='tab:gray',label=r'$\pm 2\sigma_{k_m}$')
    ax.hist(upperbound[:,i], rwidth=0.9, density=True, bins=20,alpha=0.5,color='tab:gray')    
    ax.plot([output[microindx,i],output[microindx,i]],[0,5],linewidth=3,color='black',linestyle='--',label=r'$k^*}$',alpha=1)
    
    ax.hist(k_hull[:,i], rwidth=0.9, density=True, bins=20,color='tab:red',alpha=0.5,label=r'$k_{\alpha \in H_{0:4}}$')
    #ax.plot([k_hull[:,i],k_hull[:,i]],[0,0.25],linewidth=3,color='tab:red',label=r'$k_{abq}$',alpha=0.95)
    
    ax.set_xlabel(r'$k_{'+str(int(i+1))+str(int(i+1))+'}$ (W/mK)')
    ax.set_ylabel(r'$p(k_{'+str(int(i+1))+str(int(i+1))+'})$')
    ax.set_ylim([0,4])
axes[0].legend()
plt.savefig('./images/inference/generated_resubmitted_' + str(microindx) + '_' + str(args.logprior_weight) + '_' + str(args.n_flow) + '.png', bbox_inches='tight')

###############################################################################
# Reconstruct spatial correlations
###############################################################################
n_samples=1000
z_sample = torch.randn((n_samples, args.vae_latent)).to(device)
z, logdet = generator.reverse(z_sample)
pcs = vae.decode(z).detach().cpu().numpy()    
z = z.detach().cpu().numpy()   

# Load PCA
pca = load(open(os.getcwd() + '/pca_autocorr_1024.pkl', 'rb'))
x_generated_pca = pca.inverse_transform(pcs)
x_generated_pca_hull = pca.inverse_transform(pcstot_hull)

VoxFinal = 109

def split(x_generated_in):
    x_generated_actow = x_generated_pca[:,:VoxFinal**3]
    x_generated_acmat = x_generated_pca[:,VoxFinal**3:2*VoxFinal**3]/9.8279
    x_generated_acpore = x_generated_pca[:,2*VoxFinal**3:3*VoxFinal**3]/21.5770
    x_generated_cctowpore = x_generated_pca[:,3*VoxFinal**3:4*VoxFinal**3]/8.4389
    
    return x_generated_actow, x_generated_acmat, x_generated_acpore, x_generated_cctowpore
    
def const2pt(x_generated, shape):
    x_rve = np.zeros((shape,VoxFinal,VoxFinal,VoxFinal))
    for i in range(shape):
        x_rve[i,:,:,:] = np.reshape(x_generated[i,:],(1,VoxFinal,VoxFinal,VoxFinal),order="F")
    
    return x_rve

x_generated_actow, x_generated_acmat, x_generated_acpore, x_generated_cctowpore = split(x_generated_pca)
x_generated_hull_actow, x_generated_hull_acmat, x_generated_hull_acpore, x_generated_hull_cctowpore = split(x_generated_pca_hull)

x_rve_actow = const2pt(x_generated_actow,pcs.shape[0])
x_rve_acmat = const2pt(x_generated_acmat,pcs.shape[0])
x_rve_acpore = const2pt(x_generated_acpore,pcs.shape[0])
x_rve_cctowpore = const2pt(x_generated_cctowpore,pcs.shape[0])

x_rve_hull_actow = const2pt(x_generated_hull_actow,pcs.shape[0])
x_rve_hull_acmat = const2pt(x_generated_hull_acmat,pcs.shape[0])
x_rve_hull_acpore = const2pt(x_generated_hull_acpore,pcs.shape[0])
x_rve_hull_cctowpore = const2pt(x_generated_hull_cctowpore,pcs.shape[0])

dimVec = np.arange(-54,54+1,1)
mid = 54

plt.rc('xtick',labelsize=30)
plt.rc('ytick',labelsize=30)
font = {'family' : 'serif','weight' : 'normal','size'   : 30}
plt.rc('font', **font)
c = 0
fig, axes = plt.subplots(5,5, figsize=(30, 20), sharex=False)
for i in range(5):
    for j in range(5):
        ax = axes[i,j]
        im = ax.imshow(x_rve_actow[c,:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
        #fig.colorbar(im, ax=ax) 
        ax.axis('off')
        c += 1
plt.tight_layout() 
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.savefig('./images/inference/generated_autocorrelations5x5_' + str(microindx) + '_' + str(args.logprior_weight) + '_' + str(args.n_flow) + '.png', bbox_inches='tight') 

c = 20
fig, axes = plt.subplots(3,3, figsize=(30, 20), sharex=False)
for i in range(3):
    for j in range(3):
        ax = axes[i,j]
        im = ax.imshow(x_rve_actow[c,:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
        #fig.colorbar(im, ax=ax) 
        ax.axis('off')
        c += 1
plt.tight_layout()
fig.subplots_adjust(right=0.8) 
cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])  
fig.colorbar(im, cax=cbar_ax)     
plt.savefig('./images/inference/generated_autocorrelations3x3_' + str(microindx) + '_' + str(args.logprior_weight) + '_' + str(args.n_flow) + '.png', bbox_inches='tight') 

# Plot mean and std of XY plane
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
font = {'family' : 'serif','weight' : 'normal','size'   : 30}
plt.rc('font', **font)
fig, axes = plt.subplots(2,4, figsize=(35, 15), sharex=False)
mean_corr_tow = np.mean(x_rve_actow,axis=0)
mean_corr_mat = np.mean(x_rve_acmat,axis=0)
mean_corr_pore = np.mean(x_rve_acpore,axis=0)
mean_corr_towpore = np.mean(x_rve_cctowpore,axis=0)
std_corr_tow = np.var(x_rve_actow,axis=0)
std_corr_mat = np.var(x_rve_acmat,axis=0)
std_corr_pore = np.var(x_rve_acpore,axis=0)
std_corr_towpore = np.var(x_rve_cctowpore,axis=0)

mean_stack = np.array((mean_corr_tow,mean_corr_mat,mean_corr_pore,mean_corr_towpore))
print(mean_stack.shape)
np.save(str(microindx) + '_2pt_mean.npy',mean_stack)
std_stack = np.array((std_corr_tow,std_corr_mat,std_corr_pore,std_corr_towpore))
np.save(str(microindx) + '_2pt_std.npy',std_stack)

im = axes[0,0].imshow(mean_corr_tow[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
#im = axes[0,0].contourf(dimVec,dimVec,mean_corr_tow[:,:,mid],cmap='inferno')
fig.colorbar(im, ax=axes[0,0])
axes[0,0].set_title(r'$E[f^{00}]$')
axes[0,0].axis('off')

im = axes[1,0].imshow(std_corr_tow[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
#im = axes[1,0].contourf(dimVec,dimVec,std_corr_tow[:,:,mid],cmap='inferno')
fig.colorbar(im, ax=axes[1,0])
axes[1,0].set_title(r'$var[f^{00}]$')
axes[1,0].axis('off')

im = axes[0,1].imshow(mean_corr_mat[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
#im = axes[0,1].contourf(dimVec,dimVec,mean_corr_mat[:,:,mid],cmap='inferno')
fig.colorbar(im, ax=axes[0,1])
axes[0,1].set_title(r'$E[f^{11}]$')
axes[0,1].axis('off')

im = axes[1,1].imshow(std_corr_mat[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
#im = axes[1,1].contourf(dimVec,dimVec,std_corr_mat[:,:,mid],cmap='inferno')
fig.colorbar(im, ax=axes[1,1])
axes[1,1].set_title(r'$var[f^{11}]$')
axes[1,1].axis('off')

im = axes[0,2].imshow(mean_corr_pore[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
#im = axes[0,2].contourf(dimVec,dimVec,mean_corr_pore[:,:,mid],cmap='inferno')
fig.colorbar(im, ax=axes[0,2])
axes[0,2].set_title(r'$E[f^{22}]$')
axes[0,2].axis('off')

im = axes[1,2].imshow(std_corr_pore[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
#im = axes[1,2].contourf(dimVec,dimVec,std_corr_pore[:,:,mid],cmap='inferno')
fig.colorbar(im, ax=axes[1,2])
axes[1,2].set_title(r'$var[f^{22}]$')
axes[1,2].axis('off')

im = axes[0,3].imshow(mean_corr_towpore[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
#im = axes[0,3].contourf(dimVec,dimVec,mean_corr_pore[:,:,mid],cmap='inferno')
fig.colorbar(im, ax=axes[0,3])
axes[0,3].set_title(r'$E[f^{02}]$')
axes[0,3].axis('off')

im = axes[1,3].imshow(std_corr_towpore[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
#im = axes[1,3].contourf(dimVec,dimVec,std_corr_pore[:,:,mid],cmap='inferno')
fig.colorbar(im, ax=axes[1,3])
axes[1,3].set_title(r'$var[f^{02}]$')
axes[1,3].axis('off')
plt.tight_layout()
plt.savefig('./images/inference/generated_std_mean_' + str(microindx) + '_' + str(args.logprior_weight) + '_' + str(args.n_flow) + '.png', bbox_inches='tight') 


# Plot line slices through center of XY plane
pca_mat = np.load('pca_indx.npy')

micro_dict = {
    1556 : 0,
    2032 : 1,
    2004 : 2,
    16 : 3,
    1373 : 4,   
    }     

pca_matrix_actow, pca_matrix_acmat, pca_matrix_acpore, pca_matrix_cctowpore = split(pca_mat)

pca_rve_actow = const2pt(pca_matrix_actow, pca_mat.shape[0])
pca_rve_acmat = const2pt(pca_matrix_acmat, pca_mat.shape[0])
pca_rve_acpore = const2pt(pca_matrix_acpore, pca_mat.shape[0])
pca_rve_cctowpore = const2pt(pca_matrix_cctowpore, pca_mat.shape[0])

plt.rc('xtick',labelsize=22)
plt.rc('ytick',labelsize=22)
font = {'family' : 'serif','weight' : 'normal','size'   : 22}
plt.rc('font', **font)
fig, axes = plt.subplots(1,4, figsize=(25, 5.5), sharex=False)
for i in range(n_samples):
    axes[0].plot(dimVec,x_rve_actow[i,:,mid,mid],color='tab:blue',alpha=0.025)
    axes[1].plot(dimVec,x_rve_acmat[i,:,mid,mid],color='tab:blue',alpha=0.025)
    axes[2].plot(dimVec,x_rve_acpore[i,:,mid,mid],color='tab:blue',alpha=0.025)
    axes[3].plot(dimVec,x_rve_cctowpore[i,:,mid,mid],color='tab:blue',alpha=0.025)

axes[0].plot(dimVec,x_rve_actow[i,:,mid,mid],color='tab:blue',alpha=0.2,label=r'$f^{\alpha \beta}$')

#for i in range(pcstot_hull.shape[0]):
#    axes[0].plot(dimVec,x_rve_hull_actow[i,:,mid,mid],color='tab:gray',linestyle='--',alpha=0.25)
#    axes[1].plot(dimVec,x_rve_hull_acmat[i,:,mid,mid],color='tab:gray',linestyle='--',alpha=0.25)
#    axes[2].plot(dimVec,x_rve_hull_acpore[i,:,mid,mid],color='tab:gray',linestyle='--',alpha=0.25)
#    axes[3].plot(dimVec,x_rve_hull_cctowpore[i,:,mid,mid],color='tab:gray',linestyle='--',alpha=0.25)  

#axes[0].plot(dimVec,x_rve_hull_actow[i,:,mid,mid],color='tab:gray',linestyle='--',alpha=0.25,label=r'$f^{\alpha \beta} \in H_{0:4}$')    
    
axes[0].plot(dimVec,mean_corr_tow[:,mid,mid],linestyle='--',color='black',linewidth=3,label=r'$E[f^{\alpha \beta}]$')
axes[1].plot(dimVec,mean_corr_mat[:,mid,mid],linestyle='--',color='black',linewidth=3)
axes[2].plot(dimVec,mean_corr_pore[:,mid,mid],linestyle='--',color='black',linewidth=3)
axes[3].plot(dimVec,mean_corr_towpore[:,mid,mid],linestyle='--',color='black',linewidth=3)
axes[0].plot(dimVec,pca_rve_actow[micro_dict[microindx],:,mid,mid],color='tab:red',linewidth=3,label=r'$f^{\alpha \beta}_{*}$')
axes[1].plot(dimVec,pca_rve_acmat[micro_dict[microindx],:,mid,mid],color='tab:red',linewidth=3)
axes[2].plot(dimVec,pca_rve_acpore[micro_dict[microindx],:,mid,mid],color='tab:red',linewidth=3)
axes[3].plot(dimVec,pca_rve_cctowpore[micro_dict[microindx],:,mid,mid],color='tab:red',linewidth=3)

axes[0].set_xlabel(r'$r_x$')
axes[1].set_xlabel(r'$r_x$')
axes[2].set_xlabel(r'$r_x$')
axes[3].set_xlabel(r'$r_x$') 
axes[0].set_ylabel(r'$f^{00}$')
axes[1].set_ylabel(r'$f^{11}$')
axes[2].set_ylabel(r'$f^{22}$')
axes[3].set_ylabel(r'$f^{02}$')  
#axes[0].legend()
#axes[0].legend(handles=axes[0],bbox_to_anchor=(0, 1.12), loc='lower left')
#fig.tight_layout(pad=3.0)
fig.tight_layout()
#fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
fig.legend(bbox_to_anchor=(0.5,1.2), ncol=4, loc='upper center', fontsize=22)

plt.savefig('./images/inference/generated_xy_line_' + str(microindx) + '_' + str(args.logprior_weight) + '_' + str(args.n_flow) + '.png', bbox_inches='tight') 

# plot paper X-Y 2D & 1D
plt.rc('xtick',labelsize=22)
plt.rc('ytick',labelsize=22)
font = {'family' : 'serif','weight' : 'normal','size'   : 22}
plt.rc('font', **font)

fig, axes = plt.subplots(3,4, figsize=(25, 14), sharex=False)
for i in range(n_samples):
    axes[0,0].plot(dimVec,x_rve_actow[i,:,mid,mid],color='tab:blue',alpha=0.025)
    axes[0,1].plot(dimVec,x_rve_acmat[i,:,mid,mid],color='tab:blue',alpha=0.025)
    axes[0,2].plot(dimVec,x_rve_acpore[i,:,mid,mid],color='tab:blue',alpha=0.025)
    axes[0,3].plot(dimVec,x_rve_cctowpore[i,:,mid,mid],color='tab:blue',alpha=0.025)

axes[0,0].plot(dimVec,x_rve_actow[i,:,mid,mid],color='tab:blue',alpha=0.2,label=r'$f^{\alpha \beta}$')
axes[0,3].plot(dimVec,x_rve_cctowpore[i,:,mid,mid],color='tab:blue',alpha=0.2,label=r'$f^{\alpha \beta}$')

axes[0,0].plot(dimVec,mean_corr_tow[:,mid,mid],linestyle='--',color='black',linewidth=3,label=r'$E[f^{\alpha \beta}]$')
axes[0,1].plot(dimVec,mean_corr_mat[:,mid,mid],linestyle='--',color='black',linewidth=3)
axes[0,2].plot(dimVec,mean_corr_pore[:,mid,mid],linestyle='--',color='black',linewidth=3)
axes[0,3].plot(dimVec,mean_corr_towpore[:,mid,mid],linestyle='--',color='black',linewidth=3,label=r'$E[f^{\alpha \beta}]$')
axes[0,0].plot(dimVec,pca_rve_actow[micro_dict[microindx],:,mid,mid],color='tab:red',linewidth=3,label=r'$f^{\alpha \beta}_{*}$')
axes[0,1].plot(dimVec,pca_rve_acmat[micro_dict[microindx],:,mid,mid],color='tab:red',linewidth=3)
axes[0,2].plot(dimVec,pca_rve_acpore[micro_dict[microindx],:,mid,mid],color='tab:red',linewidth=3)
axes[0,3].plot(dimVec,pca_rve_cctowpore[micro_dict[microindx],:,mid,mid],color='tab:red',linewidth=3,label=r'$f^{\alpha \beta}_{*}$')

axes[0,0].set_xlabel(r'$r_x$')
axes[0,1].set_xlabel(r'$r_x$')
axes[0,2].set_xlabel(r'$r_x$')
axes[0,3].set_xlabel(r'$r_x$') 
axes[0,0].set_ylabel(r'$f^{00}$')
axes[0,1].set_ylabel(r'$f^{11}$')
axes[0,2].set_ylabel(r'$f^{22}$')
axes[0,3].set_ylabel(r'$f^{02}$') 
axes[0,3].legend() 

im = axes[1,0].imshow(mean_corr_tow[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
axes[1,0].set_title(r'$E[f^{00}]$')
axes[1,0].axis('off')
fig.colorbar(im, ax=axes[1,0],shrink=0.75)

im = axes[2,0].imshow(std_corr_tow[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
fig.colorbar(im, ax=axes[2,0],shrink=0.75)
axes[2,0].set_title(r'$var[f^{00}]$')
axes[2,0].axis('off')

im = axes[1,1].imshow(mean_corr_mat[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
fig.colorbar(im, ax=axes[1,1],shrink=0.75)
axes[1,1].set_title(r'$E[f^{11}]$')
axes[1,1].axis('off')

im = axes[2,1].imshow(std_corr_mat[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
fig.colorbar(im, ax=axes[2,1],shrink=0.75)
axes[2,1].set_title(r'$var[f^{11}]$')
axes[2,1].axis('off')

im = axes[1,2].imshow(mean_corr_pore[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
fig.colorbar(im, ax=axes[1,2],shrink=0.75)
axes[1,2].set_title(r'$E[f^{22}]$')
axes[1,2].axis('off')

im = axes[2,2].imshow(std_corr_pore[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
fig.colorbar(im, ax=axes[2,2],shrink=0.75)
axes[2,2].set_title(r'$var[f^{22}]$')
axes[2,2].axis('off')

im = axes[1,3].imshow(mean_corr_towpore[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
fig.colorbar(im, ax=axes[1,3],shrink=0.75)
axes[1,3].set_title(r'$E[f^{02}]$')
axes[1,3].axis('off')

im = axes[2,3].imshow(std_corr_towpore[:,:,mid],cmap='inferno',extent=[dimVec.min(), dimVec.max(), dimVec.min(), dimVec.max()])
fig.colorbar(im, ax=axes[2,3],shrink=0.75)
axes[2,3].set_title(r'$var[f^{02}]$')
axes[2,3].axis('off')

plt.tight_layout()
plt.savefig('./images/inference/generated_xy_paper_' + str(microindx) + '_' + str(args.logprior_weight) + '_' + str(args.n_flow) + '.png', bbox_inches='tight') 
