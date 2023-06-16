import torch
import numpy as np
import torch.nn.functional as F
import torchvision
import gpytorch
torch.set_default_dtype(torch.float32)

class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_latents, num_tasks, num_inducing, input_dims):
        
        inducing_points = torch.randn(num_latents, num_inducing, input_dims)
        #inducing_points = torch.randn(num_latents, num_inducing)
        
        batch_shape = torch.Size([num_latents])

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=batch_shape, mean_init_std = 1e-6,
        )

        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,
                            batch_shape=batch_shape, ard_num_dims=input_dims)

        #self.covar_module = gpytorch.kernels.ScaleKernel(
        #    gpytorch.kernels.RBFKernel(batch_shape=batch_shape,ard_num_dims=input_dims),
        #    batch_shape=batch_shape, ard_num_dims=None
        #)     

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)   