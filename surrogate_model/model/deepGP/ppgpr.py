# ppgpr
from .base import DenseNetwork
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from botorch.posteriors.gpytorch import GPyTorchPosterior
import torch

from gpytorch.mlls import PredictiveLogLikelihood 

# Multi-task Variational GP:
# https://docs.gpytorch.ai/en/v1.4.2/examples/04_Variational_and_Approximate_GPs/SVGP_Multitask_GP_Regression.html


class GPModel(ApproximateGP):
    def __init__(self, inducing_points, **kwargs):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0) )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def loss(self, pred, y):
        mll = PredictiveLogLikelihood(self.likelihood, self, num_data=len(y))
        loss = -mll(pred, y.cuda()).mean()
        return loss

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior:
            self.eval()  # make sure model is in eval mode
            # self.model.eval()
            self.likelihood.eval()
            dist = self.likelihood(self(X)) 

            return GPyTorchPosterior(mvn=dist)
            
# gp model with deep kernel
class GPModelDKL(ApproximateGP):
    def __init__(self, inducing_points, likelihood=None, hidden_dims=(64, 64), **kwargs):
        if likelihood is None:
            likelihood=gpytorch.likelihoods.GaussianLikelihood().cuda()
        feature_extractor = DenseNetwork(
            input_dim=inducing_points.size(-1),
            hidden_dims=hidden_dims).to(inducing_points.device
            ) # MLP
        inducing_points = feature_extractor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
            )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.num_outputs = 1 #must be one
        self.likelihood = likelihood
        self.feature_extractor = feature_extractor
    
    
    def loss(self, pred, y):
        mll = PredictiveLogLikelihood(self.likelihood, self, num_data=len(y))
        loss = -mll(pred, y.cuda()).mean()
        return loss

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) # GP

    def __call__(self, x, *args, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, *args, **kwargs)

    def posterior(
            self, X, output_indices=None, observation_noise=False, *args, **kwargs
        ) -> GPyTorchPosterior: # used for acquisition
        self.eval()  # make sure model is in eval mode 
        self.likelihood.eval()
        dist = self.likelihood(self(X))

        return GPyTorchPosterior(mvn=dist)
