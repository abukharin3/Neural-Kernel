import math
import torch
from torch.nn import Module
import torch.nn.functional as F
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
from gpytorch.lazy import MatmulLazyTensor, RootLazyTensor
from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode
from gpytorch.kernels.kernel import Kernel
import csv
import torch
import numpy as np
import json
import branca
import folium
import arrow
import gpytorch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.means.mean import Mean                                            
import random



np.seterr(all='raise')



class FocusPointsNN(torch.nn.Module):
	'''
	Class to implement the neural network for focus points
	'''
	def __init__(self):
		super(FocusPointsNN, self).__init__()
		
		# self.fc1 = torch.nn.Linear(2, 32)
		# self.fc2 = torch.nn.Linear(32, 32)
		# self.fc3 = torch.nn.Linear(32, 3)
		self.fc1 = torch.nn.Linear(2, 32)
		self.fc2 = torch.nn.Linear(32, 32)
		self.fc3 = torch.nn.Linear(32, 32)
		self.fc4 = torch.nn.Linear(32, 3)
		
	def forward(self, x):
		'''
		Forward function of the neural network

		Parameters
		----------
		- x : spatial coordinates [batch_size, 2]

		Returns
		-------
		- x : psi and weights [batch_size, 3]
		'''
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		x = self.fc3(x)
		x = F.relu(x)
		x = self.fc4(x)#[:, :2]
		psi = (torch.sigmoid(x[:, :2]) - 0.5) * 2
		w = torch.sigmoid(x[:, -1]).unsqueeze(-1)
		x = torch.cat([psi, w], dim=-1)
		# x = torch.sigmoid(x)
		
		return x

class DeepNonstationarySpatiotemporalKernel(Kernel):
	r"""
	Computes a covariance matrix based on the deep nonstationary spatio-tempral kernel
	between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:
	
	"""

	def __init__(self):
		super(DeepNonstationarySpatiotemporalKernel, self).__init__()
		self.A = 0.02 # controls width
		self.lam = 5 # controls size
		self.sigma = torch.nn.Parameter(torch.ones(1))
		
		self.NN1 = FocusPointsNN()
		self.NN2 = FocusPointsNN()
		self.NN3 = FocusPointsNN()
		self.NN4 = FocusPointsNN()

	has_lengthscale = False


	def spatial_kernel(self, s1, s2, **params):
		'''
		Function to implement the spatial kernel

		Parameters
		----------
		- s1 : first tensor of spatial coordinates [batch_size, 2]
		- s2 : first tensor of spatial coordinates [batch_size, 2]
		- params : gpytorch parameters

		Returns
		-------
		- spatial_kernel [batch_size, batch_size]
		'''
		s1 = s1 - torch.mean(s1, dim=0)
		s2 = s2 - torch.mean(s2, dim=0) # [batch_size, 2]
		s1 = torch.stack([s1[:,1], s1[:,0]]).T
		s2 = torch.stack([s2[:,1], s2[:,0]]).T
		# First Kernel
		psi_normal = self.NN1(s1) # [batch_size, 3]
		psi_prime = self.NN1(s2) # [batch_size, 3]
		w1 = psi_prime[:, -1]
		sigma_normal = self.get_sigma(psi_normal)
		sigma_prime = self.get_sigma(psi_prime)

		sigma = sigma_normal + sigma_prime
		sigma_inv = torch.linalg.inv(sigma) # [batch_size, 2, 2]

		l = torch.cholesky(sigma_inv) # [batch_size, 2, 2], [batch_size, 2, 1]
		s1_ = torch.bmm(l, s1.unsqueeze(-1)).squeeze()
		s2_ = torch.bmm(l, s2.unsqueeze(-1)).squeeze()
		spatial_diff = self.covar_dist(s1_, s2_, **params, square_dist=True)
		kernel1 = torch.pow(torch.norm(sigma), -0.5) *  torch.exp(-0.5 * spatial_diff) / (2 * np.pi)
		
		return kernel1

		

	def get_sigma(self, psi):
		'''
		Helper function to build sigma from psi

		Attributes
		----------
		- psi : psi from neural network [batch_size, 2]

		Returns
		-------
		- Sigma : [batch_size, 2, 2]
		'''
		psi_x = psi[:, 0]
		psi_y = psi[:, 1]

		alpha = torch.atan(psi_y / (psi_x + 1e-1 * torch.sign(psi_x) + 1e-3))

		Q = torch.sqrt(4 * self.A ** 2 + torch.pow(torch.linalg.norm(psi[:, :2], ord = 2, dim=1), 4) * np.pi ** 2) / (2 * np.pi)

		sigma_11 = Q + torch.pow(torch.linalg.norm(psi[:, :2], ord = 2, dim=1), 2) / 2
		sigma_22 = Q - torch.pow(torch.linalg.norm(psi[:, :2], ord = 2, dim=1), 2) / 2
		sigma_12 = self.A ** 2 / (np.pi ** 2) * torch.cos(alpha)
		sigma_21 = self.A ** 2 / (np.pi ** 2) * torch.cos(alpha)

		left_sigma = torch.stack([sigma_11, sigma_21], dim =1)
		right_sigma = torch.stack([sigma_12, sigma_22], dim =1)
		sigma = torch.stack([left_sigma, right_sigma], dim=2) * self.lam ** 2


		return sigma

	def forward(self, x1, x2, **params):



		s1 = x1[:, 1:]
		s2 = x2[:, 1:]

		t1 = x1[:, 0].unsqueeze(dim=-1)
		t2 = x2[:, 0].unsqueeze(dim=-1)

		spatial_kernel = self.spatial_kernel(s1, s2, **params)

		time_kernel = torch.exp(-1/(2 * self.sigma ** 2) * self.covar_dist(t1, t2, **params, square_dist=True))

		return spatial_kernel * time_kernel