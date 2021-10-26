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
from DeepKernel import DeepNonstationarySpatiotemporalKernel, FocusPointsNN
from torch.nn.utils import clip_grad_norm_

class STVA(Mean):
	def __init__(self, n_features):
		"""
		Spatio-temporal Vector Autoregressive Mean Model
		"""
		super().__init__()
		# model configuration
		self.n_features = n_features # number of features for each county
		# parameters
		self.W          = torch.nn.Parameter(torch.randn((n_features), requires_grad=True))

	def forward(self, x_feature):
		"""
		Args:
		- x: spatio-temporal index [ n_batches, 2 ]
		"""
		# calculate mean given the spatio-temporal index
		mean = (x_feature * self.W.unsqueeze(0)).sum(1) # [ n_batches ]
		mean = torch.nn.functional.softplus(mean)       # [ n_batches ]
		return mean

class VanillaGP(ApproximateGP):
	"""
	Gaussian Process Model for COVID-19    
	"""
	def __init__(self, n_features, inducing_x):
		"""
		Args:
		- n_features: number of features for each county
		- inducing_x: inducing point (a subset of X)
		"""
		# posterior distribution for inducing points
		variational_distribution = CholeskyVariationalDistribution(inducing_x.size(0))
		variational_strategy     = VariationalStrategy(self, inducing_x, variational_distribution, learn_inducing_locations=True)
		super(VanillaGP, self).__init__(variational_strategy)
		self.mean_module  = STVA(n_features)
		# TODO: CUSTOMINZED KERNEL
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=torch.tensor([0]))) * \
		    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=torch.tensor([1]))) * \
		    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=torch.tensor([2])))
		# self.covar_module = DeepNonstationarySpatiotemporalKernel(active_dims=torch.tensor([1, 2]))
		# self.covar_module = DeepNonstationarySpatiotemporalKernel()
		# self.covar_module = gpytorch.kernels.ProductStructureKernel(gpytorch.kernels.RBFKernel(), num_dims=1)

	def forward(self, x):
		x_feature = x[:, 4:].clone() # spatio-temporal index
		x_coord   = x[:, :3].clone() # spatio-temporal coordinate
		mean_x    = self.mean_module(x_feature)
		covar_x   = self.covar_module(x_coord)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_test_split(C, D, H, locs, p=2, tau=None):
	'''
	Function to create a training data set (<=t) and testing data set

	Parameters
	----------
	- C :    Tensor of cases data      [T, n_weeks]
	- D :    Tensor of deaths data     [T, n_weeks]
	- H :    Tensor of hotspot data    [T, n_weeks]
	- locs : Tensor of locations data  [n_counties, 2]
	- p :    Number of lags we consider
	- t:     Time window for training data (if t is None, t = T)
 
	Returns
	-------
	- train_x : Tensor of train data x [ t * n_counties, 4 + n_features ]
	- train_y : Tensor of train data y [ t * n_counties ]
	- train_hotspot : Tensor of train hotspot data [ t * n_counties ]
	- test_x :  Tensor of test data x  [ (T-t-p) * n_counties, 4 + n_features ]
	- test_y :  Tensor of test data y  [ (T-t-p) * n_counties ]
	- test_hotspot :  Tensor of test hotspot data [ (T-t-p) * n_counties ]
	'''
	# Number of features
	n_features    = 2 * p
	T, n_counties = C.shape

	tau = T - p if tau is None else tau

	# Y: number of cases
	y = C[p:, :]                                    # [ T-p, n_counties ]
	hotspot = H[p:, :]
	# X: spatio-temporal coordinates and features
	X = np.zeros((T-p, n_counties, 4 + n_features)) # [ T-p, n_counties, 4 + n_features]
	# prepare X
	for t in range(T-p):
		X[t, :, 0]   = t + p                        # time
		X[t, :, 1:3] = locs                         # geolocation
		X[t, :, 3]   = np.arange(n_counties)        # location index
		feature      = np.concatenate((
			C[t:t+p, :].transpose(), 
			D[t:t+p, :].transpose()), axis=1)       # [ n_counties, 2p]
		X[t, :, 4:]  = feature

	train_X, train_y, train_hotspot = X[:tau, :, :], y[:tau, :], hotspot[:tau, :]
	train_y = torch.FloatTensor(train_y).view(tau * n_counties)                         # [ tau * n_counties ]
	train_hotspot = torch.FloatTensor(train_hotspot).view(tau * n_counties)             # [ tau * n_counties ]
	train_X = torch.FloatTensor(train_X).view(tau * n_counties, 4 + n_features)         # [ tau * n_counties, 4 + n_features ]
	
	if tau < T - p:
		test_X, test_y, test_hotspot = X[tau:, :, :], y[tau:, :], hotspot[tau:, :]
		test_y = torch.FloatTensor(test_y).view((T-p-tau) * n_counties)                 # [ T-p-tau * n_counties ]
		test_hotspot = torch.FloatTensor(test_hotspot).view((T-p-tau) * n_counties)                 # [ T-p-tau * n_counties ]
		test_X = torch.FloatTensor(test_X).view((T-p-tau) * n_counties, 4 + n_features) # [ T-p-tau * n_counties, 4 + n_features ]
	else: 
		test_X, test_y, test_hotspot = None, None, None
	
	return train_X, train_y, train_hotspot, test_X, test_y, test_hotspot

def train_MLE(
	model, likelihood, train_loader, n, 
	num_epochs = 10,
	ngd_lr     = 1e-7, 
	adam_lr    = 1e-1,
	print_iter = 100,
	batch_size = 500,
	modelname  = "STVAmean-RBFkernel"):
	"""
	Train GP model by MLE
	"""
	# Set training mode
	model.train()
	likelihood.train()

	# NGD optimizer for variational parameters
	variational_ngd_optimizer = gpytorch.optim.NGD(
		model.variational_parameters(),
		num_data=n, lr=ngd_lr)

	# Adam optimizer for hyperparameters
	adam_optimizer = torch.optim.Adam([
		{'params': model.hyperparameters()},
		{'params': likelihood.parameters()},
	], lr=adam_lr)

	# Loss function - Variational ELBO
	# mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)
	mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=n)

	# Training
	for i in range(num_epochs):
		# Within each iteration, we will go over each minibatch of data
		for j, data in enumerate(train_loader):
			if torch.isnan(model.variational_strategy.inducing_points).any():
				print("Bad variational points")
			# with torch.autograd.detect_anomaly():
				# Alternate between NGD and Adam optimization
			x_batch, y_batch = data
			# adam optimizer
			adam_optimizer.zero_grad()
			output = model(x_batch)
			loss   = -mll(output, y_batch)
			loss.backward()
			adam_optimizer.step()
			# ngd optimizer
			variational_ngd_optimizer.zero_grad()
			output = model(x_batch)
			loss   = -mll(output, y_batch)
			loss.backward()
			variational_ngd_optimizer.step()
			if j % print_iter == 0:
				print("[%s] Epoch : %d,\titer : %d,\tloss : %.5e" % (arrow.now(), i, j, loss / print_iter))
				torch.save(model.state_dict(), "saved_models/%s.pth" % modelname)


	return model, likelihood


def GP_prediction(model, gaussian_likelihood, data_loader, sigma=3):
	"""
	Prediction using GP models
	"""
	model.eval()
	gaussian_likelihood.eval()
	with torch.no_grad():
		# mean
		means = [ 
			gaussian_likelihood(model(x_batch)).mean.detach().numpy() # [ n_batches ] 
			for x_batch, y_batch in data_loader ]
		means = np.concatenate(means, axis=0)                # [ n ]
		# confidence interval
		lowerbs, upperbs = [], []
		for x_batch, y_batch in data_loader:
			std  = gaussian_likelihood(model(x_batch)).stddev.mul_(sigma)
			mean = gaussian_likelihood(model(x_batch)).mean
			lowerb, upperb = mean.sub(std), mean.add(std)    # 2 * [ n_batches ]
			lowerbs.append(lowerb.detach().numpy())
			upperbs.append(upperb.detach().numpy())
		lowerbs = np.concatenate(lowerbs, axis=0)            # [ n ] 
		upperbs = np.concatenate(upperbs, axis=0)            # [ n ] 
		return means, lowerbs, upperbs

def plot_nationwide_prediction(true, mean, lowerb, upperb, filename="Prediction"):
	"""
	Plot prediction trajectory against ground truth. 
	"""
	fig, ax = plt.subplots(figsize=(8, 5))

	time = np.arange(true.shape[0])
	ax.plot(time, true, linewidth=3, linestyle="--", color="gray", alpha=1, label="True")
	ax.plot(time, mean, linewidth=3, linestyle="-", color="blue", alpha=.7, label="Prediction")
	ax.fill_between(time, lowerb, upperb, facecolor='blue', alpha=0.2, label="Prediction CI")

	plt.xlabel("Week index")
	plt.ylabel("Number of cases")
	plt.legend(fontsize=15, loc='upper left')
	plt.title(filename)
	fig.tight_layout()      # otherwise the right y-label is slightly clipped
	plt.show()


#------------------------------------------------------
#
# Code for custom Kernel
#
#------------------------------------------------------



def kernel_viz(locs,old_locs, model):
	net = model.covar_module.NN4
	locs = torch.Tensor(locs)
	old_locs = torch.Tensor(locs)
	focus_points = net(locs)[:, :2]
	weights = net(locs)[:, -1]
	minv, maxv = float(min(weights)), float(max(weights))
	
	old_locs = torch.stack([old_locs[:, 1], old_locs[:, 0]], axis=-1)
	
	ps1 = old_locs + focus_points * 3 # * 5e-1
	ps2 = old_locs # * 5e-1

	# print(minv, maxv)
	# print("!!!!")
	colorscale = branca.colormap.linear.Blues_09.scale(minv, maxv)
	m = folium.Map(
		location=[38, -95],
		tiles='cartodbpositron',
		zoom_start=4
		)

	def style_function(feature):        
		county = int(feature['id'][-5:])
		
		try:
			data=weights[counties.index(str(county))]
		except Exception as e:
			data = 0.45
		return {
			'fillOpacity': 0.5,
			'weight': 0,
			'fillColor': '#black' if data is None else colorscale(data)
		}
	
	
	print(colorscale)
	colorscale.caption = "Weighting"
	colorscale.add_to(m)

	folium.TopoJson(
	uscountygeo,
	'objects.us_counties_20m',
	style_function=style_function
	).add_to(m)

	folium.Choropleth(geo_data=uscountygeo,
	   topojson='objects.us_counties_20m',
	   line_weight=0.1,
	   fill_opacity=0.0).add_to(m)

	folium.Choropleth(
	geo_data=state_json,
	topojson='objects.states',
	line_weight=0.15,
	fill_opacity=0.0
	).add_to(m)
	# colorscale.add_to(m)
	# m.save("tester.html")

	for i in range(3144):
		coords = [ps1[i, :], ps2[i, :]]
		# print(coords)
		folium.PolyLine(
			locations=coords,
			weight=1,
			color = 'red').add_to(m)
	colorscale.add_to(m)
	m.save("Kernel4.html")

def county_kernel_viz(locs, old_locs, model):
	cov = model.covar_module
	locs = torch.Tensor(locs)
	kernel = cov.spatial_kernel(locs, locs).detach().numpy()
	maxk = 0
	print(kernel.shape)
	fulton_id = I.index('13121')
	brooks_id = I.index('36005')
	la_id = I.index("6037")
	chi_id = I.index('17031')
	minv = min(kernel[fulton_id])
	maxv = max(kernel[fulton_id])
	print(minv, maxv)
	colorscale = branca.colormap.linear.Reds_09.scale(-12 + 12, (np.log(maxv) + 12) / 40)
	m = folium.Map(
		location=[38, -95],
		tiles='cartodbpositron',
		zoom_start=4
		)

	def style_function(feature):        
		county = int(feature['id'][-5:])
		
		try:
			data=(np.log(kernel[fulton_id][counties.index(str(county))]) + 12)/40
		except Exception as e:
			data = 0.0
		return {
			'fillOpacity': 0.5,
			'weight': 0,
			'fillColor': '#black' if data is None else colorscale(data)
		}
	
	
	print(colorscale)
	colorscale.caption = "Kernel Value"
	colorscale.add_to(m)

	folium.TopoJson(
	uscountygeo,
	'objects.us_counties_20m',
	style_function=style_function
	).add_to(m)

	folium.Choropleth(geo_data=uscountygeo,
	   topojson='objects.us_counties_20m',
	   line_weight=0.1,
	   fill_opacity=0.0).add_to(m)

	folium.Choropleth(
	geo_data=state_json,
	topojson='objects.states',
	line_weight=0.15,
	fill_opacity=0.0
	).add_to(m)

	m.save("kernel/Atlanta.html")
	
	# print(kernel.numpy())

def rbf_viz(kernel, locs):
	locs = torch.Tensor(locs)
	maxk = 0
	print(kernel.shape)
	fulton_id = I.index('13121')
	brooks_id = I.index('36005')
	la_id = I.index("6037")
	chi_id = I.index('17031')
	minv = min(kernel[la_id])
	maxv = max(kernel[la_id])
	print(minv, maxv)
	colorscale = branca.colormap.linear.Reds_09.scale(minv, maxv)
	m = folium.Map(
		location=[38, -95],
		tiles='cartodbpositron',
		zoom_start=4
		)

	def style_function(feature):        
		county = int(feature['id'][-5:])
		
		try:
			data=kernel[la_id][counties.index(str(county))]
		except Exception as e:
			data = 0.0
		return {
			'fillOpacity': 0.5,
			'weight': 0,
			'fillColor': '#black' if data is None else colorscale(data)
		}
	
	
	print(colorscale)
	colorscale.caption = "Kernel Value"
	colorscale.add_to(m)

	folium.TopoJson(
	uscountygeo,
	'objects.us_counties_20m',
	style_function=style_function
	).add_to(m)

	folium.Choropleth(geo_data=uscountygeo,
	   topojson='objects.us_counties_20m',
	   line_weight=0.1,
	   fill_opacity=0.0).add_to(m)

	folium.Choropleth(
	geo_data=state_json,
	topojson='objects.states',
	line_weight=0.15,
	fill_opacity=0.0
	).add_to(m)

	m.save("kernel/rbf/LA.html")
		
if __name__ == "__main__":

	#--------------------------------------------------------------------------
	#
	# LOAD DATA MATRICES
	#
	#--------------------------------------------------------------------------


	# confirmed cases and deaths
	C = np.load("data/mat/ConfirmedCases_1-17.npy") # [ T, n_counties ]
	D = np.load("data/mat/death_1-17.npy") 
	H = np.load("data/mat/hotspot_1-17.npy")       # [ T, n_counties ]

	# Load covariates
	M      = np.load("data/mat/mobility_1-17.npy").transpose([2,0,1]) # [ n_mobility, T, n_counties ]
	pop    = np.load("data/mat/population.npy")
	over60 = np.load("data/mat/over60.npy")
	cov    = np.array([pop, over60])                                                # [ n_covariates, T, n_counties ]

	T, n_counties = C.shape # 3144
	n_mobility    = M.shape[0]
	n_covariates  = cov.shape[0]

	#--------------------------------------------------------------------------
	#
	# LOAD META DATA AND CONFIGURATIONS
	#
	#--------------------------------------------------------------------------

	# Distance matrix for counties
	distance = np.sqrt(np.load("data/mat/distance.npy"))  # [ n_counties, n_counties ]
	adj      = np.load("data/mat/adjacency_matrix.npy")   # [ n_counties, n_counties ]
	# FIPS for US counties
	I        = np.load("data/mat/counties.npy").tolist()
	loc_dict = {}
	# Raw file for US counties
	with open('data/meta/county_centers.csv', newline='') as csvfile:
		locsreader = list(csv.reader(csvfile, delimiter=','))
		for row in locsreader[1:]:
			if row[1] != "NA":
				fips, lon, lat = int(row[0]), float(row[1]), float(row[2])
				loc_dict[fips] = [lon, lat]
			else:
				print(row)
	# Geolocation (longitude and latitude) of US counties
	locs = np.array([ loc_dict[int(i)] for i in I ]) # [ n_counties, 2 ]
	old_locs = locs.copy()
	locs = locs - np.mean(locs, axis = 0)
	locs = locs / np.linalg.norm(locs, axis=0) * 100

	with open (r"data/meta/states-10m.json", "r") as f:
		state_json = json.load(f)
	with open(r"data/meta/us_counties_20m_topo.json", "r") as f:
		uscountygeo = json.load(f)

	counties = list(np.load("data/mat/counties.npy"))		
	#---------------------------------------------------------------
	#
	# TRAINING DATA PREPARATION
	#
	#---------------------------------------------------------------

	modelname  = "STVAmean-RBFkernel-insample"
	p          = 2
	n_features = 2 * p
	batch_size = 786

	# Training data loader
	train_X, train_y, train_hotspot, _, _, _ = train_test_split(C, D, H, locs, p=p, tau=None)
	n = train_y.shape[0] # Number of data points
	train_dataset = TensorDataset(train_X, train_y)
	train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	# Inducing points for variational inference
	init_inducing_idx = random.sample(list(np.arange(n)), batch_size) # randomly initialized inducing point indices
	inducing_x        = train_X[init_inducing_idx, :]          # randomly selected inducing points

	# Define models
	model      = VanillaGP(n_features, inducing_x=inducing_x)
	gaussian_likelihood = gpytorch.likelihoods.GaussianLikelihood()

	# Load model
	model.load_state_dict(torch.load("saved_models/%s.pth" % modelname))
	model, _ = train_MLE(
	model, gaussian_likelihood, train_loader, n, 
	num_epochs = 25,
	ngd_lr     = 1e-2,
	adam_lr    = 1e-4,
	print_iter = 10,
	batch_size = batch_size,
	modelname  = modelname)

	kernel_data = train_X[-3144:]
	x = model(kernel_data)
	kernel = x.lazy_covariance_matrix.numpy()
	rbf_viz(kernel, locs)



	
