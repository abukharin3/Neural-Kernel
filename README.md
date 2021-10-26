# Deep Neural Kernel for Gaussian Process Regression Classification
A set of Python tools to for parametrizing the kernel in a Gaussian Process by neural networks.

### Usage
There are two vital files included in this repo 
- DeepKernel.py contains the kernel parametrized by neural networks. To change anything about the kernel, look here. This kernel is compatable woth GPyTorch (https://gpytorch.ai).
- Train.py contains the code to train the model via MLE. Note that the model in Train.py outputs both a case count prediction and a hotspot (binary) prediction. In this example we use a variational strategy to reduce the computational burden.

### Examples

A simple example of how to train a model

```
# Training data loader
train_X, train_y, train_hotspot, _, _, _ = train_test_split(C, D, H, locs, p=p, tau=None)
n = train_y.shape[0] # Number of data points
train_dataset = TensorDataset(train_X, train_y)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Inducing points for variational inference
init_inducing_idx = random.sample(list(np.arange(n)), batch_size) # randomly initialized inducing point indices
inducing_x        = train_X[init_inducing_idx, :]          # randomly selected inducing points

# Define models
model      = DeepGP(n_features, inducing_x=inducing_x)
gaussian_likelihood = gpytorch.likelihoods.GaussianLikelihood()
bernoulli_likelihood = gpytorch.likelihoods.BernoulliLikelihood()

model, _, _ = train_MLE_bernoulli(
			model, gaussian_likelihood, bernoulli_likelihood, train_loader, n, 
			delta=1e-5,
			num_epochs = 25,
			ngd_lr     = 1e-3, 
			adam_lr    = 1e-2,
			print_iter = 10,
			batch_size = batch_size,
			modelname  = modelname)

```

Neural Kernel implementation from https://arxiv.org/pdf/2106.00072.pdf
