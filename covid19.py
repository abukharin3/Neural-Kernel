import csv
import torch
import numpy as np
import json
import branca
import folium


#---------------------------------------------------------------
#
# TRAINING DATA PREPARATION
#
#---------------------------------------------------------------

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
    n_features    = N_FEATURES
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
            D[t:t+p, :].transpose(),
            # M[:, t, :].transpose(),
            cov.transpose()),
            axis=1)       # [ n_counties, 2p]
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

import math
import arrow
import gpytorch
import torch.optim as optim
  
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.means.mean import Mean                                     



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
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=torch.tensor([0]))) * \
        #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=torch.tensor([1]))) * \
        #     gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=torch.tensor([2])))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims=torch.tensor([0]))) * \
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=torch.tensor([1, 2])))
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.ProductStructureKernel(gpytorch.kernels.RBFKernel(), num_dims=1)

    def forward(self, x):
        x_feature = x[:, 4:].clone() # spatio-temporal index
        x_coord   = x[:, :3].clone() # spatio-temporal coordinate
        mean_x    = self.mean_module(x_feature)
        covar_x   = self.covar_module(x_coord)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_MLE(
    model, likelihood, train_loader, n, 
    num_epochs = 10,
    ngd_lr     = 1e-7, 
    adam_lr    = 1e-1,
    print_iter = 100,
    batch_size = 3144,
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
        with gpytorch.settings.cholesky_jitter(1e-1):
            for j, data in enumerate(train_loader):
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

def train_MLE_bernoulli(
    model, gaussian_likelihood, bernoulli_likelihood, train_loader, n, 
    delta = 0.0,
    num_epochs = 10,
    ngd_lr     = 1e-7, 
    adam_lr    = 1e-1,
    print_iter = 100,
    batch_size = 3144,
    modelname  = "STVAmean-RBFkernel"):
    """
    Train GP model by MLE
    """
    # Set training mode
    model.train()
    gaussian_likelihood.train()
    bernoulli_likelihood.train()

    # NGD optimizer for variational parameters
    variational_ngd_optimizer = gpytorch.optim.NGD(
        model.variational_parameters(),
        num_data=n, lr=ngd_lr)

    # Adam optimizer for hyperparameters
    adam_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': gaussian_likelihood.parameters()},
        {'params': bernoulli_likelihood.parameters()},
    ], lr=adam_lr)

    # Loss function - Variational ELBO
    # mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n)
    gaussian_mll = gpytorch.mlls.PredictiveLogLikelihood(gaussian_likelihood, model, num_data=n)
    bernoulli_mll = gpytorch.mlls.PredictiveLogLikelihood(bernoulli_likelihood, model, num_data=n)

    # Training
    for i in range(num_epochs):
        # Within each iteration, we will go over each minibatch of data
        with gpytorch.settings.cholesky_jitter(1e-1):
            for j, data in enumerate(train_loader):
                # Alternate between NGD and Adam optimization
                x_batch, y_batch, hotspot = data
                # adam optimizer
                adam_optimizer.zero_grad()
                output = model(x_batch)
                cases_loss   = -gaussian_mll(output, y_batch)
                hotspot_loss   = -bernoulli_mll(output, hotspot)
                loss = delta * cases_loss + (1-delta) * hotspot_loss
                loss.backward()
                adam_optimizer.step()
                # ngd optimizer
                variational_ngd_optimizer.zero_grad()
                output = model(x_batch)
                cases_loss   = -gaussian_mll(output, y_batch)
                hotspot_loss   = -bernoulli_mll(output, hotspot)
                loss = delta * cases_loss + hotspot_loss
                loss.backward()
                variational_ngd_optimizer.step()
                if j % print_iter == 0:
                    print("[%s] Epoch : %d,\titer : %d,\tloss : %.5e" % (arrow.now(), i, j, loss / print_iter))
                    torch.save(model.state_dict(), "saved_models/%s.pth" % modelname)

    return model, gaussian_likelihood, bernoulli_likelihood


from matplotlib import pyplot as plt

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
            for x_batch, y_batch, _ in data_loader ]
        means = np.concatenate(means, axis=0)                # [ n ]
        # confidence interval
        lowerbs, upperbs = [], []
        for x_batch, y_batch, _ in data_loader:
            std  = gaussian_likelihood(model(x_batch)).stddev.mul_(sigma)
            mean = gaussian_likelihood(model(x_batch)).mean
            lowerb, upperb = mean.sub(std), mean.add(std)    # 2 * [ n_batches ]
            lowerbs.append(lowerb.detach().numpy())
            upperbs.append(upperb.detach().numpy())
        lowerbs = np.concatenate(lowerbs, axis=0)            # [ n ] 
        upperbs = np.concatenate(upperbs, axis=0)            # [ n ] 
        return means, lowerbs, upperbs

def hotspot_prediction(model, bernoulli_likelihood, data_loader):
    '''
    Hotspot classification
    '''
    model.eval()
    bernoulli_likelihood.eval()
    with torch.no_grad():
        # mean
        means = [ 
            bernoulli_likelihood(model(x_batch)).mean.detach().numpy() # [ n_batches ] 
            for x_batch, y_batch, _ in data_loader ]
        means = np.concatenate(means, axis=0)                # [ n ]

        return means

def model_plot(test_hotspot, hotspots, true, mean, c):
    t = range(49)
    fig, ax1 = plt.subplots()

    color = 'black'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Cases', color=color)
    # ax1.plot(t, true[:, c], color="green", label="Real Cases")
    # ax1.plot(t, mean[:, c], color=color, label="Predicted Cases")
    # ax1.tick_params(axis='y', labelcolor=color)
    # ax1.legend(loc="upper left")
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ht = [i for i in range(49) if test_hotspot[i, c]]
    ax2.vlines(ht, 0, 1, color="tab:red", label="Real Hotspots")
    color = 'tab:blue'
    ax2.set_ylabel('Hotspots', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, hotspots[:, c], color=color, label="Hotspot Prediction")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax2.legend(loc="upper right")
    plt.title(c)
    plt.savefig("Results/hotspot_viz/1e-2county{}.png".format(c))
    '''
    Good Results: c = 
    - 244
    - 256
    - 499
    - 652
    - 29
    - 121
    - 126
    '''
    # plt.show()

def plot_hotspot(test_hotspot, hotspots=None, plot_map=True):
    # print("Prediction Accuracy: ", 1 - np.sum((test_hotspot - np.round(hotspots))**2) / (3144 * 49))
    # print("Percent Hotspots: ", np.sum(test_hotspot)/(3144*49))
    # plt.plot(np.sum(test_hotspot, axis=1))
    # plt.plot(np.sum(np.round(hotspots), axis=1))
    # plt.show()


    if plot_map:
        pass
    else:
        return

    for i in range(49):
        print(i)
        # colorscale = branca.colormap.linear.YlOrRd_09.scale(0, max(hotspots[i]))
        # colorscale.caption = "Hotspot Probability Week " + str(i) 

        def style_function(feature):
            county = int(feature['id'][-5:])
            try:
                data=hotspots[i, counties.index(str(county))]
                
            except Exception as e:
                data = 0
            return {
                'fillOpacity': 0.5,
                'weight': 0,
                'fillColor': '#black' if data is None else colorscale(data)
            }

        m = folium.Map(
        location=[38, -95],
        tiles='cartodbpositron',
        zoom_start=4
        )

        # folium.TopoJson(
        # uscountygeo,
        # 'objects.us_counties_20m',
        # style_function=style_function
        # ).add_to(m)

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

        '''
        Find the hotspots in county i
        '''
        weekly_hotspot = test_hotspot[i]
        nonzeros = weekly_hotspot.nonzero()[0]
        for county in nonzeros:
            y, x = locs[county]
            folium.CircleMarker(
                location=[x, y],
                radius=5,
                popup="Laurelhurst Park",
                color="#black",
                fill=True,
                fillOpacity=10,
                fill_color="#black",
            ).add_to(m)

        # colorscale.add_to(m)
        m.save("hotspot_map_viz/viz_noprob{}.html".format(i))

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


# Normalize data
C = C / (np.expand_dims(pop, 0)+1) * pop.mean()
D = D / (np.expand_dims(pop, 0)+1) * pop.mean()

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
# Normalization
# print("FIPS", I)
# print("Geolocation matrix shape", locs.shape)
p = 2
N_FEATURES    = 2 * p + 2

import random

# Model configurations
modelname  = "STVAmean-RBFkernel-insample"
p          = 2
n_features = N_FEATURES
batch_size = 3144

# Training data loader
train_X, train_y, train_hotspot, _, _, _ = train_test_split(C, D, H, locs, p=p, tau=None)
n = train_y.shape[0] # Number of data points
train_dataset = TensorDataset(train_X, train_y, train_hotspot)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Inducing points for variational inference
init_inducing_idx = random.sample(list(np.arange(n)), 500) # randomly initialized inducing point indices
inducing_x        = train_X[init_inducing_idx, :]          # randomly selected inducing points

# Define models
model      = VanillaGP(n_features, inducing_x=inducing_x)
gaussian_likelihood = gpytorch.likelihoods.GaussianLikelihood()
bernoulli_likelihood = gpytorch.likelihoods.BernoulliLikelihood()

with open (r"data/meta/states-10m.json", "r") as f:
    state_json = json.load(f)
with open(r"data/meta/us_counties_20m_topo.json", "r") as f:
    uscountygeo = json.load(f)


counties = list(np.load("data/mat/counties.npy"))
deltas = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 5]
deltas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 5]

# # Training
# # NOTE: Comment lines below if load the model from file
# for d in deltas:
#     modelname = "STVAmean-RBFkernel" + str(d)
#     model, _, _ = train_MLE_bernoulli(
#         model, gaussian_likelihood, bernoulli_likelihood, train_loader, n, 
#         delta=d,
#         num_epochs = 20,
#         ngd_lr     = 1e-2, 
#         adam_lr    = 1e-2,
#         print_iter = 10,
#         batch_size = batch_size,
#         modelname  = modelname)
#     # Load model
#     model.load_state_dict(torch.load("saved_models/%s.pth" % modelname))

#     # Testing data loader
#     data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

#     # Evaluate Model
#     means, lowerbs, upperbs = GP_prediction(model, gaussian_likelihood, data_loader)
#     hotspots = hotspot_prediction(model, bernoulli_likelihood, data_loader)
#     hotspots   = hotspots.reshape(C[p:, :].shape)

#     means   = means.reshape(C[p:, :].shape)   # [ T-p, n_counties ]
#     lowerbs = lowerbs.reshape(C[p:, :].shape) # [ T-p, n_counties ]
#     upperbs = upperbs.reshape(C[p:, :].shape) # [ T-p, n_counties ]

#     real_hotspots=H[p:, :]
#     real_cases=C[p:, :]

#     means, lowerbs1, upperbs1 = GP_prediction(model, gaussian_likelihood, data_loader, sigma=1)
#     _, lowerbs2, upperbs2     = GP_prediction(model, gaussian_likelihood, data_loader, sigma=2)
#     means   = means.reshape(C[p:, :].shape)     # [ T-p, n_counties ]
#     lowerbs1 = lowerbs1.reshape(C[p:, :].shape) # [ T-p, n_counties ]
#     upperbs1 = upperbs1.reshape(C[p:, :].shape) # [ T-p, n_counties ]
#     lowerbs2 = lowerbs2.reshape(C[p:, :].shape) # [ T-p, n_counties ]
#     upperbs2 = upperbs2.reshape(C[p:, :].shape) # [ T-p, n_counties ]

#     # Plot nationwide insample result
#     truecase = C * (np.expand_dims(pop, 0)+1) / pop.mean()
#     predmean = means * (np.expand_dims(pop, 0)+1) / pop.mean()
#     predlow1 = lowerbs1 * (np.expand_dims(pop, 0)+1) / pop.mean()
#     predlow2 = lowerbs2 * (np.expand_dims(pop, 0)+1) / pop.mean()
#     predup1  = upperbs1 * (np.expand_dims(pop, 0)+1) / pop.mean()
#     predup2  = upperbs2 * (np.expand_dims(pop, 0)+1) / pop.mean()

#     np.save("Results/insample/predicted_cases%s.npy" % d, predmean)
#     np.save("Results/insample/predicted_cases_lb1%s.npy" % d, predlow1)
#     np.save("Results/insample/predicted_cases_lb2%s.npy" % d, predlow2)
#     np.save("Results/insample/predicted_cases_ub1%s.npy" % d, predup1)
#     np.save("Results/insample/predicted_cases_ub2%s.npy" % d, predup2)
#     np.save("Results/insample/predicted_hotspot%s.npy" % d, hotspots)
#     np.save("Results/insample/real_hotspot.npy", real_hotspots)
#     np.save("Results/insample/real_cases.npy", truecase)

#---------------------------------------------------------------------
#
# Insample Hotspot prediction accuracy
#
#---------------------------------------------------------------------
with open (r"data/meta/states-10m.json", "r") as f:
    state_json = json.load(f)
with open(r"data/meta/us_counties_20m_topo.json", "r") as f:
    uscountygeo = json.load(f)


counties = list(np.load("data/mat/counties.npy"))
test_hotspot = H[p:, :]
#print(hotspots.shape, test_hotspot.shape)
# np.save("Results/hotspots_pred.npy", hotspots)
# np.save("Results/hotspots_real.npy", test_hotspot)
plot_hotspot(test_hotspot)
# print(counties)
# for c in range(3144):
#     print(c)
#     model_plot(r, p1, C[p:,:], C[p:, :], c)

# pred=np.load("Results/onestep_hotspots.npy")
# real=np.load("Results/onestep_test.npy")
# plot_hotspot
# plot_hotspot(real, pred, True)
# print(pred)
# for c in range(50):
#     print(c)
#     model_plot(real, pred, np.ones(real.shape), C[p:, :], c)

#---------------------------------------------------------------------
#
# Out of sample Hotspot prediction accuracy
#
#---------------------------------------------------------------------

# start_week = 36
# end_week   = T-2
# p          = 2
# n_features = N_FEATURES
# batch_size = 3144

# onestep_hotspots     = []
# onestep_lowerbs1  = []
# onestep_uppperbs1 = []
# onestep_lowerbs2  = []
# onestep_uppperbs2 = []
# for tau in range(start_week, end_week):

#     # Model configurations
#     modelname  = "STVAmean-Prod-RBFkernel-NewObj-p%d-tau%d" % (p, tau)

#     # Training data loader
#     # data from start_week to tau + p
#     train_X, train_y, train_hotspot, test_X, test_y, test_H = train_test_split(C, D, H, locs, p=p, tau=tau)
#     n = train_y.shape[0] # Number of data points
#     train_dataset = TensorDataset(train_X, train_y, train_hotspot)
#     train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

#     # Inducing points for variational inference
#     init_inducing_idx = random.sample(list(np.arange(n)), 500) # randomly initialized inducing point indices
#     inducing_x        = train_X[init_inducing_idx, :]          # randomly selected inducing points

#     # Define models
#     model      = VanillaGP(n_features, inducing_x=inducing_x)
#     gaussian_likelihood = gpytorch.likelihoods.GaussianLikelihood()
#     bernoulli_likelihood = gpytorch.likelihoods.BernoulliLikelihood()

#     # Training
#     # NOTE: Comment lines below if load the model from file
#     model, _, _ = train_MLE_bernoulli(
#         model, gaussian_likelihood, bernoulli_likelihood, train_loader, n, 
#         delta=0.1,
#         num_epochs = 25,
#         ngd_lr     = 1e-3, 
#         adam_lr    = 1e-2,
#         print_iter = 10,
#         batch_size = batch_size,
#         modelname  = modelname)

#     print("[%s] training the %d-th model..." % (arrow.now(), tau))

#     # Load model
#     model.load_state_dict(torch.load("saved_models/%s.pth" % modelname))

#     # Testing data loader
#     test_dataset = TensorDataset(test_X, test_y, test_H)
#     test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     # Generate out-of-sample prediction
#     hotspots = hotspot_prediction(model, bernoulli_likelihood, test_loader)
#     hotspots   = hotspots.reshape(C[tau + p:, :].shape)

#     # Get the first prediction result (one-step ahead)
#     onestep_hotspots.append(hotspots[-1, :])
# onestep_hotspots    = np.stack(onestep_hotspots, 0)
# print(onestep_hotspots.shape)
# onestep_test = H[start_week+p:end_week+2]
# print(onestep_test.shape)
# np.save("Results/onestep_hotspots.npy", onestep_hotspots)
# np.save("Results/onestep_test.npy", onestep_test)

# print("Prediction Accuracy: ", 1 - np.sum((onestep_means - np.round(onestep_test))**2) / (3144 * 29))


# # # Plot Countywise one-step ahead result
# # fulton_id = I.index('13089')
# # plot_nationwide_prediction(
# #     C[start_week+p:end_week+2, fulton_id], 
# #     onestep_means[:, fulton_id], 
# #     onestep_lowerbs1[:, fulton_id], 
# #     onestep_uppperbs1[:, fulton_id], 
# #     onestep_lowerbs2[:, fulton_id],
# #     onestep_uppperbs2[:, fulton_id], 
# #     filename="One-step ahead prediction for Dekalb County, GA")
# # brooks_id = I.index('13027')
# # plot_nationwide_prediction(
# #     C[start_week+p:end_week+2, brooks_id], 
# #     onestep_means[:, brooks_id], 
# #     onestep_lowerbs1[:, brooks_id], 
# #     onestep_uppperbs1[:, brooks_id], 
# #     onestep_lowerbs2[:, brooks_id], 
# #     onestep_uppperbs2[:, brooks_id], 
# #     filename="One-step ahead prediction for Brooks County, GA")

# # # Plot nationwide insample result
# # plot_nationwide_prediction(
# #     C[start_week+p:end_week+2, :].sum(1), 
# #     onestep_means.sum(1), 
# #     onestep_lowerbs1.sum(1),
# #     onestep_uppperbs1.sum(1), 
# #     onestep_lowerbs2.sum(1), 
# #     onestep_uppperbs2.sum(1), 
# #     filename="One-step ahead prediction for US")




# # Plot nationwide insample result
# # plot_nationwide_prediction(
# #     C[p:, :].sum(1), means.sum(1), lowerbs.sum(1), upperbs.sum(1), 
# #     filename="In-sample estimation for US")
# # # Plot Countywise insample result
# # fulton_id = I.index('13121')
# # plot_nationwide_prediction(
# #     C[p:, fulton_id], means[:, fulton_id], lowerbs[:, fulton_id], upperbs[:, fulton_id], 
# #     filename="In-sample estimation for Fulton County, GA")
# # brooks_id = I.index('13027')
# # plot_nationwide_prediction(
# #     C[p:, brooks_id], means[:, brooks_id], lowerbs[:, brooks_id], upperbs[:, brooks_id], 
# #     filename="In-sample estimation for Brooks County, GA")










