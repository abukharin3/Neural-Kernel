import numpy as np


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



deltas = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1, 5]

for d in deltas:
	means = np.load("Results/insample/predicted_cases%s.npy" % d)
	lower3sigma = np.load("Results/insample/predicted_cases_lb%s.npy" % d)
	upper3sigma = np.load("Results/insample/predicted_cases_ub%s.npy" % d)
	hotspot_pred = np.load("Results/insample/predicted_hotspot%s.npy" % d)
	truehotspot = np.load("Results/insample/real_hotspot.npy")
	cases = np.load("Results/insample/real_cases.npy")

	truecase = C * (np.expand_dims(pop, 0)+1) / pop.mean()
	predmean = means * (np.expand_dims(pop, 0)+1) / pop.mean()
	predlow3 = lower3sigma * (np.expand_dims(pop, 0)+1) / pop.mean()
	predup3  = upper3sigma * (np.expand_dims(pop, 0)+1) / pop.mean()

	np.save("Results/insample_delta/real_cases.npy", truecase)
	np.save("Results/insample_delta/predicted_cases%s.npy" % d, predmean)
	np.save("Results/insample_delta/predicted_hotspot%s.npy" % d, hotspot_pred)
	np.save("Results/insample_delta/real_hotspot.npy", truehotspot)
	np.save("Results/insample_delta/predicted_lb3_%s.npy" % d, predlow3)
	np.save("Results/insample_delta/predicted_ub3_%s.npy" % d, predup3)
