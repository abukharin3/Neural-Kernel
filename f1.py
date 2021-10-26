import numpy as np
import matplotlib.pyplot as plt

# realp=	np.load("Results/onestep_hotspots.npy")
real = np.load("data/mat/hotspot_1-17.npy")[-16:-1]       # [ T, n_counties ]
pred=	np.load("Results/onestep_test.npy")

#print(real.shape, pred.shape)



for i in range(6):
	real = real[1:-1]
	pred = pred[1:-1]
	preds = []
	for c in range(3144):
		maxf1 = 0
		best_t = 1
		for threshold in np.linspace(0, 1, 5):
			thotspots = (pred[:, c] > threshold).astype(int)
			tp = np.sum(thotspots * real[:, c])
			fp = np.sum(thotspots * (1 - real[:, c]))

			fn = np.sum((1-thotspots) * real[:, c])
			#print(fp, fn)
			precision = tp / (tp + fp)
			recall = tp / (tp + fn)
			f1 = tp / (tp + 0.5*(fp + fn))

			if f1 > maxf1:
				maxf1 = f1
				best_t = threshold
		if maxf1 != 0:
			pass
			#print(maxf1)

		thotspots = (pred[:, c] > best_t).astype(int)
		preds.append(thotspots)
	preds = np.array(preds).T
	tp = np.sum(preds * real)
	fp = np.sum(preds * (1 - real))
	fn = np.sum((1-preds) * real)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1 = tp / (tp + 0.5*(fp + fn))
	#mse = np.sum((predmean - truecase[2:]) ** 2 / (3144 * 49))
	#MD.append(mse)
	print(precision, recall, f1)


# [(0.614, 32, 1), (0.621, 64, 1), (0.636, 128, 1), (0.637, 256, 1), (0.630, 32, 2) (0.6211, 64, 2), (0.624, 128, 2), (0.627, 256, 2)]