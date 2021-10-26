import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def det_sigma(psix, psiy):
	psi = np.array([psix, psiy])
	alpha = np.arctan(psiy / psix)
	A = 1
	Q = np.sqrt(4 + np.linalg.norm(psi) ** 4 * (np.pi**2)) / 2
	return (Q + np.linalg.norm(psi)**2 / 2)*(Q - np.linalg.norm(psi)**2 / 2) - (np.cos(alpha)/(np.pi**2))


xline = np.linspace(-1, 1, 1000)
yline = np.linspace(-1, 1, 1000)
for x in xline:
	for y in yline:
		if det_sigma(x, y) <= 0:
			print(det_sigma)
