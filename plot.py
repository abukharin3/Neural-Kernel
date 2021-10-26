import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn")

H = np.load("data/mat/hotspot_1-17.npy") # [51, 3144]

plt.plot(np.sum(H, axis = 1))
plt.show()


