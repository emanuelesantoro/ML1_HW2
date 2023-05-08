import numpy as np
import matplotlib.pyplot as plt

x_dataset = np.load('data/x_datapoints.npy')
y_targets = np.load('data/y_datapoints.npy')
print(f'Shapes: {x_dataset.shape}, {y_targets.shape}')

plt.scatter(x_dataset[:,0], x_dataset[:,1], c=y_targets, cmap="plasma")
plt.show()
plt.scatter(x_dataset[:,0], y_targets, c=x_dataset[:,1], cmap="plasma")
plt.show()
plt.scatter(x_dataset[:,1], y_targets, c=x_dataset[:,0], cmap="plasma")
plt.show()