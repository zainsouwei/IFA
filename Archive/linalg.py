import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Generate random 2x2 data with non-Gaussian original signals
np.random.seed(0)

# Generate non-Gaussian original signals
signal1 = np.random.exponential(scale=1, size=100)
signal2 = np.sin(2 * np.pi * np.random.rand(100))

data = np.vstack((signal1, signal2))

# Mix the data to create observed signals
mixing_matrix = np.array([[1, 2], [3, 4]])
observed_signals = np.dot(mixing_matrix, data)

# Perform PCA on the observed signals
mean_centered_data = observed_signals - observed_signals.mean(axis=1, keepdims=True)
cov_matrix = np.cov(mean_centered_data)
_, pca_components = np.linalg.eig(cov_matrix)

# Step 1: Center the data
centered_data = observed_signals - observed_signals.mean(axis=1, keepdims=True)

# Step 2: Whitening - Decorrelate and scale the data
cov_matrix = np.cov(centered_data)
_, S, V = np.linalg.svd(cov_matrix, full_matrices=False)
whitening_matrix = np.dot(np.dot(V.T, np.diag(1.0 / np.sqrt(S))), V)
whitened_data = np.dot(whitening_matrix, centered_data)

# Step 3: Apply ICA to find independent components
ica = FastICA(n_components=2)
independent_components = ica.fit_transform(whitened_data.T).T

# Step 4: Plot the data as described with lines representing PCA components
plt.figure(figsize=(15, 10))

# Plot the observed data with PCA lines
plt.subplot(2, 2, 1)
plt.scatter(observed_signals[0], observed_signals[1], c='r')
plt.quiver(observed_signals.mean(1)[0], observed_signals.mean(1)[1], pca_components[0, 0], pca_components[1, 0], angles='xy', scale_units='xy', scale=0.5, color='g', label='PCA Component 1')
plt.quiver(observed_signals.mean(1)[0], observed_signals.mean(1)[1], pca_components[0, 1], pca_components[1, 1], angles='xy', scale_units='xy', scale=0.5, color='b', label='PCA Component 2')
plt.title('Observed Data with PCA Lines')
plt.legend()

# Plot the whitened data with PCA lines
plt.subplot(2, 2, 2)
plt.scatter(whitened_data[0], whitened_data[1], c='g')
plt.quiver(whitened_data.mean(1)[0], whitened_data.mean(1)[1], pca_components[0, 0], pca_components[1, 0], angles='xy', scale_units='xy', scale=0.5, color='b', label='PCA Component 1')
plt.quiver(whitened_data.mean(1)[0], whitened_data.mean(1)[1], pca_components[0, 1], pca_components[1, 1], angles='xy', scale_units='xy', scale=0.5, color='r', label='PCA Component 2')
plt.title('Whitened Data with PCA Lines')
plt.legend()

# Plot the independent components
plt.subplot(2, 2, 3)
plt.scatter(independent_components[0], independent_components[1], c='b')
plt.title('Independent Components')

# Plot the original unobserved data
plt.subplot(2, 2, 4)
plt.scatter(data[0], data[1], c='b')
plt.title('Original Unobserved Data')

plt.tight_layout()
plt.show()


