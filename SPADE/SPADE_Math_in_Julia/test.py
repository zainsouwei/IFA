import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, beta
import scipy.linalg
from sklearn.decomposition import FastICA

# Define the dimensions
num_regions = 3  # Number of parcellated brain regions
num_time_points = 3  # Number of time points

# Generate the simulated data for motor task (X)
X = np.zeros((num_regions, num_time_points))

for region in range(num_regions):
    # Simulate non-Gaussian data using a uniform distribution
    X[region, :] = np.random.uniform(0, 1, num_time_points)

# Generate the simulated data for resting state (Y)
Y = np.zeros((num_regions, num_time_points))

for region in range(num_regions):
    # Simulate non-Gaussian data using a different uniform distribution
    Y[region, :] = np.random.uniform(1, 2, num_time_points)

Cx = X@X.T
Cy = Y@Y.T
C = Cx+Cy
w, V = scipy.linalg.eigh(C,Cx,eigvals_only=False)
print(V.T@V)
# Create a list of pairs of basis vectors
# print(V)
# pairs = [(i, j) for i in range(V.shape[1]) for j in range(i+1, V.shape[1])]

# # Create a 3D plot for each pair of basis vectors and calculate the angle
# for pair in pairs:
#     i, j = pair
#     vi = V[:, i]
#     vj = V[:, j]

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     ax.quiver(0, 0, 0, vi[0], vi[1], vi[2], color='b', label=f'v{i}')
#     ax.quiver(0, 0, 0, vj[0], vj[1], vj[2], color='r', label=f'v{j}')
    
#     angle = np.arccos(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj)))  # Calculate the angle
#     angle_deg = np.degrees(angle)  # Convert angle to degrees
#     ax.text(0.1, 0.1, 1.1, f'Angle: {angle_deg:.2f} degrees', fontsize=12)

#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.set_zlim(0, 1)
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title(f'Relationship between v{i} and v{j}')
#     ax.legend()

# plt.show()
# P, W = np.linalg.eig(C)
# P = P**(-1/2)
# P= np.diag(P)
# Ctran = P.T@W.T @ C @ W @P

# # Calculate Kx and Ky
# Kx = P.T@W.T @ Cx @ W @P
# Ky = P.T@W.T @ Cy @ W @P

# # Perform eigenvalue decomposition on Ky
# Dy, Zy =np.linalg.eig(Ky)
# Dy = np.diag(Dy)
# # Define V as WPZy
# V = W @ P @ Zy
# # Calculate V'CxV and V'CyV
# result1 = V.T @ Cx @ V
# result2 = V.T @ Cy @ V
# # Calculate I - Dy
# identity_minus_Dy = np.eye(M=Dy.shape[0],N=Dy.shape[1],like=Dy) - Dy