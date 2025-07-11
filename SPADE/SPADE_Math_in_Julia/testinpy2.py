import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, beta
from sklearn.decomposition import FastICA


logvarallx = []
logvarally = []
icalogvarallx = []
icalogvarally = []
# vx = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
vy = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
# Parameters for the Cauchy distribution
ux = np.random.rand()
sx = np.random.rand()
# Define Cauchy distributions for x and y axes
xdist = beta(ux, sx)
# xdist = chi2(df=vx)
ydist = chi2(df=vy)  # Chi-squared distribution with ν degrees of freedom

# Number of samples and matrix size
num_samples = 1
matrix_size = (10, 10)

# Initialize x and y matrices
x = np.zeros(matrix_size)
y = np.zeros(matrix_size)

# Generate samples and populate matrices
for i in range(matrix_size[0]):
    for j in range(matrix_size[1]):
        x[i, j] = xdist.rvs(size=1)[0]
        y[i, j] = ydist.rvs(size=1)[0]
x_flat = x.flatten()
y_flat = y.flatten()
# Create histograms for x and y
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(x_flat, bins=20, edgecolor='k')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of x')

plt.subplot(1, 2, 2)
plt.hist(y_flat, bins=20, edgecolor='k')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of y')

plt.tight_layout()
plt.show()

 # Calculate the covariance matrix C as the sum of Cx and Cy
Cx = x@x.T
Cy = y@y.T
C = Cx + Cy
P, W = np.linalg.eig(C)
P = P**(-1/2)
P= np.diag(P)
Ctran = P.T@W.T @ C @ W @P

# Calculate Kx and Ky
Kx = P.T@W.T @ Cx @ W @P
Ky = P.T@W.T @ Cy @ W @P

# Perform eigenvalue decomposition on Ky
Dy, Zy =np.linalg.eig(Ky)
Dy = np.diag(Dy)
# Define V as WPZy
V = W @ P @ Zy

# Calculate V'CxV and V'CyV
result1 = V.T @ Cx @ V
result2 = V.T @ Cy @ V
# Calculate I - Dy
identity_minus_Dy = np.eye(M=Dy.shape[0],N=Dy.shape[1],like=Dy) - Dy

# Display the results
print("V'CxV:")
print(result1)
print("I - Dy:")
print(identity_minus_Dy)
print("V'CyV:")
print(result2)
print("Dy:")
print(Dy)
print("Reconstructed Cy:")
print(W @ np.linalg.inv(P) @ Zy @ Dy @ (Zy.T@np.linalg.inv(P)@W.T))
print("Cy:")
print(Cy)

selected_basis_vectors = V[[0,-1],:]
projected_datax =  selected_basis_vectors @ x
projected_datay = selected_basis_vectors @ y

print(projected_datax)
plt.scatter(projected_datax[:,0], projected_datax[:,1], label="Data X")
plt.scatter(projected_datay[:,0], projected_datay[:,1], label="Data Y")
plt.xlabel("Basis Vector 1")
plt.ylabel("Basis Vector End")
plt.title("Linear Separability of Projected Data")
plt.legend()
plt.show()