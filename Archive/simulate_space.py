import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.decomposition import FastICA

# Set parameters (same as before)
num_subjects_per_group = 10
num_parcellated_regions = 165
num_time_points = 600
vmin = 0.0
vmax = 2

# Generate structured data for Group 1 and Group 2 with added noise, subject-specific shifts, and general noise
group1_structure = np.zeros((num_parcellated_regions, num_time_points))
group2_structure = np.zeros((num_parcellated_regions, num_time_points))

# Define the structures for each group
# Group 1: Information in all four corners and the middle
group1_structure[:num_parcellated_regions//6, :num_time_points//6] = 1.0
group1_structure[:num_parcellated_regions//6, -num_time_points//6:] = 1.0
group1_structure[-num_parcellated_regions//6:, -num_time_points//6:] = 1.0
group1_structure[-num_parcellated_regions//6:, :num_time_points//6] = 1.0

group1_structure[num_parcellated_regions//4:num_parcellated_regions*3//4, num_time_points//4:num_time_points*3//4] = 1.0

# Group 2: Information in two vertical lines near the sides and in the middle
group2_structure[num_parcellated_regions//6+3:num_parcellated_regions//6+6, :num_time_points//6] = 1.0
group2_structure[num_parcellated_regions*3//4+3:num_parcellated_regions*3//4+6, :num_time_points//6] = 1.0
group2_structure[num_parcellated_regions//4:num_parcellated_regions*3//4, num_time_points//4:num_time_points*3//4] = 1.0
# Variance for subject-specific shifts
subject_shift_variance = 0  # Adjust this value to control the shift

# Variance for general noise
general_noise_variance = 0  # Adjust this value to control the noise level

# TODO Add random scaling to each row

# Create synthetic data for Group 1
group1_data = np.zeros((num_subjects_per_group, num_parcellated_regions, num_time_points))
for i in range(num_subjects_per_group):
    subject_shift_x = int(subject_shift_variance * np.random.randn())  # Shift in the x-direction
    subject_shift_y = int(subject_shift_variance * np.random.randn())  # Shift in the y-direction
    
    subject_variation = np.zeros((num_parcellated_regions, num_time_points))
    for x in range(num_parcellated_regions):
        for y in range(num_time_points):
            x_idx = (x - subject_shift_x) % num_parcellated_regions
            y_idx = (y - subject_shift_y) % num_time_points
            subject_variation[x, y] = group1_structure[x_idx, y_idx]

    # Add general noise
    general_noise = general_noise_variance * np.random.randn(num_parcellated_regions, num_time_points)
    group1_data[i] = subject_variation + general_noise

# Create synthetic data for Group 2
group2_data = np.zeros((num_subjects_per_group, num_parcellated_regions, num_time_points))
for i in range(num_subjects_per_group):
    subject_shift_x = int(subject_shift_variance * np.random.randn())  # Shift in the x-direction
    subject_shift_y = int(subject_shift_variance * np.random.randn())  # Shift in the y-direction
    
    subject_variation = np.zeros((num_parcellated_regions, num_time_points))
    for x in range(num_parcellated_regions):
        for y in range(num_time_points):
            x_idx = (x + subject_shift_x) % num_parcellated_regions
            y_idx = (y + subject_shift_y) % num_time_points
            subject_variation[x, y] = group2_structure[x_idx, y_idx]

    # Add general noise
    general_noise = general_noise_variance * np.random.randn(num_parcellated_regions, num_time_points)
    group2_data[i] = subject_variation + general_noise

# Create plots for each matrix in Group 1
plt.figure(figsize=(12, 6))
for i in range(num_subjects_per_group):
    plt.subplot(2, num_subjects_per_group, i + 1)
    plt.imshow(group1_data[i], cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
    plt.title(f'Subject {i+1}')

# Create plots for each matrix in Group 2
for i in range(num_subjects_per_group):
    plt.subplot(2, num_subjects_per_group, num_subjects_per_group + i + 1)
    plt.imshow(group2_data[i], cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
    plt.title(f'Subject {i+1}')

plt.tight_layout()
plt.show()

# # Normalize the Data
# group1 = (group1_data - np.mean(group1_data, axis=2, keepdims=True))/np.std(group1_data, axis=2, keepdims=True)
# group2 = (group2_data - np.mean(group2_data, axis=2, keepdims=True))/np.std(group2_data, axis=2, keepdims=True)
# # Create plots for each matrix in Group 1
# plt.figure(figsize=(12, 6))
# for i in range(num_subjects_per_group):
#     plt.subplot(2, num_subjects_per_group, i + 1)
#     plt.imshow(group1[i], cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
#     plt.title(f'Subject {i+1}')

# # Create plots for each matrix in Group 2
# for i in range(num_subjects_per_group):
#     plt.subplot(2, num_subjects_per_group, num_subjects_per_group + i + 1)
#     plt.imshow(group2[i], cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
#     plt.title(f'Subject {i+1}')

# plt.tight_layout()
# plt.show()

# Compute Regularized Covariance Matrices
# Set parameters (You can adjust γ, β, and sc values)
gamma = 0.1  # Regularization parameter γ
beta = 0.1   # Regularization parameter β
sc = 1.0    # Scaling parameter (you can adjust this value)

# Initialize arrays to store the regularized covariance matrices for each subject
group1_reg_cov_matrices = []
group2_reg_cov_matrices = []
# Compute the ground truth covariance matrices for each group (before adding noise)
ground_truth_C1 = group1_structure@group1_structure.T
ground_truth_C2 = group2_structure@group2_structure.T
for i in range(num_subjects_per_group):
    # Compute initial spatial covariance matrices for each group
    C1sub = group1_data[i]@group1_data[i].T
    C2sub = group2_data[i]@group2_data[i].T

    I = np.identity(C1sub.shape[0])  # Identity matrix of the same size as C1 and C2

    # Apply the second regularization term using the ground truth covariance matrices as Gc
    C1_hat = (1 - beta) * sc * C1sub + beta * ground_truth_C1
    C2_hat = (1 - beta) * sc * C2sub + beta * ground_truth_C2

    # Apply regularization to the initial covariance matrices
    C1reg = (1 - gamma) * C1_hat + gamma * I
    C2reg = (1 - gamma) * C2_hat + gamma * I

    group1_reg_cov_matrices.append(C1reg)
    group2_reg_cov_matrices.append(C2reg)
group1_reg_cov_matrices = np.array(group1_reg_cov_matrices)
group2_reg_cov_matrices = np.array(group2_reg_cov_matrices)
# Create plots for the regularized covariance matrices of each subject within each group
plt.figure(figsize=(12, 6))

for i in range(num_subjects_per_group):
    plt.subplot(2, num_subjects_per_group, i + 1)
    plt.imshow(group1_reg_cov_matrices[i], aspect='auto')
    plt.title(f'Group 1, Subject {i + 1}')

for i in range(num_subjects_per_group):
    plt.subplot(2, num_subjects_per_group, num_subjects_per_group + i + 1)
    plt.imshow(group2_reg_cov_matrices[i], aspect='auto')
    plt.title(f'Group 2, Subject {i + 1}')

plt.tight_layout()
plt.show()

Cx = np.mean(group1_reg_cov_matrices,axis=0)
Cy = np.mean(group2_reg_cov_matrices,axis=0)
# Create a new figure
plt.figure()

# Plot C1avg
plt.subplot(1, 2, 1)
plt.imshow(Cx, cmap='YlGnBu', aspect='auto')
plt.title(f'Group 1')

# Plot C2avg
plt.subplot(1, 2, 2)
plt.imshow(Cy, cmap='YlGnBu', aspect='auto')
plt.title(f'Group 2')

plt.tight_layout()
plt.show()
# Begin SPADE Pipeline
C = Cx + Cy
wy, Vy = scipy.linalg.eigh(Cx,C,eigvals_only=False)
Dy = np.diag(wy[::-1])
wx, Vx = scipy.linalg.eigh(Cy,C,eigvals_only=False)
Dx = np.diag(wx[::-1])

# Select components to get
n = 2
basisy = Vy[:,:n]
basisx = Vx[:,:n]

# Calculate the dot product of the two vectors
basis = np.concatenate((basisy,basisx),axis=1)

# Regression to get spatial maps time course for each subject
# Initialize arrays to store the spatial maps time course for each subject
group1_filter_time = []
group2_filter_time = []

basisplus = np.linalg.pinv(basis)
for i in range(num_subjects_per_group):
    group1_filter_time.append(basisplus@group1_data[i])
    group2_filter_time.append(basisplus@group2_data[i])
group1_filter_time = np.array(group1_filter_time)
group2_filter_time = np.array(group2_filter_time)
# Get plots to compare discriminality
group1 = np.var(group1_filter_time,axis=2)
group2 = np.var(group2_filter_time,axis=2)

# # Scatter plot for group1
plt.scatter(group1[:, 0], group1[:, 1], label='Group 1', c='b', marker='o')

# Scatter plot for group2
plt.scatter(group2[:, 0], group2[:, 1], label='Group 2', c='r', marker='x')

# Add labels and legend
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.legend()

# Show the plot
plt.show()

# Regression to get subject specific spatial maps
# Initialize arrays to store the spatial maps time course for each subject
# TODO eventually regress onto voxels
group1_filter_space = []
group2_filter_space = []

for i in range(num_subjects_per_group):
    timeplus1 = np.linalg.pinv(group1_filter_time[i].T)
    timeplus2 = np.linalg.pinv(group2_filter_time[i].T)
    group1_filter_space.append(timeplus1@group1_data[i].T)
    group2_filter_space.append(timeplus2@group2_data[i].T)
group1_filter_space = np.array(group1_filter_space)
group2_filter_space = np.array(group2_filter_space)

# Plot the original data (group1)
plt.figure(figsize=(12, 6))
plt.subplot(2, 4, 1)
plt.imshow(group1_data[0, :, :], cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
plt.title('Original Data 1')
plt.colorbar()

# Plot the time course of the spatial map (group1_filter_time)
plt.subplot(2, 4, 2)
plt.imshow(group1_filter_time[0, :,:], aspect='auto')
plt.title('Time Course of Spatial Map 1')
plt.xlabel('Time')
plt.ylabel('Value')

# Plot the spatial map (group1_filter_space)
plt.subplot(2, 4, 3)
plt.imshow(group1_filter_space[0, :, :].T, aspect='auto')
plt.title('Spatial Map 1')
plt.colorbar()

plt.subplot(2, 4, 4)
plt.imshow(group1_filter_space[0, :, :].T@group1_filter_time[0, :,:], cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
plt.title('Reconstructed 1')
plt.colorbar()

plt.subplot(2, 4, 5)
plt.imshow(group2_data[0, :, :], cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
plt.title('Original Data 2 ')
plt.colorbar()

# Plot the time course of the spatial map (group1_filter_time)
plt.subplot(2, 4, 6)
plt.imshow(group2_filter_time[0, :,:], aspect='auto')
plt.title('Time Course of Spatial Map 2')
plt.xlabel('Time')
plt.ylabel('Value')

# Plot the spatial map (group1_filter_space)
plt.subplot(2, 4, 7)
plt.imshow(group2_filter_space[0, :, :].T, aspect='auto')
plt.title('Spatial Map 2')
plt.colorbar()

plt.subplot(2, 4, 8)
plt.imshow(group2_filter_space[0, :, :].T@group2_filter_time[0, :,:], cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
plt.title('Reconstructed 2')
plt.colorbar()

plt.tight_layout()
plt.show()

# ICA extension
# Create an instance of FastICA with whitening (default behavior)
ica = FastICA(n_components=n)
# Fit and transform your data
ica_components = ica.fit_transform((basis@basisplus@group1_data[0]).T)
mixer = ica.mixing_


# Create a new figure
plt.figure()
# Plot C2avg
plt.subplot(2, 2, 1)
plt.imshow(basis@basisplus@group1_data[0], cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
plt.title(f'Projection onto Basis')
plt.subplot(2, 2, 2)
plt.imshow(mixer, aspect='auto')
plt.title(f'Mixing Matrix')
# Plot C1avg
plt.subplot(2, 2, 3)
plt.imshow(ica_components.T, aspect='auto')
plt.title(f'Components')
# Plot C2avg
plt.subplot(2, 2, 4)
plt.imshow(mixer@ica_components.T, cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
plt.title(f'ICA Reconstructoin')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 8))
# First Component
plt.subplot(2, 2, 1)
plt.imshow(basis[:, 1:2] @ np.linalg.pinv(basis[:, 1:2])@group1_data[0], cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
plt.title(f'First Component')
# Second Component
plt.subplot(2, 2, 2)
plt.imshow(basis[:, 2:3] @ np.linalg.pinv(basis[:, 2:3])@group1_data[0], cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
plt.title(f'Second Component')
# ICA Components
plt.subplot(2, 2, 3)
plt.imshow(basis[:, 3:4] @ np.linalg.pinv(basis[:, 3:4])@group1_data[0], cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
plt.title(f'ICA Components')
# ICA Reconstruction
plt.subplot(2, 2, 4)
plt.imshow(basis[:, 4:] @ np.linalg.pinv(basis[:, 4:])@group1_data[0], cmap='YlGnBu', aspect='auto', vmin=vmin, vmax=vmax)
plt.title(f'ICA Reconstruction')
plt.tight_layout()
plt.show()