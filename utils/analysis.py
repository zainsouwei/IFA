import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap
from mne_connectivity.viz import plot_connectivity_circle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy.linalg import eigh,subspace_angles
from pyriemann.utils.tangentspace import untangent_space, mean_covariance
from sklearn.metrics import accuracy_score
from sklearn.covariance import OAS,LedoitWolf, EmpiricalCovariance
import hcp_utils as hcp
from nilearn import plotting
from pyriemann.estimation import Covariances
from sklearn.decomposition import MiniBatchSparsePCA
from filters import TSSF
from classification import linear_classifier
import torch
from nilearn.mass_univariate import permuted_ols
from mne.stats import permutation_t_test, permutation_cluster_1samp_test, permutation_cluster_test, ttest_ind_no_p, ttest_1samp_no_p
from nilearn.plotting import view_surf
import sys
import pickle
sys.path.append('/utils')

from tangent import tangent_transform, tangent_classification
from filters import feature_generation, FKT
from regression import deconfound

def scatter_with_lines(data1, data2, label1='Series 1', label2='Series 2', xlabel='X', ylabel='Y', title='Scatter Plot with Connecting Lines',output_dir='path'):
    """
    Creates a scatter plot with lines connecting corresponding points from two series. 
    Supports data provided either as lists of coordinates or as dictionaries.

    Parameters:
    - data1, data2: Either lists of coordinates (x, y) or dictionaries with classifiers as keys and 'accuracy' as values.
    - label1, label2: Labels for the two series.
    - xlabel, ylabel: Labels for the x and y axes.
    - title: Title for the plot.
    """
    # Check if data is in dictionary format
    if isinstance(data1, dict) and isinstance(data2, dict):
        # Ensure both dictionaries have the same classifiers
        classifiers1 = list(data1.keys())
        classifiers2 = list(data2.keys())
        assert classifiers1 == classifiers2, "The classifiers (keys) must match between the two dictionaries."
        
        # Extract accuracies and set x positions based on classifiers
        accuracies1 = [metrics['accuracy'] for metrics in data1.values()]
        accuracies2 = [metrics['accuracy'] for metrics in data2.values()]
        x_positions = range(len(classifiers1))
        
        # Plotting for dictionaries
        plt.figure(figsize=(14, 8))
        plt.scatter(x_positions, accuracies1, label=label1, color='blue', s=100)
        plt.scatter(x_positions, accuracies2, label=label2, color='orange', s=100)
        
        # Draw lines connecting points
        for i in range(len(x_positions)):
            plt.plot([x_positions[i], x_positions[i]], [accuracies1[i], accuracies2[i]], color='gray', linestyle='--')
        
        plt.xticks(x_positions, classifiers1, rotation=45, ha='right', fontsize=12)
        
    else:
        # Assuming data is in list format for (x, y) coordinates
        x1, y1 = data1[:,0], data1[:,1]
        x2, y2 = data2[:,0], data2[:,1]
        plt.figure(figsize=(12, 6))
        plt.scatter(x1, y1, label=label1, color='blue')
        plt.scatter(x2, y2, label=label2, color='orange')
        
        # Draw lines connecting points
        for x_1, y_1, x_2, y_2 in zip(x1, y1, x2, y2):
            plt.plot([x_1, x_2], [y_1, y_2], color='gray', linestyle='--')
    
    # Common labels and title for both cases
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title}.svg"))
    plt.close('all')

def unupper_noweighting(vec):
    # Compute the matrix size `n`
    n = int((np.sqrt(1 + 8 * len(vec)) - 1) / 2)
    full_matrix = np.zeros((n, n), dtype=vec.dtype)
    upper_indices = np.triu_indices(n)
    full_matrix[upper_indices] = vec
    i_lower = np.tril_indices(n, -1)
    full_matrix[i_lower] = full_matrix.T[i_lower]
    return full_matrix

def tangent_t_test(train_covs, test_covs, test_labels, alpha=.05, permutations=False, paired=True, metric='riemann', deconf=False, con_confounder_train=None, cat_confounder_train=None, con_confounder_test=None, cat_confounder_test=None,output_dir="path",basis="ICA", random_seed=42):
    train_vecs, test_vecs, mean = tangent_transform(train_covs, test_covs, metric=metric)
    if deconf:
        train_vecs, test_vecs = deconfound(train_vecs, con_confounder_train, cat_confounder_train, X_test=test_vecs, con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test)

    if paired:
        # Paired: compute differences between conditions and use MNE's one-sample permutation t-test.
        diffs = test_vecs[test_labels == 1] - test_vecs[test_labels == 0]
        t_values, corrected_p_values, _ = permutation_t_test(diffs, n_permutations=permutations, tail=0, n_jobs=-1, seed=random_seed, verbose=False)
        reject = corrected_p_values < alpha
    else:
        groupA = test_vecs[test_labels == 1]
        groupB = test_vecs[test_labels == 0]
        design_matrix = np.column_stack((np.ones(test_vecs.shape[0]), np.concatenate([np.ones(groupA.shape[0]), np.zeros(groupB.shape[0])])))
        data = np.concatenate([groupA, groupB])
        res = permuted_ols(design_matrix, data, n_perm=permutations, two_sided_test=True, n_jobs=-1, random_state=random_seed, output_type='dict')
        t_values = res['t'][1, :]
        corrected_p_values = 10 ** (-res['logp_max_t'][1, :])
        reject = corrected_p_values < alpha

    t_values_thresholded = t_values * reject
    diff_thresholded = corrected_p_values * reject
    diff_thresholded[diff_thresholded == 0.0] = alpha + 1e-5
    
    diff_thresholded_matrix = unupper_noweighting(diff_thresholded)
    t_values_thresholded_matrix = unupper_noweighting(t_values_thresholded)
    groupA = untangent_space(np.mean(test_vecs[test_labels==1], axis=0)[np.newaxis, :], mean, metric=metric)[0, :, :] * unupper_noweighting(reject)
    groupB = untangent_space(np.mean(test_vecs[test_labels==0], axis=0)[np.newaxis, :], mean, metric=metric)[0, :, :] * unupper_noweighting(reject)
    
    global_min = min(groupA.min(), groupB.min())
    global_max = max(groupA.max(), groupB.max())
    colors = [(0, 'blue'), (0.5, 'white'), (1, 'red')]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    labels = [f"Region {i}" for i in range(groupA.shape[0])]
    green_cmap = plt.get_cmap('Greens_r').copy()
    green_cmap.set_over('white')
    # Plot each chord plot in its own figure
    # Create a single figure for subplots
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle("Connectivity Plots", fontsize=18, y=0.98)

    # Adjust the layout to minimize dead space
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)

    # Create four polar subplots manually
    ax1 = fig.add_subplot(2, 2, 1, polar=True)
    ax2 = fig.add_subplot(2, 2, 2, polar=True)
    ax3 = fig.add_subplot(2, 2, 3, polar=True)
    ax4 = fig.add_subplot(2, 2, 4, polar=True)

    # Plot 1: Corrected P-values
    im1 = plot_connectivity_circle(np.where(diff_thresholded_matrix < alpha, diff_thresholded_matrix, np.nan), labels, facecolor=(0.3, 0.3, 0.3, 1), colormap=green_cmap, vmin=diff_thresholded_matrix.min(), vmax=alpha, fig=fig, ax=ax1, show=False, colorbar=False)
    ax1.set_title("Corrected P-values", fontsize=12)
    cbar1 = fig.colorbar(plt.cm.ScalarMappable(cmap=green_cmap, norm=plt.Normalize(vmin=diff_thresholded_matrix.min(), vmax=alpha)), ax=ax1, shrink=0.7, orientation='vertical')
    cbar1.set_label('Corrected P-values', fontsize=10)

    # Plot 2: T-values
    im2 = plot_connectivity_circle(np.where(np.abs(t_values_thresholded_matrix) > 0, t_values_thresholded_matrix, np.nan), labels, facecolor=(0.3, 0.3, 0.3, 1), colormap=custom_cmap, vmin=t_values_thresholded_matrix.min(), vmax=t_values_thresholded_matrix.max(), fig=fig, ax=ax2, show=False, colorbar=False)
    ax2.set_title("T-values", fontsize=12)
    cbar2 = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=t_values_thresholded_matrix.min(), vmax=t_values_thresholded_matrix.max())), ax=ax2, shrink=0.7, orientation='vertical')
    cbar2.set_label('Tangent T-values', fontsize=10)

    # Plot 3: Group A Connectivity
    im3 = plot_connectivity_circle(np.where(np.abs(groupA) > 0, groupA, np.nan), labels, facecolor=(0.3, 0.3, 0.3, 1), colormap=custom_cmap, vmin=groupA.min(), vmax=groupA.max(), fig=fig, ax=ax3, show=False, colorbar=False)
    ax3.set_title(f'Group {1} Connectivity', fontsize=12)
    cbar3 = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=groupA.min(), vmax=groupA.max())), ax=ax3, shrink=0.7, orientation='vertical')
    cbar3.set_label('Covariance', fontsize=10)

    # Plot 4: Group B Connectivity
    im4 = plot_connectivity_circle(np.where(np.abs(groupB) > 0, groupB, np.nan),labels, facecolor=(0.3, 0.3, 0.3, 1), colormap=custom_cmap, vmin=groupB.min(), vmax=groupB.max(), fig=fig, ax=ax4, show=False, colorbar=False)
    ax4.set_title(f'Group {0} Connectivity', fontsize=12)
    cbar4 = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=groupB.min(), vmax=groupB.max())),ax=ax4, shrink=0.7, orientation='vertical')
    cbar4.set_label('Covariance', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{basis}_chord_connectivity.svg"))
    
    # Create heatmaps in subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Heatmaps of Connectivity Differences", fontsize=14)
    sns.heatmap(diff_thresholded_matrix, cmap=green_cmap, vmin=diff_thresholded_matrix.min(), vmax=alpha, cbar_kws={'label': 'Corrected P-values'}, ax=axes[0, 0])
    axes[0, 0].set_title("Corrected P-values")
    sns.heatmap(t_values_thresholded_matrix, cmap=custom_cmap, center=0, cbar_kws={'label': 'Tangent T-values'}, ax=axes[0, 1])
    axes[0, 1].set_title(f'T-values on Tangent Plane | {np.linalg.norm(t_values_thresholded_matrix)}')
    sns.heatmap(groupA, cmap=custom_cmap, vmin=global_min, vmax=global_max, center=0, cbar_kws={'label': 'Covariance'}, ax=axes[1, 0])
    axes[1, 0].set_title(f'Group {1} Mean Connectivity')
    sns.heatmap(groupB, cmap=custom_cmap, vmin=global_min, vmax=global_max, center=0, cbar_kws={'label': 'Covariance'}, ax=axes[1, 1])
    axes[1, 1].set_title(f'Group {0} Mean Connectivity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{basis}_connectivity.svg"))
    plt.close('all')

    return (diff_thresholded_matrix, t_values_thresholded_matrix, groupA, groupB)


# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5662067
def var_diff(train_data, train_covs, train_labels, test_data, test_labels, metric='riemann', method='log-var', basis="ICA", deconf=False, con_confounder_train=None, cat_confounder_train=None, con_confounder_test=None, cat_confounder_test=None,output_dir="path"):
    unique_labels = np.unique(train_labels)
    clf = SVC(kernel='linear', C=0.1, class_weight='balanced')

    _, filters_all = FKT(train_covs, train_labels, metric=metric, deconf=deconf, con_confounder_train=con_confounder_train, cat_confounder_train=cat_confounder_train, visualize=False, output_dir=None)

    # Initialize list to store results (accuracy and distance)
    results = []
    for n in range(1, filters_all.shape[1] // 2 + 1): 
        filters = np.hstack([filters_all[:, :n], filters_all[:, -n:]])  # Select top and bottom n eigenvectors
        train_features, test_features = feature_generation(train_data,test_data, filters,method=method,metric=metric,cov="oas")
        if deconf:
            train_features, test_features = deconfound(train_features, con_confounder_train, cat_confounder_train, X_test=test_features, con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test)

        # Train SVM regression classifier on training data
        clf.fit(train_features, train_labels)

        # Predict on the test data and calculate accuracy
        y_pred = clf.predict(test_features)
        accuracy = accuracy_score(test_labels, y_pred)

        # Calculate class means for distance (using the training data)
        mean_group1_test = np.mean(test_features[test_labels==unique_labels[1]], axis=0)
        mean_group2_test = np.mean(test_features[test_labels==unique_labels[0]], axis=0)
        mean_dist = np.linalg.norm(mean_group1_test - mean_group2_test)

        # Store accuracy and Riemannian distance for this n
        results.append(np.array([n, mean_dist, accuracy]))
        # Plot when n=1
        if n == 1:
            plt.figure(figsize=(8, 6))
            plt.scatter(test_features[test_labels==unique_labels[1]][:, 0], test_features[test_labels==unique_labels[1]][:, 1], label=f'Group {unique_labels[1]} {method} (Test)', color='blue')
            plt.scatter(test_features[test_labels==unique_labels[0]][:, 0], test_features[test_labels==unique_labels[0]][:, 1], label=f'Group {unique_labels[0]} {method} (Test)', color='red')

            # Plot the line connecting the two means
            plt.plot([mean_group1_test[0], mean_group2_test[0]], [mean_group1_test[1], mean_group2_test[1]], 'k--', label=f'Mean Distance: {mean_dist:.2f}')

            # Decision boundary
            x_values = np.array([train_features[:, 0].min(), train_features[:, 0].max()])
            y_values = -(clf.intercept_ + clf.coef_[0][0] * x_values) / clf.coef_[0][1]
            plt.plot(x_values, y_values, 'g-', label='Decision Boundary')

            # Display plot
            plt.xlabel(f'{method} Feature {unique_labels[0]}')
            plt.ylabel(f'{method} Feature {unique_labels[1]}')
            plt.title(f'{basis} {method} FKT Feature Comparison and SVM Decision Boundary')
            plt.text(0.05, 0.95, f'Accuracy: {accuracy:.2f}', transform=plt.gca().transAxes, fontsize=12,verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightgrey'))
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'{basis}_{method}_2d.svg'))
            plt.close('all')
    return np.array(results)

def reconstruction_plot(label1_recon, label2_recon, label1="Label 1", label2="Label 2", output_dir="path"):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(label1_recon, label=f'{label1} Reconstruction Error', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(label2_recon, label=f'{label2} Reconstruction Error', color='orange', fill=True, alpha=0.3)
    plt.xlabel('% of Original Data Variance Explained by Reconstructing Subject Data from Group Spatial Maps')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {label1} and {label2} Reconstruction Errors')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{label1}_{label2}_reconstruction.svg'))
    plt.close('all')



#############################################################################################################################################################
################################################################## Under Development ########################################################################
# Sparse Bilinear Logistic Regression https://ww3.math.ucla.edu/camreport/cam14-12.pdf
class SparseBilinearLogisticRegression:
    def __init__(self, s, t, r, mu1=0.1, mu2=1.0, nu1=0.1, nu2=1.0, max_iter=100, tol=1e-4):
        self.s = s  # Feature matrix rows
        self.t = t  # Feature matrix columns
        self.r = r  # Low-rank factorization dimension
        self.mu1 = mu1  # L1 regularization on U
        self.mu2 = mu2  # L2 regularization on U
        self.nu1 = nu1  # L1 regularization on V
        self.nu2 = nu2  # L2 regularization on V
        self.max_iter = max_iter
        self.tol = tol  # Convergence tolerance

        # Initialize weight matrices
        self.U = np.random.randn(s, r)
        self.V = np.random.randn(t, r)
        self.b = 0.0  # Bias term
    
    # Expit as defined by 12(a-c)
    def expit(self, X, y):
        trace = np.array([np.trace(self.U.T @ X_i @ self.V) for X_i in X])
        return  np.power((1 + np.exp(y*(trace + self.b))),-1) 

    # 15a
    def gradient_U(self, X, y):
        all_subs_grad_U = (self.expit(X, y) * y)[:, np.newaxis, np.newaxis] * (X@self.V)
        grad_U = -np.mean(all_subs_grad_U,axis=0)
        return grad_U
    
    # 15b
    def gradient_V(self, X, y):
        all_subs_grad_V = (self.expit(X, y) * y)[:, np.newaxis, np.newaxis] * (np.transpose(X,(0,2,1))@self.U)
        grad_V = -np.mean(all_subs_grad_V,axis=0)
        return grad_V
    
    # 15c
    def gradient_b(self, X, y):
        all_subs_grad_b = (self.expit(X, y) * y)
        grad_b = -np.mean(all_subs_grad_b,axis=0)
        return grad_b

    def soft_thresholding(self, Z, tau):
        # https://eeweb.poly.edu/iselesni/lecture_notes/sparse_penalties/sparse_penalties.pdf Page 5
        return np.sign(Z) * np.maximum(np.abs(Z) - tau, 0)
    
    def fit(self, X, y):
        n = y.shape[0]
        U,E,Vh = np.linalg.svd(np.mean(X,axis=0),full_matrices=False)
        self.U = U[:,:self.r]
        self.V = Vh[:self.r,:].T

        for _ in range(self.max_iter):
            U_old = self.U
            V_old = self.V
            b_old = self.b
            # Lipshitz Constant for U 
            L_u = (np.sqrt(2)/n)*(np.sum((np.linalg.norm(X@self.V,ord='fro',axis=(1,2)) + 1)**2,axis=0))
            # Update bhat
            self.b = self.b - (1/L_u)*self.gradient_b(X, y)

            # Update U
            grad_U = self.gradient_U(X, y)
            self.U = self.soft_thresholding((L_u*self.U - grad_U)/(L_u+ self.mu2), self.mu1/(L_u+ self.mu2))
            
            # Lipshitz Constant for v 
            L_v = (np.sqrt(2)/n)*(np.sum((np.linalg.norm(np.transpose(X,(0,2,1))@self.U,ord='fro',axis=(1,2)) + 1)**2,axis=0))

            # Update b
            self.b = self.b - (1/L_v)*self.gradient_b(X, y)
            
            # Update V
            grad_V = self.gradient_V(X, y)
            self.V = self.soft_thresholding((L_v*self.V - grad_V)/(L_v+ self.nu2), self.nu1 /(L_v+ self.nu2))
            
            print(np.linalg.norm(self.U - U_old, 'fro') + np.linalg.norm(self.V - V_old, 'fro') + abs(self.b - b_old))

            # Convergence check
            if np.linalg.norm(self.U - U_old, 'fro') + np.linalg.norm(self.V - V_old, 'fro') + abs(self.b - b_old) < self.tol:
                break

    
    def predict(self, X):
        logits = np.array([np.trace(self.U.T @ X_i @ self.V) for X_i in X]) + self.b
        return np.sign(logits)

def bilinear_class(method, time_norm=False):
    for fold in range(1):
        # Load indices and labels
        indices_dir = os.path.join(outputfolder, f"fold_{fold}", "Indices")
        train_idx = np.load(os.path.join(indices_dir, "train_idx.npy"))
        test_idx = np.load(os.path.join(indices_dir, "test_idx.npy"))
        labels = np.load(os.path.join(outputfolder, "labels.npy"))
        labels[labels == 0] = -1
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        # Load spatial maps
        dual_dir = os.path.join(outputfolder, f"fold_{fold}", f"Dual_Regression_{nPCA}")
        if time_norm:
            SpatialMaps = np.load(os.path.join(dual_dir, f"{method}_spatial_maps.npy"))
        else:
            SpatialMaps = np.load(os.path.join(dual_dir, f"{method}_spatial_mapdm.npy"))

        train_maps = SpatialMaps[train_idx]
        test_maps = SpatialMaps[test_idx]
        print(f"Training Shape: {train_maps.shape}, Testing Shape: {test_maps.shape}")

        # Train Sparse Bilinear Logistic Regression
        sblr = SparseBilinearLogisticRegression(
            s=train_maps.shape[1], 
            t=train_maps.shape[2], 
            r=1,  # Assuming a reduced rank of 10
            mu1=.1, mu2=.1, nu1=.1, nu2=.1
        )

        sblr.fit(train_maps, train_labels)

        # Predict on test set
        predictions = sblr.predict(test_maps)

        # Evaluate performance
        accuracy = np.mean(predictions == test_labels)
        print(f"Fold {fold}: Accuracy = {accuracy:.4f}")
        return sblr.U, sblr.V
#############################################################################################################################################################
#############################################################################################################################################################
       
# https://ieeexplore.ieee.org/abstract/document/1467563
def spatial_fda(train_maps,train_labels,within=True):

    # Separate Train Maps into Groups
    groupA_train = train_maps[train_labels==1]
    groupB_train = train_maps[train_labels==0]

    # Calculate Average Set of Train Spatial Maps per Group
    groupA_mean_train = np.mean(groupA_train,axis=0)
    groupB_mean_train = np.mean(groupB_train,axis=0)
    # cov_est = EmpiricalCovariance(assume_centered=True)
    # cov_est = LedoitWolf(assume_centered=True)
    cov_est = OAS(assume_centered=True)

    all_mean = np.mean(train_maps,axis=0)
    Sb = groupA_train.shape[0]*cov_est.fit((groupA_mean_train - all_mean).T).covariance_
    Sb += groupB_train.shape[0]*cov_est.fit((groupB_mean_train - all_mean).T).covariance_
    # Sb = (groupA_mean_train - groupB_mean_train)@(groupA_mean_train - groupB_mean_train).T

    if within:
        groupA_within = np.sum(np.array([cov_est.fit((A-groupA_mean_train).T).covariance_ for A in groupA_train]),axis=0)
        groupB_within = np.sum(np.array([cov_est.fit((B-groupB_mean_train).T).covariance_ for B in groupB_train]),axis=0)
        Sw = (groupA_within + groupB_within) 
        # # An Alternative Using Weighted & Riemannian Means Instead of Sums
        # Sb /= groupA_train.shape[0] + groupB_train.shape[0]
        # groupA_within = np.array([cov_est.fit((A-groupA_mean_train).T).covariance_ for A in groupA_train])
        # groupB_within = np.array([cov_est.fit((B-groupB_mean_train).T).covariance_ for B in groupB_train])
        # Sw = mean_covariance(np.vstack((groupA_within,groupB_within)),metric=metric)
        # For class A (shape: (N_A, C, V)) with mean groupA_mean_train (C, V)

        # # An Alternative Using Pooling Prior to Covariance Formation
        # deviations_A = np.concatenate([subject - groupA_mean_train for subject in groupA_train], axis=1)
        # deviations_A = deviations_A.T  # shape: (N_A * V, C)
        # # For class B (shape: (N_B, C, V)) with mean groupB_mean_train (C, V)
        # deviations_B = np.concatenate([subject - groupB_mean_train for subject in groupB_train], axis=1)
        # deviations_B = deviations_B.T  # shape: (N_B * V, C)
        # # Pool all deviations together:
        # all_deviations = np.vstack((deviations_A, deviations_B))
        # # Compute the pooled covariance:
        # Sw = cov_est.fit(all_deviations).covariance_
        # print("Pooled within-class covariance (both classes) shape:", Sw.shape)
        _, U = eigh(Sb,Sw)
    else:
        _, U = eigh(Sb)
    
    return U, U.T

def sparse_spatial_dist(train_maps,train_labels,n_components=None,alpha=.01,batch_size=100):
    # Separate Train Maps into Groups
    groupA_train = train_maps[train_labels==1]
    groupB_train = train_maps[train_labels==0]

    # Calculate Average Set of Train Spatial Maps per Group
    A = np.mean(groupA_train,axis=0)
    B = np.mean(groupB_train,axis=0)
    # Parameters for MiniBatch Sparse PCA
    if n_components is None:
        n_components = A.shape[0]  # Number of sparse components to extract

    # Apply MiniBatch Sparse PCA
    mini_batch_sparse_pca = MiniBatchSparsePCA(
        n_components=n_components,
        alpha=alpha,
        batch_size=batch_size,
        random_state=42
    )
    
    diff_matrix = A - B

    # Fit the model to the difference matrix
    # Enforce sparsity in the Space Direction not the combination of component directions
    mini_batch_sparse_pca.fit(diff_matrix)

    # Explained variance approximation (for interpretation)
    approximation = mini_batch_sparse_pca.transform(diff_matrix)
    # Reverse the order of columns to be in ascending order like other two methods
    approximation = approximation[:, ::-1]

    print(np.linalg.norm(approximation,axis=0))
    norms = np.linalg.norm(approximation,axis=0)
    norms[norms==0] = 1
    U = approximation/norms
    
    ## Get the sparse components (principal directions)
    # sparse_components = mini_batch_sparse_pca.components_

    ## Rough equivalent of eigenvalues (only measures variance explained for that component)
    ##   U's in this case are not forced to be orthogonal, so this norm does not account for covariance
    # e = np.linalg.norm(approximation,axis=0)**2

    return U, U.T

## This Learns Seperate Subspaces to Project Each Group onto so will result in higher 
### accuracies than the other methods if want to purely maximize cosine similarity can do eigh since 
#### this amounts to vAB.Tv where A and B are orthonormal basis, however no longer represents subspace angles
def grassmann_dist(train_maps,train_labels):
    # Separate Train Maps into Groups
    groupA_train = train_maps[train_labels==1]
    groupB_train = train_maps[train_labels==0]

    # Calculate Average Set of Train Spatial Maps per Group
    A = np.mean(groupA_train,axis=0)
    B = np.mean(groupB_train,axis=0)
    # The commented out code is a potential way for regularization by reducing the "sample/ambient" space via PCA"
    # A = hcp.normalize(A.T).T
    # B = hcp.normalize(B.T).T
    # U, S, Red = np.linalg.svd(np.vstack((A,B)),full_matrices=False)
    # A_q, A_r = np.linalg.qr((A@Red[:20,:].T).T)
    # B_q, B_r = np.linalg.qr((B@Red[:20,:].T).T)
    A_q, _ = np.linalg.qr((A).T)
    B_q, _ = np.linalg.qr((B).T)
    product = A_q.T @ B_q
    U, S, Vt = np.linalg.svd(product, full_matrices=False)
    # print(product.shape)
    # S, U = np.linalg.eigh(product)
    # Vt = U.T
    angles = np.arccos(S)  # Principal angles
    print(S)
    # Confirm S matches subspace_angles result
    assert np.all(np.abs(angles - subspace_angles(A.T, B.T)[::-1]) < 1e-3), "Mismatch in singular values and principal angles"
    return U, Vt  # Principal angles and directions


def spatial_vis(map_accs, discrim_dir_acc,outputfolder=None,basis="IFA"):

    # === 1. Gather the accuracies per classifier ===
    # Assume map_accs is a list of length n_maps; each element is a dict with keys = classifier names.
    n_maps = len(map_accs)
    clf_names = list(map_accs[0].keys())  # e.g., ["SVM (C=0.1)", "LDA", ...]
    # Initialize a dictionary for each classifier to hold distributions by method.
    acc_by_clf = {clf: {} for clf in clf_names}
    # Collect baseline accuracies (from each map) for each classifier.
    for clf in clf_names:
        # Each map_accs[i][clf] is assumed to be a dict with an 'accuracy' key.
        baseline_vals = [map_accs[i][clf]['accuracy'] for i in range(n_maps)]
        acc_by_clf[clf]["Baseline"] = baseline_vals

    # Collect accuracies from each discriminative method.
    # Here, discrim_dir_acc is assumed to be a dictionary:
    #   key: method (e.g., 1, 2, 4, etc.)
    #   value: a list (length = n_maps) of result dictionaries.
    for method, res_list in discrim_dir_acc.items():
        label = f"Method {method}"
        for clf in clf_names:
            # For each projection direction (assumed one per map),
            # extract the accuracy value from each result dictionary.
            vals = [res[clf]['accuracy'] for res in res_list]
            acc_by_clf[clf][label] = vals

    # === 2. Plot histograms (one per classifier) with distributions overlaid ===
    n_clf = len(clf_names)
    fig, axes = plt.subplots(n_clf, 1, figsize=(8, 4 * n_clf))
    if n_clf == 1:
        axes = [axes]

    # For each classifier, plot a histogram for each method distribution.
    for i, clf in enumerate(clf_names):
        ax = axes[i]
        method_labels = list(acc_by_clf[clf].keys())  # e.g., ["Baseline", "Method 1", "Method 2", ...]
        n_methods = len(method_labels)
        colors = sns.color_palette("husl", n_methods)  # Generate distinct colors for each method.
        for j, m_label in enumerate(method_labels):
            data = np.array(acc_by_clf[clf][m_label])
            # Plot histogram with a step style and overlay a KDE.
            sns.histplot(data, bins=data.shape[0]*2, stat="frequency", element="step", fill=True,
                         color=colors[j], alpha=0.4, ax=ax)
            # sns.kdeplot(data, color=colors[j], ax=ax)
            # Mark the mean of this distribution with a vertical dashed line.
            mean_val = np.mean(data)
            ax.axvline(mean_val, color=colors[j], linestyle="--", linewidth=2,
                       label=f"{m_label} mean, max: {mean_val:.2f},{np.max(data):.2f}")
        ax.set_title(f"Classifier: {clf}")
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Frequency")
        ax.legend()
    plt.tight_layout()

     # === 3. Save the plot as an SVG file if an output folder is provided ===
    if outputfolder:
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        svg_path = os.path.join(outputfolder, f"{basis}_spatial_discrim.svg")
        plt.savefig(svg_path, format="svg")
        print(f"Plot saved to {svg_path}")
    else:
        plt.show()
    plt.close(fig)

    # Scatter plot visualization
    fig, axes = plt.subplots(n_clf, 1, figsize=(8, 4 * n_clf))
    if n_clf == 1:
        axes = [axes]

    for i, clf in enumerate(clf_names):
        ax = axes[i]
        method_labels = list(acc_by_clf[clf].keys())
        n_methods = len(method_labels)
        colors = sns.color_palette("husl", n_methods)

        for j, m_label in enumerate(method_labels):
            data = np.array(acc_by_clf[clf][m_label])
            ax.plot(range(data.shape[0]), np.sort(data), '--o',alpha=0.6, label=m_label, color=colors[j])

        ax.set_xticks(range(data.shape[0]))
        ax.set_xlabel("Map/Basis Vector Sorted By Classification Accuracy")
        ax.set_title(f"Classifier: {clf}")
        ax.set_ylabel("Accuracy")
        ax.legend()

    plt.tight_layout()

    # Save scatter plot
    if outputfolder:
        scatter_path = os.path.join(outputfolder, f"{basis}_spatial_discrim_scatter.svg")
        plt.savefig(scatter_path, format="svg")
        print(f"Scatter plot saved to {scatter_path}")
    else:
        plt.show()
    plt.close(fig)


def reduce_dimensionality_torch(train_spatial_maps, test_spatial_maps, device=None, n=100, svd=True, demean=True):
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        
    if not isinstance(train_spatial_maps, torch.Tensor):
        train_spatial_maps = torch.tensor(train_spatial_maps, dtype=torch.float32, device=device)
    else:
        train_spatial_maps = train_spatial_maps.to(device)
    
    if not isinstance(test_spatial_maps, torch.Tensor):
        test_spatial_maps = torch.tensor(test_spatial_maps, dtype=torch.float32, device=device)
    else:
        test_spatial_maps = test_spatial_maps.to(device)
    
    n_comp = int(train_spatial_maps.shape[-1] / n)
    
    if demean:
        train_mean = torch.mean(train_spatial_maps, dim=0)
        train_data = train_spatial_maps - train_mean
        test_data = test_spatial_maps - train_mean
    else:
        train_data = train_spatial_maps
        test_data = test_spatial_maps

    if svd:
        _, S, Vh = torch.linalg.svd(train_data, full_matrices=False)
        # Take the top n_comp components.
        reduced_train = train_data @ Vh[:n_comp, :].T
        reduced_test = test_data @ Vh[:n_comp, :].T
        return reduced_train.cpu().numpy(), reduced_test.cpu().numpy()
    else:
        # EVD branch: Compute the dual covariance matrix and perform eigen-decomposition.
        cov_train = train_data @ train_data.T / train_spatial_maps.size(1)
        
        e, U = torch.linalg.eigh(cov_train)
        # Compute principal components in the original space using the dual formulation.
        V = ((train_data.T @ U) / torch.sqrt(e.unsqueeze(0) + 1e-10))[:, -n_comp:]
        
        X_train_reduced = train_data @ V
        X_test_reduced = test_data @ V
        return X_train_reduced.cpu().numpy(), X_test_reduced.cpu().numpy()

def spatial_discrimination(train_maps, train_labels, test_maps, test_labels,methods=[1,2,4,5],metric="riemann",visualize=True,outputfolder=None,basis="IFA"):
    # Note for method 1, 2, & 4 Vt == U.T, This is just done so the same code can be used for the grassmann dist
    #           which operates on two different subspaces where U != V
    classifier_model = "SVM (C=0.1)"
    # First look at accuracy of individual maps that span the subspace
    map_accs = []
    for i in range(train_maps.shape[1]):
        # train_map_reduced, test_map_reduced = reduce_dimensionality_torch(train_maps[:, i, :], test_maps[:, i, :], device=None, n=100, svd=True, demean=True)
        # results = linear_classifier(train_map_reduced, train_labels, test_map_reduced, test_labels, clf_str='Logistic Regression', z_score=1)
        results = linear_classifier(train_maps[:, i, :], train_labels, test_maps[:, i, :], test_labels, clf_str=classifier_model, z_score=1)
        map_accs.append(results)

    # Compute the maximum separating directions within that subspace based on different heurstics
    discrim_dir_acc = {}   # key: method code, value: list of accuracy result dicts (one per projection direction)
    discrim_dir = {}       # key: method code, value: (U, Vt)
    for method in methods:
        # Maximize Between Class Distance Measured via Euclidean Distance
        if method == 1:
            U, Vt = spatial_fda(train_maps, train_labels,within=False)
         # Maximize Between Class Distance and Minimize Within Class Spread Measured via Euclidean Distance
        elif method == 2:
            U, Vt = spatial_fda(train_maps, train_labels,within=True)
        # Sparse Maximization of Between Class Distance Measured via Euclidean Distance
        elif method == 3:
            U, Vt = sparse_spatial_dist(train_maps, train_labels,n_components=None,alpha=.01,batch_size=100)
        # Maximizition of Between Class Distance Measure via Cosine Similarity (i.e., Maximize Subspace Angles)
        elif method == 4:
            U, Vt = grassmann_dist(train_maps,train_labels)
        # CSP (Maximize Distance Between Class Average Covariances)
        elif method == 5:
            cov_est = Covariances(estimator='oas')
            train_covs = cov_est.transform(train_maps)
            eigs, U, _, _ = TSSF(train_covs, train_labels, clf_str='Logistic Regression', metric=metric, deconf=False, con_confounder_train=None, cat_confounder_train=None, z_score=0, haufe=False, visualize=False, output_dir=None)
            U = U[:,np.argsort(eigs)]
            Vt = U.T

        groupA_train = train_maps[train_labels==1]
        groupB_train = train_maps[train_labels==0]

        groupA_test = test_maps[test_labels==1]
        groupB_test = test_maps[test_labels==0]
        accs = []
        for i in range(train_maps.shape[1]):
            train_proj = np.vstack((U[:,i].T@groupA_train, Vt[i,:]@groupB_train))
            proj_train_labels = np.hstack((np.ones(groupA_train.shape[0]),np.zeros(groupB_train.shape[0])))
            test_proj = np.vstack((U[:,i].T@groupA_test, Vt[i,:]@groupB_test))
            proj_test_labels = np.hstack((np.ones(groupA_test.shape[0]),np.zeros(groupB_test.shape[0])))
            # train_reduced,test_reduced = reduce_dimensionality_torch(train_proj, test_proj, device=None, n=100, svd=True, demean=True)
            # direction_results = linear_classifier(train_reduced, proj_train_labels, test_reduced, proj_test_labels, clf_str='Logistic Regression', z_score=1)
            direction_results = linear_classifier(train_proj, proj_train_labels, test_proj, proj_test_labels, clf_str=classifier_model, z_score=1)            
            accs.append(direction_results)

        discrim_dir_acc[method] = accs
        discrim_dir[method] = (U, Vt)

    if visualize:
        spatial_vis(map_accs, discrim_dir_acc,outputfolder=outputfolder,basis=basis)
    
    
    return (map_accs,discrim_dir_acc,discrim_dir)


def spatial_comparison_vis(results_one, results_two, label_one="Label 1", label_two="Label 2", outputfolder=None, basis="comparison"):
    """
    Compare spatial discrimination results from two methods using the provided labels.
    
    Parameters:
      - results_one: Tuple containing map accuracies and discriminative accuracies for the first method.
      - results_two: Tuple containing map accuracies and discriminative accuracies for the second method.
      - label_one, label_two: Descriptive names for the two methods.
      - outputfolder: Directory to save the SVG plots.
      - basis: Basis string used in file names.
    """
    map_accs_one, discrim_dir_acc_one, _ = results_one
    map_accs_two, discrim_dir_acc_two, _ = results_two
    
    # Extract classifier names and methods
    clf_names = list(map_accs_one[0].keys())  
    methods = list(discrim_dir_acc_one.keys())  
    
    # Define colors
    color1 = "blue"
    color2 = "orange"

    for clf in clf_names:
        for method in methods:

            ### ======= MAP ACCURACY PLOTS (Comparison) ======= ###
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # Prepare data
            data_one_map = [res[clf]['accuracy'] for res in map_accs_one]
            data_two_map = [res[clf]['accuracy'] for res in map_accs_two]

            # Histogram (Map Accuracy)
            ax = axes[0]
            sns.histplot(data_one_map, bins=20, color=color1, alpha=0.5, label=label_one, ax=ax)
            sns.histplot(data_two_map, bins=20, color=color2, alpha=0.5, label=label_two, ax=ax)
            
            ax.axvline(np.mean(data_one_map), color=color1, linestyle="--", linewidth=2, label=f"{label_one} Mean: {np.mean(data_one_map):.2f}")
            ax.axvline(np.mean(data_two_map), color=color2, linestyle="--", linewidth=2, label=f"{label_two} Mean: {np.mean(data_two_map):.2f}")
            ax.set_title(f"Histogram - {clf} (Method {method}) - Map Accuracy")
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Frequency")
            ax.legend()

            # Scatter Plot (Sorted map accuracy values)
            ax = axes[1]
            sorted_one = np.sort(data_one_map)
            sorted_two = np.sort(data_two_map)
            ax.plot(range(len(sorted_one)), sorted_one, 'o--', alpha=0.6, color=color1, label=label_one)
            ax.plot(range(len(sorted_two)), sorted_two, 'o--', alpha=0.6, color=color2, label=label_two)
            ax.set_xticks(range(len(sorted_one)))
            ax.set_xlabel("Maps Ordered by Classifcation Accuracy")
            ax.set_title(f"Scatter Plot - {clf} (Method {method}) - Map Accuracy")
            ax.set_ylabel("Accuracy")
            ax.legend()
            ax.grid(True)  # Add grid lines to the scatter plot

            plt.tight_layout()
            if outputfolder:
                if not os.path.exists(outputfolder):
                    os.makedirs(outputfolder)
                plot_path = os.path.join(outputfolder, f"{basis}_{clf}_Method_{method}_Map_Accuracy.svg")
                plt.savefig(plot_path, format="svg")
                print(f"Map Accuracy Plot saved to {plot_path}")
            else:
                plt.show()
            plt.close(fig)

            ### ======= DISCRIMINATIVE ACCURACY PLOTS (Comparison) ======= ###
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            data_one_disc = [res[clf]['accuracy'] for res in discrim_dir_acc_one[method]]
            data_two_disc = [res[clf]['accuracy'] for res in discrim_dir_acc_two[method]]
            
            ax = axes[0]
            sns.histplot(data_one_disc, bins=20, color=color1, alpha=0.6, label=label_one, ax=ax)
            sns.histplot(data_two_disc, bins=20, color=color2, alpha=0.6, label=label_two, ax=ax)
            ax.axvline(np.mean(data_one_disc), color=color1, linestyle="--", alpha=0.6, linewidth=2, label=f"{label_one} Mean: {np.mean(data_one_disc):.2f}")
            ax.axvline(np.mean(data_two_disc), color=color2, linestyle="--", alpha=0.6, linewidth=2, label=f"{label_two} Mean: {np.mean(data_two_disc):.2f}")
            ax.set_title(f"Histogram - {clf} (Method {method}) - Discriminative Accuracy")
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Frequency")
            ax.legend()

            ax = axes[1]
            sorted_one_disc = np.sort(data_one_disc)
            sorted_two_disc = np.sort(data_two_disc)
            ax.plot(range(len(sorted_one_disc)), sorted_one_disc, 'o--', alpha=0.6, color=color1, label=label_one)
            ax.plot(range(len(sorted_two_disc)), sorted_two_disc, 'o--', alpha=0.6, color=color2, label=label_two)
            ax.set_xticks(range(len(sorted_one_disc)))
            ax.set_xlabel("Maps Ordered by Classifcation Accuracy")
            ax.set_title(f"Scatter Plot - {clf} (Method {method}) - Discriminative Accuracy")
            ax.set_ylabel("Accuracy")
            ax.legend()
            ax.grid(True)  # Add grid lines to the scatter plot

            plt.tight_layout()
            if outputfolder:
                plot_path = os.path.join(outputfolder, f"{basis}_{clf}_Method_{method}_Discriminative_Accuracy.svg")
                plt.savefig(plot_path, format="svg")
                print(f"Discriminative Accuracy Plot saved to {plot_path}")
            else:
                plt.show()
            plt.close(fig)


def spatial_t_test(maps, labels, paired=True, cluster=False, TFCE=None ,perm=10000, all_maps=True, random_seed=42):
    # Set tfce to None use non TFCE cluster based {"start": 0, "step": 0.3}
    # TODO Add per map cluster 
    # Extract groups from the maps
    maps_A = maps[labels == 1]
    maps_B = maps[labels == 0]
    A_samples = maps_A.shape[0]
    B_samples = maps_B.shape[0]
    samples = A_samples + B_samples
    n_maps = maps_A.shape[1]
    n_space = maps_A.shape[2]

    if cluster:
        maps_A_cortex = np.array([single_map[:,hcp.struct.cortex] for single_map in maps_A])
        maps_B_cortex = np.array([single_map[:,hcp.struct.cortex] for single_map in maps_B])

        if paired:          
            
            assert maps_A.shape[0] == maps_B.shape[0], (
                "For a paired T-Test there needs to be the same number of samples in both groups"
            )

            t_vals, clusters, temp_p_vals, _ = permutation_cluster_1samp_test(
                maps_A_cortex - maps_B_cortex,
                threshold=TFCE,  # TFCE parameters
                n_permutations=perm,
                tail=0,                              # Two-tailed test
                adjacency=hcp.cortical_adjacency,    # Adjacency for HCP grayordinates
                # stat_fun = stat_fun_hat,
                n_jobs=-1,
                # buffer_size=1000, 
                out_type='indices',
                seed=random_seed  
            )
        
        else:
            X = [maps_A_cortex, maps_B_cortex]  # Combine groups for cluster-based permutation
            t_vals, clusters, temp_p_vals, _ = permutation_cluster_test(
                X,
                threshold=TFCE,  # TFCE parameters
                n_permutations=perm,
                tail=0,                              # Two-tailed test
                stat_fun=ttest_ind_no_p,
                adjacency=hcp.cortical_adjacency,    # Adjacency for HCP grayordinates
                n_jobs=-1,
                # buffer_size=1000, 
                out_type='indices',
                seed=random_seed 
            )

        n_space_cortex = maps_A_cortex.shape[-1]
        # Default to 1 ensures non signficant p_values
        p_vals = np.ones((n_maps, n_space_cortex))
        for i, ind in enumerate(clusters):
            p_vals[ind] = temp_p_vals[i]
        return t_vals, p_vals
    else:
        if paired:
            assert maps_A.shape[0] == maps_B.shape[0], (
                "For a paired T-Test there needs to be the same number of samples in both groups"
            )

            diff = maps_A - maps_B
            if all_maps:
                t_vals, p_vals, _ = permutation_t_test(
                    diff.reshape(A_samples, -1),
                    n_permutations=perm,
                    tail=0,  # Two-tailed test,
                    n_jobs=-1,
                    seed=random_seed,
                    verbose=False
                )
                
                p_vals = p_vals.reshape(n_maps, -1)
                t_vals = t_vals.reshape(n_maps, -1)
                return t_vals, p_vals
            else:
                t_vals, p_vals = [], []
                for i in range(n_maps):
                    t_obs, p_obs, _ = permutation_t_test(
                        diff[:, i, :], 
                        n_permutations=perm, 
                        tail=0, 
                        n_jobs=-1,
                        seed=random_seed,
                        verbose=False)
                    
                    t_vals.append(t_obs); p_vals.append(p_obs)
                p_vals = np.array(p_vals)
                t_vals = np.array(t_vals)
                return t_vals, p_vals
        else:
            group_labels = np.concatenate([np.ones(A_samples), np.zeros(B_samples)])
            X = np.column_stack((np.ones(len(group_labels)), group_labels))
            # Combine all maps: data shape becomes (n_subjects_total, n_maps, n_space)
            data = np.concatenate([maps_A, maps_B], axis=0)
            if all_maps:

                # Reshape the data to (n_subjects_total, n_maps * n_space)
                Y = data.reshape(samples, -1)
                result = permuted_ols(
                    X, Y, n_perm=perm,
                    two_sided_test=True,
                    n_jobs=-1,
                    random_state=random_seed,
                    output_type='dict'
                )
                neg_log_pvals = result['logp_max_t'][1, :]
                t_scores = result['t'][1, :]

                # Convert negative log p-values to standard p-values.
                p_vals = 10 ** (-neg_log_pvals)
                # Reshape the results to (n_maps, n_space)
                t_vals = t_scores.reshape(n_maps, n_space)
                p_vals = p_vals.reshape(n_maps, n_space)
                return t_vals, p_vals
            
            else:
                # Process each map individually.
                t_scores_list = []
                p_vals_list = []
                for i in range(n_maps):
                    # For map i, combine the data across subjects.
                    data_i = np.concatenate([maps_A[:, i, :], maps_B[:, i, :]], axis=0)
                    results_i = permuted_ols(
                        X, data_i, n_perm=perm,
                        two_sided_test=True,
                        n_jobs=-1,
                        random_state=random_seed,
                        output_type='dict'
                    )
                    neg_log_pvals_i = results_i['logp_max_t'][1, :]
                    t_scores_i = results_i['t'][1, :]             
                    p_vals_i = 10 ** (-neg_log_pvals_i)
                    t_scores_list.append(t_scores_i)
                    p_vals_list.append(p_vals_i)

                t_vals = np.array(t_scores_list)
                p_vals = np.array(p_vals_list)
                return t_vals, p_vals

def spatial_analysis(maps,labels,perm=10000, alpha=0.05, paired=True, cluster=False, TFCE=None , all_maps=True, random_seed=42,output_dir=None):
    
    t_vals, p_vals = spatial_t_test(maps, labels, paired=paired, cluster=cluster, TFCE=TFCE ,perm=perm, all_maps=all_maps, random_seed=random_seed)
    epsilon = 1e-10  # or some other small value
    p_vals = np.clip(p_vals, epsilon, 1)  # ensures p-values are at least epsilon
    p_vals_log = -np.log(p_vals)
    t_thresh = t_vals.copy()
    t_thresh[p_vals >= alpha] = 0

    np.save(os.path.join(output_dir, "p_vals.npy"), p_vals)
    np.save(os.path.join(output_dir, "p_vals_log.npy"), p_vals_log)
    np.save(os.path.join(output_dir, "t_vals.npy"), t_vals)
    np.save(os.path.join(output_dir, "t_vals_thresh.npy"), t_thresh)


    view_surf(
        surf_mesh=hcp.mesh.inflated,
        surf_map=hcp.cortex_data(np.max(p_vals_log, axis=0)),
        bg_map=hcp.mesh.sulc,
        title=f"-log(p) max: {np.sum(np.max(p_vals_log, axis=0))}",
        vmin=0, 
        vmax=np.max(np.max(p_vals_log, axis=0)),
        symmetric_cmap=False,
        cmap='inferno',
    ).save_as_html(os.path.join(output_dir, "logp_max.html"))

    view_surf(
        surf_mesh=hcp.mesh.inflated,
        surf_map=hcp.cortex_data(np.max(p_vals < alpha, axis=0)),
        bg_map=hcp.mesh.sulc,
        title=f"Count Significant (max): {np.sum(np.max(p_vals < alpha, axis=0))}",
    ).save_as_html(os.path.join(output_dir, "sig.html"))

    view_surf(
        surf_mesh=hcp.mesh.inflated,
        surf_map=hcp.cortex_data(np.max(np.abs(t_thresh), axis=0)),
        bg_map=hcp.mesh.sulc,
        title=f"Thresholded T-values (max abs): {np.sum(np.max(np.abs(t_thresh), axis=0))}",
        vmin=0, 
        # vmax=np.max(np.sum(np.max(np.abs(t_thresh), axis=0))),
        symmetric_cmap=False,
        cmap='inferno',
    ).save_as_html(os.path.join(output_dir, "T_val.html"))

    return (t_vals, p_vals, p_vals_log, t_thresh)

import numpy as np
import matplotlib.pyplot as plt
import os

def spatial_t_compare(results_one, results_two, label_one="basis_one", label_two="basis_two", alpha=0.05, output_dir=None):
    t_vals_one, p_vals_one, p_vals_log_one, t_thresh_one = results_one
    t_vals_two, p_vals_two, p_vals_log_two, t_thresh_two = results_two
    n_maps = p_vals_one.shape[0]

    # --- Existing Surface Visualizations ---
    view_surf(
        surf_mesh=hcp.mesh.inflated,
        surf_map=hcp.cortex_data(np.max(p_vals_log_one, axis=0) - np.max(p_vals_log_two, axis=0)),
        bg_map=hcp.mesh.sulc,
        title=f"Difference in -log(p): {np.sum(np.max(p_vals_log_one, axis=0) - np.max(p_vals_log_two, axis=0))}",
    ).save_as_html(os.path.join(output_dir, f"{label_one}_minus_{label_two}_logp_max.html"))

    view_surf(
        surf_mesh=hcp.mesh.inflated,
        surf_map=hcp.cortex_data((np.sum((p_vals_one < alpha), axis=0) > 0) & ~(np.sum((p_vals_two < alpha), axis=0) > 0)),
        bg_map=hcp.mesh.sulc,
        title=f"{label_one} & ~{label_two}: {np.sum((np.sum((p_vals_one < alpha), axis=0) > 0) & ~(np.sum((p_vals_two < alpha), axis=0) > 0))}",
    ).save_as_html(os.path.join(output_dir, f"{label_one}_not{label_two}.html"))

    view_surf(
        surf_mesh=hcp.mesh.inflated,
        surf_map=hcp.cortex_data(~(np.sum((p_vals_one < alpha), axis=0) > 0) & (np.sum((p_vals_two < alpha), axis=0) > 0)),
        bg_map=hcp.mesh.sulc,
        title=f"~{label_one} & {label_two}: {np.sum(~(np.sum((p_vals_one < alpha), axis=0) > 0) & (np.sum((p_vals_two < alpha), axis=0) > 0))}",
    ).save_as_html(os.path.join(output_dir, f"{label_two}_not{label_one}.html"))

    # --- Cumulative and Per-Map Scatter Plots with Shaded Regions and Grid Lines ---
    x = np.arange(n_maps)

    # Plot: Significance Count (p < alpha)
    sig_counts_one = np.sum((p_vals_one < alpha), axis=1)
    sig_counts_two = np.sum((p_vals_two < alpha), axis=1)
    sorted_sig_counts_one = np.sort(sig_counts_one)
    sorted_sig_counts_two = np.sort(sig_counts_two)
    cum_sig_counts_one = np.cumsum(sorted_sig_counts_one)
    cum_sig_counts_two = np.cumsum(sorted_sig_counts_two)

    plt.figure(figsize=(8, 6))
    plt.plot(x, sorted_sig_counts_one, '--o', label=f"{label_one} (per map)", color="blue", alpha=0.5)
    plt.plot(x, sorted_sig_counts_two, '--o', label=f"{label_two} (per map)", color="orange", alpha=0.5)
    plt.plot(x, cum_sig_counts_one, '-s', label=f"{label_one} (cumulative)", color="blue", alpha=1)
    plt.plot(x, cum_sig_counts_two, '-s', label=f"{label_two} (cumulative)", color="orange", alpha=1)
    plt.title(f"Significance Count (p < {alpha}) per Map & Cumulative")
    plt.xlabel("Map Index")
    plt.ylabel("Count of Significant Vertices")
    plt.grid(True)
    # Annotate final cumulative sum values in the upper right of the axes
    plt.text(0.95, 0.95, f"{label_one} total: {cum_sig_counts_one[-1]}", transform=plt.gca().transAxes,
             fontsize=10, color="blue", ha="right", va="top")
    plt.text(0.95, 0.90, f"{label_two} total: {cum_sig_counts_two[-1]}", transform=plt.gca().transAxes,
             fontsize=10, color="orange", ha="right", va="top")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "Significance_Count.svg"), format="svg")
    plt.close()

    # Plot: -log(p) Value Sum
    logp_sum_one = np.sum(p_vals_log_one, axis=1)
    logp_sum_two = np.sum(p_vals_log_two, axis=1)
    sorted_logp_sum_one = np.sort(logp_sum_one)
    sorted_logp_sum_two = np.sort(logp_sum_two)
    cum_logp_sum_one = np.cumsum(sorted_logp_sum_one)
    cum_logp_sum_two = np.cumsum(sorted_logp_sum_two)

    plt.figure(figsize=(8, 6))
    plt.plot(x, sorted_logp_sum_one, '--o', label=f"{label_one} (per map)", color="blue", alpha=0.5)
    plt.plot(x, sorted_logp_sum_two, '--o', label=f"{label_two} (per map)", color="orange", alpha=0.5)
    plt.plot(x, cum_logp_sum_one, '-s', label=f"{label_one} (cumulative)", color="blue", alpha=1)
    plt.plot(x, cum_logp_sum_two, '-s', label=f"{label_two} (cumulative)", color="orange", alpha=1)
    plt.title("-log(p) Value Sum per Map & Cumulative")
    plt.xlabel("Map Index")
    plt.ylabel("-log(p) Sum")
    plt.grid(True)
    # Annotate final cumulative sum values
    plt.text(0.95, 0.95, f"{label_one} total: {cum_logp_sum_one[-1]:.2f}", transform=plt.gca().transAxes,
             fontsize=10, color="blue", ha="right", va="top")
    plt.text(0.95, 0.90, f"{label_two} total: {cum_logp_sum_two[-1]:.2f}", transform=plt.gca().transAxes,
             fontsize=10, color="orange", ha="right", va="top")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "LogP_Value_Sum.svg"), format="svg")
    plt.close()

    # Surface view for difference in thresholded T-values remains unchanged
    view_surf(
        surf_mesh=hcp.mesh.inflated,
        surf_map=hcp.cortex_data(np.max(np.abs(t_thresh_one), axis=0) - np.max(np.abs(t_thresh_two), axis=0)),
        bg_map=hcp.mesh.sulc,
        title=f"Difference in Thresholded T-values ({label_one} - {label_two}): {np.sum(np.max(np.abs(t_thresh_one), axis=0) - np.max(np.abs(t_thresh_two), axis=0))}",
    ).save_as_html(os.path.join(output_dir, "T_val_diff.html"))

    # Plot: Count of Thresholded T-values (nonzero values)
    t_count_one = np.sum(np.abs(t_thresh_one) > 0, axis=1)
    t_count_two = np.sum(np.abs(t_thresh_two) > 0, axis=1)
    sorted_t_count_one = np.sort(t_count_one)
    sorted_t_count_two = np.sort(t_count_two)
    cum_t_count_one = np.cumsum(sorted_t_count_one)
    cum_t_count_two = np.cumsum(sorted_t_count_two)

    plt.figure(figsize=(8, 6))
    plt.plot(x, sorted_t_count_one, '--o', label=f"{label_one} T count (p<{alpha})", color="blue", alpha=0.5)
    plt.plot(x, sorted_t_count_two, '--o', label=f"{label_two} T count (p<{alpha})", color="orange", alpha=0.5)
    plt.plot(x, cum_t_count_one, '-s', label=f"{label_one} T count (cumulative)", color="blue", alpha=1)
    plt.plot(x, cum_t_count_two, '-s', label=f"{label_two} T count (cumulative)", color="orange", alpha=1)
    plt.title(f"Count of Thresholded T-values (p < {alpha}) per Component & Cumulative")
    plt.xlabel("Map Index")
    plt.ylabel("Count of Nonzero T-values")
    plt.grid(True)
    # Annotate final cumulative sum values
    plt.text(0.95, 0.95, f"{label_one} total: {cum_t_count_one[-1]}", transform=plt.gca().transAxes,
             fontsize=10, color="blue", ha="right", va="top")
    plt.text(0.95, 0.90, f"{label_two} total: {cum_t_count_two[-1]}", transform=plt.gca().transAxes,
             fontsize=10, color="orange", ha="right", va="top")
    plt.legend()    
    plt.savefig(os.path.join(output_dir, "T_Value_Count.svg"), format="svg")
    plt.close()

    # Plot: Sum of Thresholded T-values
    t_sum_one = np.sum(np.abs(t_thresh_one), axis=1)
    t_sum_two = np.sum(np.abs(t_thresh_two), axis=1)
    sorted_t_sum_one = np.sort(t_sum_one)
    sorted_t_sum_two = np.sort(t_sum_two)
    cum_t_sum_one = np.cumsum(sorted_t_sum_one)
    cum_t_sum_two = np.cumsum(sorted_t_sum_two)

    plt.figure(figsize=(8, 6))
    plt.plot(x, sorted_t_sum_one, '--o', label=f"{label_one} T sum (p<{alpha})", color="blue", alpha=0.5)
    plt.plot(x, sorted_t_sum_two, '--o', label=f"{label_two} T sum (p<{alpha})", color="orange", alpha=0.5)
    plt.plot(x, cum_t_sum_one, '-s', label=f"{label_one} T sum (cumulative)", color="blue", alpha=1)
    plt.plot(x, cum_t_sum_two, '-s', label=f"{label_two} T sum (cumulative)", color="orange", alpha=1)
    plt.title(f"Sum of Thresholded T-values (p < {alpha}) per Component & Cumulative")
    plt.xlabel("Map Index")
    plt.ylabel("Sum of |T| values")
    plt.grid(True)
    # Annotate final cumulative sum values
    plt.text(0.95, 0.95, f"{label_one} total: {cum_t_sum_one[-1]:.2f}", transform=plt.gca().transAxes,
             fontsize=10, color="blue", ha="right", va="top")
    plt.text(0.95, 0.90, f"{label_two} total: {cum_t_sum_two[-1]:.2f}", transform=plt.gca().transAxes,
             fontsize=10, color="orange", ha="right", va="top")
    plt.legend() 
    plt.savefig(os.path.join(output_dir, "T_Value_sum.svg"), format="svg")
    plt.close()




def evaluate(data_set, labels, train_indx, test_indx, metric='riemann', 
                        alpha=0.05, paired=False, permutations=10000, deconf=False, 
                        con_confounder_train=None, cat_confounder_train=None, 
                        con_confounder_test=None, cat_confounder_test=None, output_dir="path",
                        random_seed=42, basis="Method"):

    A, SpatialMaps, recon_error = data_set
    A_train = A[train_indx]
    A_test = A[test_indx]
    SpatialMaps_train = SpatialMaps[train_indx]
    SpatialMaps_test = SpatialMaps[test_indx]
    train_recon_error = recon_error[train_indx]
    test_recon_error = recon_error[test_indx]
    train_labels = labels[train_indx]
    test_labels = labels[test_indx]

    if deconf:
        n_maps = SpatialMaps_train.shape[1]
        # Loop over each spatial map (assumed along axis 1)
        for i in range(n_maps):
            # Extract the i-th spatial map for training and testing
            sm_train = SpatialMaps_train[:, i, :]  # shape: (n_subjects_train, spatial_dim)
            sm_test = SpatialMaps_test[:, i, :]    # shape: (n_subjects_test, spatial_dim)
            # Apply deconfounding on this map
            sm_train_dc, sm_test_dc = deconfound(
                sm_train,
                con_confounder_train,
                cat_confounder_train,
                X_test=sm_test,
                con_confounder_test=con_confounder_test,
                cat_confounder_test=cat_confounder_test
            )
            # Replace with deconfounded maps
            SpatialMaps_train[:, i, :] = sm_train_dc
            SpatialMaps_test[:, i, :] = sm_test_dc

    recon = (train_recon_error, test_recon_error)

    # Example: Calculate netmat from covariance estimator:
    cov_est = Covariances(estimator='oas')
    Netmats_train = cov_est.transform(np.transpose(A_train, (0, 2, 1)))
    Netmats_test = cov_est.transform(np.transpose(A_test, (0, 2, 1)))
    
    
    var_results = var_diff(A_train, Netmats_train, train_labels, A_test, test_labels, 
                           metric=metric, method='log-var', basis=basis, deconf=deconf, 
                           con_confounder_train=con_confounder_train, cat_confounder_train=cat_confounder_train, 
                           con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test,
                           output_dir=output_dir)
    
    cov_results = var_diff(A_train, Netmats_train, train_labels, A_test, test_labels, 
                           metric=metric, method='log-cov', basis=basis, deconf=deconf, 
                           con_confounder_train=con_confounder_train, cat_confounder_train=cat_confounder_train, 
                           con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test,
                           output_dir=output_dir)
    
    Class_Result = tangent_classification(Netmats_train, train_labels, Netmats_test, test_labels, 
                                          clf_str='all', z_score=0, metric=metric, deconf=deconf, 
                                          con_confounder_train=con_confounder_train, cat_confounder_train=cat_confounder_train, 
                                          con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test)

    t_test = tangent_t_test(Netmats_train, Netmats_test, test_labels, 
                                alpha=alpha, paired=paired, permutations=permutations, metric=metric, deconf=deconf, 
                                con_confounder_train=con_confounder_train, cat_confounder_train=cat_confounder_train, 
                                con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test, 
                                output_dir=output_dir, basis=basis,random_seed=random_seed)
    
    spatial_results = spatial_discrimination(SpatialMaps_train, train_labels, SpatialMaps_test, test_labels,
                                             methods=[2],metric=metric,visualize=True,
                                             outputfolder=output_dir,basis=basis)
    
    spatial_t_test_dir = os.path.join(output_dir,"spatial_T_test")
    if not os.path.exists(spatial_t_test_dir):
        os.makedirs(spatial_t_test_dir)

    spatial_t_test_results = spatial_analysis(SpatialMaps_test,test_labels,
                                              perm=permutations, alpha=alpha, 
                                              paired=paired, cluster=False, TFCE=None , all_maps=True, 
                                              random_seed=random_seed,output_dir=spatial_t_test_dir)
    
    spatial_t_test_discrim = os.path.join(output_dir,"spatial_T_test_discrim")
    if not os.path.exists(spatial_t_test_discrim):
        os.makedirs(spatial_t_test_discrim)
    
    U, _ = spatial_fda(SpatialMaps_train, train_labels,within=True)

    spatial_t_test_discrim_results = spatial_analysis(U.T@SpatialMaps_test,test_labels,
                                            perm=permutations, alpha=alpha, 
                                            paired=paired, cluster=False, TFCE=None , all_maps=True, 
                                            random_seed=random_seed,output_dir=spatial_t_test_discrim)
    
    results = {
        "var_results": var_results,
        "cov_results": cov_results,
        "Class_Result": Class_Result,
        "t_test": t_test,
        "recon": recon,
        "Spatial_discrim": spatial_results,
        "Spatial_t_test": spatial_t_test_results,
        "Spatial_t_test_discrim": spatial_t_test_discrim_results,

    }

    with open(os.path.join(output_dir, f"results.pkl"), "wb") as f:
        pickle.dump(results, f)

    return results

def compare(results_one, results_two, label_one="basis_one", label_two="basis_two", alpha=0.05, output_dir="path"):
    # Generate reconstruction plots
    reconstruction_plot(results_one["recon"][0], results_two["recon"][0], label1=f"Train {label_one}", label2=f"Train {label_two}", output_dir=output_dir)
    reconstruction_plot(results_one["recon"][1], results_two["recon"][1], label1=f"Test {label_one}", label2=f"Test {label_two}", output_dir=output_dir)

    # Scatter plots for FKT dimensions
    scatter_with_lines(results_one["var_results"][:, [0, 2]], results_two["var_results"][:, [0, 2]], 
                       label1=label_one, label2=label_two, 
                       xlabel='Number of FKT Filters', ylabel='SVM Accuracy', 
                       title='Accuracies_Across_FKT_Dimensions_(log-var)',
                       output_dir=output_dir)

    scatter_with_lines(results_one["var_results"][:, [0, 1]], results_two["var_results"][:, [0, 1]], 
                       label1=label_one, label2=label_two, 
                       xlabel='Number of FKT Filters', ylabel='Riemannian Distance', 
                       title='Distance_of_Group_Means_Across_FKT_Dimensions_(log-var)',
                       output_dir=output_dir)
    
    # Repeat for log-cov method
    scatter_with_lines(results_one["cov_results"][:, [0, 2]], results_two["cov_results"][:, [0, 2]], 
                       label1=label_one, label2=label_two, 
                       xlabel='Number of FKT Filters', ylabel='SVM Accuracy', 
                       title='Accuracies_Across_FKT_Dimensions_(log-cov)',
                       output_dir=output_dir)
    
    scatter_with_lines(results_one["cov_results"][:, [0, 1]], results_two["cov_results"][:, [0, 1]], 
                       label1=label_one, label2=label_two, 
                       xlabel='Number of FKT Filters', ylabel='Riemannian Distance', 
                       title='Distance_of_Group_Means_Across_FKT_Dimensions_(log-cov)',
                       output_dir=output_dir)

    # Scatter plot for classifier accuracies
    scatter_with_lines(results_one["Class_Result"], results_two["Class_Result"], 
                       label1=label_one, label2=label_two, 
                       xlabel='Classifiers', ylabel='Accuracies', 
                       title='Netmat_Tangent_Classifier_Accuracies',
                       output_dir=output_dir)
    
    spatial_comparison_vis(results_one["Spatial_discrim"], results_two["Spatial_discrim"], label_one=label_one, label_two=label_two, outputfolder=output_dir, basis=f"{label_one}_vs_{label_two}")
    
    spatial_t_test_dir_compare = os.path.join(output_dir,"spatial_T_test_compare")
    if not os.path.exists(spatial_t_test_dir_compare):
        os.makedirs(spatial_t_test_dir_compare)
    spatial_t_compare(results_one["Spatial_t_test"], results_two["Spatial_t_test"], label_one=label_one, label_two=label_two, alpha=alpha,output_dir=spatial_t_test_dir_compare)


    spatial_t_test_discrim_dir_compare = os.path.join(output_dir,"spatial_T_test_discrim_compare")
    if not os.path.exists(spatial_t_test_discrim_dir_compare):
        os.makedirs(spatial_t_test_discrim_dir_compare)
    spatial_t_compare(results_one["Spatial_t_test_discrim"], results_two["Spatial_t_test_discrim"], label_one=label_one, label_two=label_two, alpha=alpha,output_dir=spatial_t_test_discrim_dir_compare)