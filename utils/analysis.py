import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from matplotlib.colors import LinearSegmentedColormap
from mne_connectivity.viz import plot_connectivity_circle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy.linalg import eigh
from pyriemann.utils.tangentspace import untangent_space, mean_covariance
from sklearn.metrics import accuracy_score

import sys
sys.path.append('/utils')

from tangent import tangent_transform
from filters import feature_generation
from tangent import tangent_classification

def scatter_with_lines(data1, data2, label1='Series 1', label2='Series 2', xlabel='X', ylabel='Y', title='Scatter Plot with Connecting Lines'):
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
    plt.show()

def unupper_noweighting(vec):
    # Compute the matrix size `n`
    n = int((np.sqrt(1 + 8 * len(vec)) - 1) / 2)
    full_matrix = np.zeros((n, n), dtype=vec.dtype)
    upper_indices = np.triu_indices(n)
    full_matrix[upper_indices] = vec
    i_lower = np.tril_indices(n, -1)
    full_matrix[i_lower] = full_matrix.T[i_lower]
    return full_matrix

def tangent_t_test(train_covs, test_covs, test_labels, alpha=.05, permutations=False, correction='fdr_bh', metric='riemannian'):
    unique_labels = np.unique(test_labels)
    _, test_vecs, mean = tangent_transform(train_covs, test_covs, metric=metric)
    t_values, p_values = ttest_ind(test_vecs[test_labels==unique_labels[1]], test_vecs[test_labels==unique_labels[0]], axis=0, permutations=permutations)
    reject, corrected_p_values, _, _ = multipletests(p_values, alpha=alpha, method=correction)
    
    if np.sum(reject) == 0:
        print("No significant differences")
        return
    
    t_values_thresholded = t_values * reject
    diff_thresholded = corrected_p_values * reject
    diff_thresholded[diff_thresholded == 0.0] = alpha + 1e-5
    
    diff_thresholded_matrix = unupper_noweighting(diff_thresholded)
    t_values_thresholded_matrix = unupper_noweighting(t_values_thresholded)
    groupA = untangent_space(np.mean(test_vecs[test_labels==unique_labels[1]], axis=0)[np.newaxis, :], mean, metric=metric)[0, :, :] * unupper_noweighting(reject)
    groupB = untangent_space(np.mean(test_vecs[test_labels==unique_labels[0]], axis=0)[np.newaxis, :], mean, metric=metric)[0, :, :] * unupper_noweighting(reject)
    
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
    im1 = plot_connectivity_circle(np.where(diff_thresholded_matrix < alpha, diff_thresholded_matrix, np.nan), labels, facecolor=(0.2, 0.2, 0.2, 1), colormap=green_cmap, vmin=diff_thresholded_matrix.min(), vmax=alpha, fig=fig, ax=ax1, show=False, colorbar=False)
    ax1.set_title("Corrected P-values", fontsize=12)
    cbar1 = fig.colorbar(plt.cm.ScalarMappable(cmap=green_cmap, norm=plt.Normalize(vmin=diff_thresholded_matrix.min(), vmax=alpha)), ax=ax1, shrink=0.7, orientation='vertical')
    cbar1.set_label('Corrected P-values', fontsize=10)

    # Plot 2: T-values
    im2 = plot_connectivity_circle(np.where(np.abs(t_values_thresholded_matrix) > 0, t_values_thresholded_matrix, np.nan), labels, facecolor=(0.2, 0.2, 0.2, 1), colormap=custom_cmap, vmin=t_values_thresholded_matrix.min(), vmax=t_values_thresholded_matrix.max(), fig=fig, ax=ax2, show=False, colorbar=False)
    ax2.set_title("T-values", fontsize=12)
    cbar2 = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=t_values_thresholded_matrix.min(), vmax=t_values_thresholded_matrix.max())), ax=ax2, shrink=0.7, orientation='vertical')
    cbar2.set_label('Tangent T-values', fontsize=10)

    # Plot 3: Group A Connectivity
    im3 = plot_connectivity_circle(np.where(np.abs(groupA) > 0, groupA, np.nan), labels, facecolor=(0.2, 0.2, 0.2, 1), colormap=custom_cmap, vmin=groupA.min(), vmax=groupA.max(), fig=fig, ax=ax3, show=False, colorbar=False)
    ax3.set_title(f'Group {unique_labels[1]} Connectivity', fontsize=12)
    cbar3 = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=groupA.min(), vmax=groupA.max())), ax=ax3, shrink=0.7, orientation='vertical')
    cbar3.set_label('Covariance', fontsize=10)

    # Plot 4: Group B Connectivity
    im4 = plot_connectivity_circle(np.where(np.abs(groupB) > 0, groupB, np.nan),labels, facecolor=(0.2, 0.2, 0.2, 1), colormap=custom_cmap, vmin=groupB.min(), vmax=groupB.max(), fig=fig, ax=ax4, show=False, colorbar=False)
    ax4.set_title(f'Group {unique_labels[0]} Connectivity', fontsize=12)
    cbar4 = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=groupB.min(), vmax=groupB.max())),ax=ax4, shrink=0.7, orientation='vertical')
    cbar4.set_label('Covariance', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Create heatmaps in subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Heatmaps of Connectivity Differences", fontsize=14)
    sns.heatmap(diff_thresholded_matrix, cmap=green_cmap, vmin=diff_thresholded_matrix.min(), vmax=alpha, cbar_kws={'label': 'Corrected P-values'}, ax=axes[0, 0])
    axes[0, 0].set_title("Corrected P-values")
    sns.heatmap(t_values_thresholded_matrix, cmap=custom_cmap, center=0, cbar_kws={'label': 'Tangent T-values'}, ax=axes[0, 1])
    axes[0, 1].set_title(f'T-values on Tangent Plane | {np.linalg.norm(t_values_thresholded_matrix)}')
    sns.heatmap(groupA, cmap=custom_cmap, vmin=global_min, vmax=global_max, center=0, cbar_kws={'label': 'Covariance'}, ax=axes[1, 0])
    axes[1, 0].set_title(f'Group {unique_labels[1]} Mean Connectivity')
    sns.heatmap(groupB, cmap=custom_cmap, vmin=global_min, vmax=global_max, center=0, cbar_kws={'label': 'Covariance'}, ax=axes[1, 1])
    axes[1, 1].set_title(f'Group {unique_labels[0]} Mean Connectivity')
    plt.tight_layout()
    plt.show()

# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5662067
def var_diff(train_data, train_covs, train_labels, test_data, test_labels, metric='riemann', method='log-var', basis="ICA"):
    unique_labels = np.unique(train_labels)
    clf = SVC(kernel='linear', C=0.1, class_weight='balanced')

    # Compute the mean covariances using the training data only
    group1_mean = mean_covariance(train_covs[train_labels==unique_labels[1]], metric=metric)
    group2_mean = mean_covariance(train_covs[train_labels==unique_labels[0]], metric=metric)

    # Genearlized Eigendecomposition (FKT/SPADE/CSP/GEVD) to get features
    _, filters_all = eigh(group1_mean, group2_mean + group2_mean, eigvals_only=False)

    # Initialize list to store results (accuracy and distance)
    results = []
    for n in range(1, filters_all.shape[1] // 2 + 1): 
        filters = np.hstack([filters_all[:, :n], filters_all[:, -n:]])  # Select top and bottom n eigenvectors
        train_features, test_features = feature_generation(train_data,test_data, filters,method=method,metric=metric,cov="oas")

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
            plt.scatter(test_features[test_labels==unique_labels[1]][:, 0], test_features[test_labels==unique_labels[1]][:, 1], label=f'Group {unique_labels[1]} Log Variance (Test)', color='blue')
            plt.scatter(test_features[test_labels==unique_labels[0]][:, 0], test_features[test_labels==unique_labels[0]][:, 1], label=f'Group {unique_labels[0]} Log Variance (Test)', color='red')

            # Plot the line connecting the two means
            plt.plot([mean_group1_test[0], mean_group2_test[0]], [mean_group1_test[1], mean_group2_test[1]], 'k--', label=f'Mean Distance: {mean_dist:.2f}')

            # Decision boundary
            x_values = np.array([train_features[:, 0].min(), train_features[:, 0].max()])
            y_values = -(clf.intercept_ + clf.coef_[0][0] * x_values) / clf.coef_[0][1]
            plt.plot(x_values, y_values, 'g-', label='Decision Boundary')

            # Display plot
            plt.xlabel(f'Log Variance Feature {unique_labels[0]}')
            plt.ylabel(f'Log Variance Feature {unique_labels[1]}')
            plt.title(f'{basis} Log Variance FKT Feature Comparison and SVM Decision Boundary')
            plt.text(0.05, 0.95, f'Accuracy: {accuracy:.2f}', transform=plt.gca().transAxes, fontsize=12,verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='lightgrey'))
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()

    return np.array(results)

def evaluate_IFA_results(IFA, ICA, train_labels, test_labels, alpha=.05, permutations=False, correction='fdr_bh', metric='riemannian'):
    IFA_A_train, IFA_Netmats_train, IFA_A_test, IFA_Netmats_test = IFA
    ICA_A_train, ICA_Netmats_train, ICA_A_test, ICA_Netmats_test = ICA
    
    IFA_var_results = var_diff(IFA_A_train, IFA_Netmats_train, train_labels, IFA_A_test, test_labels, metric=metric, method='log-var' basis="IFA")
    ICA_var_results = var_diff(ICA_A_train, ICA_Netmats_train, train_labels, ICA_A_test, test_labels, metric=metric, method='log-var' basis="ICA")
    scatter_with_lines(IFA_var_results[:, [0, 2]], ICA_var_results[:, [0, 2]], label1='IFA', label2='ICA', xlabel='Number of FKT Filters', ylabel='SVM Accuracy', title='Accuracies Across FKT Dimensions (log-var)')
    scatter_with_lines(IFA_var_results[:, [0, 1]], ICA_var_results[:, [0, 1]], label1='IFA', label2='ICA', xlabel='Number of FKT Filters', ylabel='Riemannian Distance', title='Distance of Group Means Across FKT Dimensions  (log-var)')

    IFA_var_results = var_diff(IFA_A_train, IFA_Netmats_train, train_labels, IFA_A_test, test_labels, metric=metric, method='log-cov', basis="IFA")
    ICA_var_results = var_diff(ICA_A_train, ICA_Netmats_train, train_labels, ICA_A_test, test_labels, metric=metric, method='log-cov', basis="ICA")
    scatter_with_lines(IFA_var_results[:, [0, 2]], ICA_var_results[:, [0, 2]], label1='IFA', label2='ICA', xlabel='Number of FKT Filters', ylabel='SVM Accuracy', title='Accuracies Across FKT Dimensions (log-cov)')
    scatter_with_lines(IFA_var_results[:, [0, 1]], ICA_var_results[:, [0, 1]], label1='IFA', label2='ICA', xlabel='Number of FKT Filters', ylabel='Riemannian Distance', title='Distance of Group Means Across FKT Dimensions (log-cov)')

    IFA_Class_Result = tangent_classification(IFA_Netmats_train, train_labels, IFA_Netmats_test, test_labels, clf_str='all', z_score=0, metric=metric, deconf=False)
    ICA_Class_Result = tangent_classification(ICA_Netmats_train, train_labels, ICA_Netmats_test, test_labels, clf_str='all', z_score=0, metric=metric, deconf=False)
    scatter_with_lines(IFA_Class_Result, ICA_Class_Result, label1='IFA', label2='ICA', xlabel='Classifiers', ylabel='Accuracies', title='Netmat Tangent Classifier Accuracies')
    tangent_t_test(IFA_Netmats_train, IFA_Netmats_test,test_labels, alpha=alpha, permutations=permutations, correction=correction, metric=metric)
    tangent_t_test(ICA_Netmats_train, ICA_Netmats_test,test_labels, alpha=alpha, permutations=permutations, correction=correction, metric=metric)