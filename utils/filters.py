import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import untangent_space
from pyriemann.estimation import Covariances
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec

import sys
sys.path.append('/utils')

from tangent import tangent_transform
from classification import linear_classifier, clf_dict
from haufe import haufe_transform
from regression import deconfound

def feature_generation(train,test, filters,method='log-var',metric='riemann',cov="oas"):
    train_transformed = train @ filters
    test_transformed = test @ filters

    if method == 'log-var':
        train_features = np.log(np.var(train_transformed, axis=1))
        test_features = np.log(np.var(test_transformed, axis=1))
    
    elif method == 'log-cov':
        cov_est = Covariances(estimator=cov)
        train_cov = cov_est.transform(np.transpose(train_transformed, (0, 2, 1)))
        test_cov = cov_est.transform(np.transpose(test_transformed, (0, 2, 1)))
        train_features, test_features, _ = tangent_transform(train_cov,  test_cov, metric)

    return train_features, test_features

def test_filters(train, train_labels, test, test_labels, filters, metric="riemann", method='log-cov',clf_str="SVM (C=1)", deconf=False,con_confounder_train=None, cat_confounder_train=None, con_confounder_test=None, cat_confounder_test=None):
    train_features, test_features = feature_generation(train, test, filters, method=method,metric=metric)
    if deconf:
        train_features, test_features = deconfound(train_features, con_confounder_train, cat_confounder_train, X_test=test_features, con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test, age_var="Age_in_Yrs", sex_var="Gender")
    accuracy = linear_classifier(train_features, train_labels, test_features, test_labels, clf_str=clf_str, z_score=2)
    return accuracy

def test_visualize_variance(data, labels, filters):
    for i in range(0,filters.shape[1]//2):
        data_transform = np.var(data@filters[:,[i,-(i+1)]],axis=1)
        unique_labels = np.unique(labels)

        # Visualize variance based on the unique labels
        if len(unique_labels) == 2:  # Assuming binary classification
            data_1_transform = data_transform[labels == unique_labels[0]]
            data_2_transform = data_transform[labels == unique_labels[1]]
            
        # Create figure and gridspec layout
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(4, 4)

        # Define the axes
        ax_scatter = fig.add_subplot(gs[1:4, 0:3])
        ax_hist_x = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)
        ax_hist_y = fig.add_subplot(gs[1:4, 3], sharey=ax_scatter)

        # Scatter plot
        ax_scatter.scatter(data_1_transform[:, 0], data_1_transform[:, 1], label=f'Group {unique_labels[0]}', color='blue', alpha=0.5)
        ax_scatter.scatter(data_2_transform[:, 0], data_2_transform[:, 1], label=f'Group {unique_labels[1]}', color='red', alpha=0.5)
        ax_scatter.set_xlabel(f'Projection onto Filter {i}')
        ax_scatter.set_ylabel(f'Projection onto Filter {filters.shape[1]-(i+1)}')
        ax_scatter.legend()
        ax_scatter.grid(True)

        # Histograms
        bins = 30

        # Histograms for X axis (top)
        ax_hist_x.hist(data_1_transform[:, 0], bins=bins, color='blue', alpha=0.5, density=True, label=f'Group {unique_labels[0]}')
        ax_hist_x.hist(data_2_transform[:, 0], bins=bins, color='red', alpha=0.5, density=True, label=f'Group {unique_labels[1]}')
        ax_hist_x.set_ylabel('Density')
        ax_hist_x.legend()
        ax_hist_x.grid(True)

        # Histograms for Y axis (right)
        ax_hist_y.hist(data_1_transform[:, 1], bins=bins, orientation='horizontal', color='blue', alpha=0.5, density=True)
        ax_hist_y.hist(data_2_transform[:, 1], bins=bins, orientation='horizontal', color='red', alpha=0.5, density=True)
        ax_hist_y.set_xlabel('Density')
        ax_hist_y.grid(True)

        # Hide tick labels on histograms to avoid clutter
        plt.setp(ax_hist_x.get_xticklabels(), visible=False)
        plt.setp(ax_hist_y.get_yticklabels(), visible=False)

        # Adjust layout
        plt.tight_layout()
        plt.show()

def evaluate_filters(train, train_labels, test, test_labels, filters, metric="riemann", deconf=False, con_confounder_train=None, cat_confounder_train=None, con_confounder_test=None, cat_confounder_test=None):

    test_visualize_variance(test, test_labels, filters)
    metrics_dict_logvar = test_filters(train, train_labels, test, test_labels, filters, metric=metric, method='log-var',clf_str='all')
    metrics_dict_logcov = test_filters(train, train_labels, test, test_labels, filters, metric=metric, method='log-cov',clf_str='all')
    if deconf:
        metrics_dict_logvar = test_filters(train, train_labels, test, test_labels, filters, metric=metric, method='log-var',clf_str='all', deconf=deconf,con_confounder_train=con_confounder_train, cat_confounder_train=cat_confounder_train, con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test)
        metrics_dict_logcov = test_filters(train, train_labels, test, test_labels, filters, metric=metric, method='log-cov',clf_str='all', deconf=deconf,con_confounder_train=con_confounder_train, cat_confounder_train=cat_confounder_train, con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test)

    return metrics_dict_logvar, metrics_dict_logcov

def FKT(cov_matrices, labels, metric="riemann", deconf=True, con_confounder_train=None, cat_confounder_train=None, visualize=True):
    # Eigenvalues in ascending order from scipy eigh https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
    unique_labels = np.unique(labels)

    if deconf:
        data, Frechet_Mean = tangent_transform(cov_matrices,metric=metric)
        data = deconfound(data, con_confounder_train, cat_confounder_train, X_test=None, con_confounder_test=None, cat_confounder_test=None, age_var="Age_in_Yrs", sex_var="Gender")
        groupA_cov_matrices_deconf = untangent_space(data[labels == unique_labels[1]],Frechet_Mean,metric=metric)
        groupB_cov_matrices_deconf = untangent_space(data[labels == unique_labels[0]],Frechet_Mean,metric=metric)
        groupA_mean_cov= mean_covariance(groupA_cov_matrices_deconf, metric=metric)
        groupB_mean_cov = mean_covariance(groupB_cov_matrices_deconf, metric=metric)
    else:
        groupA_mean_cov= mean_covariance(cov_matrices[labels == unique_labels[1]], metric=metric)
        groupB_mean_cov = mean_covariance(cov_matrices[labels == unique_labels[0]], metric=metric)

    eigsA, filtersA  = eigh(groupA_mean_cov, groupA_mean_cov + groupB_mean_cov,eigvals_only=False,subset_by_value=[0.5,np.inf])
    eigsB, filtersB = eigh(groupB_mean_cov, groupA_mean_cov + groupB_mean_cov,eigvals_only=False,subset_by_value=[0.5,np.inf])
       
    eigs = np.concatenate((eigsB[::-1], eigsA))
    filters = np.concatenate((filtersB[:, ::-1], filtersA), axis=1)

    # Transform Eigenvalues to Approximate Riemannian Distance https://ieeexplore.ieee.org/document/5662067
    # TODO Specific for AIR and is support of distance not distance, make equivalance between TSSF calculation and FKT
    fkt_riem_eigs = np.abs(np.log(eigs/(1-eigs)))**2
    
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.scatter(range(0,fkt_riem_eigs.shape[0]),fkt_riem_eigs)
        plt.title("Riemannian Distance Supported by Spatial Filter")
        plt.xlabel("Max Eigenvector for Group B to Max Eigenvector for Group A")
        plt.ylabel(r"$|\log\left(\frac{\lambda}{1 - \lambda}\right)|^2$")
        plt.show()

    return fkt_riem_eigs, filters

def TSSF(covs, labels, clf_str="L2 SVM (C=1)", metric="riemann", deconf=True, con_confounder_train=None, cat_confounder_train=None, z_score=2, haufe=True, visualize=False):
    clf = clf_dict[clf_str]
    # https://ieeexplore.ieee.org/abstract/document/9630144/references#references
    # https://arxiv.org/abs/1909.10567
    data, Frechet_Mean = tangent_transform(covs,metric=metric)
    
    if deconf:
        data = deconfound(data, con_confounder_train, cat_confounder_train, X_test=None, con_confounder_test=None, cat_confounder_test=None, age_var="Age_in_Yrs", sex_var="Gender")

    if z_score == 1:
        scaler = StandardScaler(with_mean=True, with_std=False)
        data = scaler.fit_transform(data)
    elif z_score == 2:
        scaler = StandardScaler(with_mean=True, with_std=True)
        data = scaler.fit_transform(data)

    clf.fit(data, labels)

    coef = np.atleast_2d(clf.coef_)
    if coef.shape[1] != data.shape[1]:
        coef = coef.T

    if haufe:
        coef = haufe_transform(data, clf.coef_.T,method="basic")

    boundary_matrix = untangent_space(coef, Frechet_Mean)[0,:,:]
    eigs, filters  = eigh(boundary_matrix, Frechet_Mean)
    # TODO Dffferent transformation functions from each paper
    # fkt_riem_eigs = np.maximum(eigs,1/eigs)
    # TODO Specific for Log Euclidean case
    # Equivalent to the 2 norm of the distance to reference eigenvalue on the tangent space
    # sqrt((log(lamda) - log(1))^2) == abs(log(lamda) - log(1)) == abs(log(lambda))
    riem_eig = np.abs(np.log(eigs))

    if visualize:
        plt.scatter(range(0,riem_eig.shape[0]),riem_eig)
        plt.title("Riemannian Distance Supported by Spatial Filter")
        plt.xlabel("Max Eigenvector for Group B to Max Eigenvector for Group A")
        plt.ylabel(r"$|log(\lambda)|$")
        plt.show()

    return riem_eig, filters
