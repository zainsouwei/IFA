import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import untangent_space
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('/utils')

from tangent import tangent_transform, clf_dict
from haufe import haufe_transform

def FKT(groupA_cov_matrices, groupB_cov_matrices, metric="riemann", visualize=True):
    # Eigenvalues in ascending order from scipy eigh https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
    groupA_mean_cov= mean_covariance(groupA_cov_matrices, metric=metric)
    groupB_mean_cov = mean_covariance(groupB_cov_matrices, metric=metric)    

    eigsA, filtersA  = eigh(groupA_mean_cov, groupA_mean_cov + groupB_mean_cov,eigvals_only=False,subset_by_value=[0.5,np.inf])
    eigsB, filtersB = eigh(groupB_mean_cov, groupA_mean_cov + groupB_mean_cov,eigvals_only=False,subset_by_value=[0.5,np.inf])
       
    eigs = np.concatenate((eigsB[::-1], eigsA))
    filters = np.concatenate((filtersB[:, ::-1], filtersA), axis=1)

    # Transform Eigenvalues to Approximate Riemannian Distance https://ieeexplore.ieee.org/document/5662067
    fkt_riem_eigs = np.abs(np.log(eigs/(1-eigs)))**2
    
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.scatter(range(0,fkt_riem_eigs.shape[0]),fkt_riem_eigs)
        plt.title("Riemannian Distance Supported by Spatial Filter")
        plt.xlabel("Max Eigenvector for Group B to Max Eigenvector for Group A")
        plt.ylabel(r"$|\log\left(\frac{\lambda}{1 - \lambda}\right)|^2$")
        plt.show()

    return fkt_riem_eigs, filters, filtersA, filtersB

def TSSF(groupA_covs, groupB_covs, clf=clf_dict["L2 SVM (C=1)"], metric="riemann", z_score=0, haufe=True, visualize=False,n=0):
    # https://ieeexplore.ieee.org/abstract/document/9630144/references#references
    # https://arxiv.org/abs/1909.10567
    covs = np.concatenate((groupA_covs, groupB_covs))
    labels = np.concatenate((np.ones(groupA_covs.shape[0]) , np.zeros(groupB_covs.shape[0])))
    data, Frechet_Mean = tangent_transform(covs,metric=metric)

    if z_score == 1:
        scaler = StandardScaler(with_mean=True, with_std=False)
        data = scaler.fit_transform(data)
    elif z_score == 2:
        scaler = StandardScaler(with_mean=True, with_std=True)
        data = scaler.fit_transform(data)

    clf.fit(data, labels)

    coef = np.atleast_2d(clf.coef)
    if coef.shape[1] != data.shape[1]:
        coef = coef.T

    if haufe:
        coef = haufe_transform(data, clf.coef_.T,method="basic")

    boundary_matrix = untangent_space(coef, Frechet_Mean)[0,:,:]
    fkt_eigs, filters  = eigh(boundary_matrix, Frechet_Mean)

    # Compute the differences between consecutive eigenvalues
    differences = np.diff(np.abs(fkt_eigs))

    # Use argmax to find the index of the maximum difference (inflection point)
    inflection_point = np.argmax(differences) + 1  # +1 to adjust for the index offset from np.diff

    # Partition filters
    filtersA = filters[:,:inflection_point]  # First half up to the inflection point
    filtersB = filters[:,inflection_point:]  # Second half beyond the inflection point
        
    if visualize:
        plt.scatter(range(0,fkt_eigs.shape[0]),np.abs(fkt_eigs))
        plt.title("Riemannian Distance Supported by Spatial Filter")
        plt.xlabel("Max Eigenvector for Group B to Max Eigenvector for Group A")
        plt.ylabel(r"$|\lambda|$")
        plt.show()

        return fkt_eigs, filters, filtersA, filtersB
