import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import untangent_space, unupper
from pyriemann.estimation import Covariances
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from skopt.utils import use_named_args
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score
from joblib import Parallel, delayed

import matplotlib.gridspec as gridspec
from scipy.linalg import svd
import torch
import jax.numpy as jnp
from pymanopt.manifolds import Stiefel, Grassmann, Sphere
from pymanopt.optimizers import ConjugateGradient, TrustRegions
from pymanopt import Problem
import pymanopt
import optax
import gc

import sys
sys.path.append('/utils')

from tangent import tangent_transform
from classification import linear_classifier, clf_dict
from haufe import haufe_transform
from regression import deconfound
from preprocessing import gpu_mem, cpu_mem

def feature_generation(train, test, filters, method='log-var', metric='riemann', cov="oas"):
    # Transform each subject individually
    train_transformed = [subj @ filters for subj in train]
    test_transformed  = [subj @ filters for subj in test]

    if method == 'log-var':
        # Compute log-variance feature for each subject
        train_features = np.array([np.log(np.var(subj, axis=0)) for subj in train_transformed])
        test_features  = np.array([np.log(np.var(subj, axis=0)) for subj in test_transformed])
    
    elif method == 'log-cov':
        # Compute covariances one subject at a time
        cov_est = Covariances(estimator=cov)
        train_cov = np.array([ cov_est.transform(subj.T[np.newaxis, :, :])[0] for subj in train_transformed ])
        test_cov = np.array([ cov_est.transform(subj.T[np.newaxis, :, :])[0] for subj in test_transformed ])
        train_features, test_features, _ = tangent_transform(train_cov, test_cov, metric)

    return train_features, test_features

# TODO decide on z scoring projection 
def test_filters(train, train_labels, test, test_labels, filters, metric="riemann", method='log-cov', deconf=False, con_confounder_train=None, cat_confounder_train=None, con_confounder_test=None, cat_confounder_test=None, cv_splits=5, random_state=0,z_score=2):
    make_final = clf_dict["svc"]["make"]
    C_grid = np.logspace(-6, 3, 10)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    def eval_C_on_fold(C, tr_idx, val_idx):
        tr_train = [train[i] for i in tr_idx]; tr_val = [train[i] for i in val_idx]
        Xtr, Xval = feature_generation(tr_train, tr_val, filters, method=method, metric=metric)
        if deconf:
            ctr  = None if con_confounder_train is None else con_confounder_train.iloc[tr_idx]
            cvr  = None if con_confounder_train is None else con_confounder_train.iloc[val_idx]
            cta  = None if cat_confounder_train is None else cat_confounder_train.iloc[tr_idx]
            cva  = None if cat_confounder_train is None else cat_confounder_train.iloc[val_idx]
            Xtr, Xval = deconfound(Xtr, ctr, cta, X_test=Xval, con_confounder_test=cvr, cat_confounder_test=cva)

        ytr, yval = train_labels[tr_idx], train_labels[val_idx]
        out = linear_classifier(Xtr, ytr, Xval, yval, clfs_list=[make_final(C=C)], z_score=z_score)        
        yhat = next(iter(out.values()))['predictions']
        return balanced_accuracy_score(yval, yhat)

    def score_C(C):
        folds = list(skf.split(train, train_labels))
        scores = Parallel(n_jobs=cv_splits)(
            delayed(eval_C_on_fold)(C, tr, va) for tr, va in folds
        )
        return np.mean(scores)

    C_scores = np.array([score_C(C) for C in C_grid])
    C_best = C_grid[int(np.argmax(C_scores))]

    # Final refit on full TRAIN (apply deconf train→test once)
    Xtr_full, Xte_full = feature_generation(train, test, filters, method=method, metric=metric)
    if deconf:
        Xtr_full, Xte_full = deconfound(Xtr_full, con_confounder_train, cat_confounder_train,
                                        X_test=Xte_full, con_confounder_test=con_confounder_test,
                                        cat_confounder_test=cat_confounder_test)
    best_model = make_final(C=C_best)
    return linear_classifier(Xtr_full, train_labels, Xte_full, test_labels, clfs_list=[best_model], z_score=z_score), C_best


def test_visualize_variance(train, test, test_labels, filters, deconf=False, con_confounder_train=None, cat_confounder_train=None, con_confounder_test=None, cat_confounder_test=None,  output_dir="plots", metric="riemann", cov="oas"):
    os.makedirs(output_dir, exist_ok=True)

    # Generate features like in test_filters
    train_features, test_features = feature_generation(train, test, filters, method="log-var", metric=metric, cov=cov)
    
    # Deconfound if needed
    if deconf:
        train_features, test_features = deconfound(train_features, con_confounder_train, cat_confounder_train, X_test=test_features, con_confounder_test=con_confounder_test,cat_confounder_test=cat_confounder_test)

    # undo log
    test_features = np.exp(test_features)

    for i in range(0, filters.shape[1] // 2):
        test_features_iter = test_features[:,[i,-(i+1)]]
        unique_labels = np.unique(test_labels)
        data_1_transform = test_features_iter[test_labels == unique_labels[0]]
        data_2_transform = test_features_iter[test_labels == unique_labels[1]]

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
        plt.savefig(os.path.join(output_dir, f"Filter_{i}_{filters.shape[1]-(i+1)}_var.svg"))
        plt.close('all')


def evaluate_filters(train, train_labels, test, test_labels, filters, metric="riemann", deconf=False, con_confounder_train=None, cat_confounder_train=None, con_confounder_test=None, cat_confounder_test=None,output_dir="plots"):
    test_visualize_variance(train, test, test_labels, filters,  deconf=deconf,con_confounder_train=con_confounder_train, cat_confounder_train=cat_confounder_train, con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test,  output_dir=output_dir, metric=metric, cov="oas")
    metrics_dict_logvar, _ = test_filters(train, train_labels, test, test_labels, filters, metric=metric, method='log-var', deconf=deconf,con_confounder_train=con_confounder_train, cat_confounder_train=cat_confounder_train, con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test)
    metrics_dict_logcov, _ = test_filters(train, train_labels, test, test_labels, filters, metric=metric, method='log-cov', deconf=deconf,con_confounder_train=con_confounder_train, cat_confounder_train=cat_confounder_train, con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test)

    return metrics_dict_logvar, metrics_dict_logcov

def FKT(cov_matrices, labels, a_label, b_label, metric="riemann", deconf=True, con_confounder_train=None, cat_confounder_train=None, visualize=True,output_dir="plots"):
    # Eigenvalues in ascending order from scipy eigh https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html

    if deconf:
        data, Frechet_Mean = tangent_transform(cov_matrices,metric=metric)
        data = deconfound(data, con_confounder_train, cat_confounder_train, X_test=None, con_confounder_test=None, cat_confounder_test=None)
        groupA_cov_matrices_deconf = untangent_space(data[labels == a_label],Frechet_Mean,metric=metric)
        groupB_cov_matrices_deconf = untangent_space(data[labels == b_label],Frechet_Mean,metric=metric)
        groupA_mean_cov= mean_covariance(groupA_cov_matrices_deconf, metric=metric)
        groupB_mean_cov = mean_covariance(groupB_cov_matrices_deconf, metric=metric)
    else:
        groupA_mean_cov= mean_covariance(cov_matrices[labels == a_label], metric=metric)
        groupB_mean_cov = mean_covariance(cov_matrices[labels == b_label], metric=metric)

    eigsA, filtersA  = eigh(groupA_mean_cov, groupA_mean_cov + groupB_mean_cov,eigvals_only=False,subset_by_value=[0.5,np.inf])
    eigsB, filtersB = eigh(groupB_mean_cov, groupA_mean_cov + groupB_mean_cov,eigvals_only=False,subset_by_value=[0.5,np.inf])
       
    eigs = np.concatenate((eigsB[::-1], eigsA))
    filters = np.concatenate((filtersB[:, ::-1], filtersA), axis=1)

    # Transform Eigenvalues to Approximate Riemannian Distance https://ieeexplore.ieee.org/document/5662067
    # Specific for AIR and is support of distance not distance
    # As noted in the TSSF function and appendix of tssf paper, eigenvalues manifold representation can be calculated by 
    # doing the eigh in the tangent space. Classic FKT is linked to AIR so this transformation holds
    fkt_riem_eigs = np.abs(np.log(eigs/(1-eigs)))**2
    
    if visualize:
        plt.figure(figsize=(10, 5))
        plt.scatter(range(0,fkt_riem_eigs.shape[0]),fkt_riem_eigs)
        plt.title("Riemannian Distance Supported by Spatial Filter")
        plt.xlabel("Max Eigenvector for Group B to Max Eigenvector for Group A")
        plt.ylabel(r"$|\log\left(\frac{\lambda}{1 - \lambda}\right)|^2$")
        plt.savefig(os.path.join(output_dir, "fkt_scree.svg"))
        plt.close('all')


    return fkt_riem_eigs, filters


def TSSF_select(covs, labels, train_data, a_label, b_label, n=1,
                metric="riemann", feature_kind="log-var",                
                deconf=False, con_confounder_train=None, cat_confounder_train=None,
                tan_model_keys=("logreg_en","svc_l2_sq"), final_svm_key="svc",                              
                z_score_tan=0, haufe=False,                             
                n_inner_splits=5, n_calls=25, n_initial=6, random_state=0):
    
    C_grid = np.logspace(-6, 3, 8)

    skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)

    def _fit_filters(covs_tr, y_tr, model_key, params, con_tr=None, cat_tr=None):
        X_tr, _ = tangent_transform(covs_tr, metric=metric)
        if deconf: 
            X_tr = deconfound(X_tr, con_tr, cat_tr)
        if z_score_tan == 1:
            X_tr = StandardScaler(with_mean=True, with_std=False).fit_transform(X_tr)
        elif z_score_tan == 2:
            X_tr = StandardScaler(with_mean=True, with_std=True ).fit_transform(X_tr)

        make_clf = clf_dict[model_key]["make"]
        clf = make_clf(**params).fit(X_tr, y_tr)
        coef = np.atleast_2d(clf.coef_)
        
        if coef.shape[1] != X_tr.shape[1]:
            coef = coef.T

        if haufe:
            coef = haufe_transform(X_tr, coef.T, method="basic")

        boundary = unupper(np.atleast_2d(coef))[0, :, :]
        _, V = eigh(boundary)                      # ascending eigs

        if a_label < b_label:
            B, A = V[:, -n:], V[:, :n]
        else:
            A, B = V[:, -n:], V[:, :n]
        
        return np.concatenate((B, A), axis=1)

    def _fold_confs(idx):
        con = None if con_confounder_train is None else con_confounder_train.iloc[idx]
        cat = None if cat_confounder_train is None else cat_confounder_train.iloc[idx]
        return con, cat

    def _score_with_C(tr_idx, val_idx, V_filt):
        Xtr, Xval = feature_generation([train_data[i] for i in tr_idx], [train_data[i] for i in val_idx], V_filt, method=feature_kind, metric=metric)
        if deconf:
            ctr, cta = _fold_confs(tr_idx)
            cvr, cva = _fold_confs(val_idx)
            Xtr, Xval = deconfound(Xtr, ctr, cta, X_test=Xval, con_confounder_test=cvr, cat_confounder_test=cva)
        ytr, yval = labels[tr_idx], labels[val_idx]

        make_final = clf_dict[final_svm_key]["make"]
        best, bestC = -np.inf, None
        for C in C_grid:
            mdl = make_final(C=C)
            # TODO should i z score here? 
            out = linear_classifier(Xtr, ytr, Xval, yval, clfs_list=[mdl], z_score=2)
            yhat = next(iter(out.values()))['predictions']
            # TODO 
            s =  balanced_accuracy_score(yval, yhat)
            # s = f1_score(yval, yhat, average="macro")
            if s > best: 
                best = s
        return best

    def _tune_model(model_key):
        space = clf_dict[model_key]["space"]

        def _fold_eval(tr_idx, val_idx, params):
            con_tr, cat_tr = _fold_confs(tr_idx)
            V = _fit_filters(covs[tr_idx], labels[tr_idx], model_key, params, con_tr, cat_tr)
            return _score_with_C(tr_idx, val_idx, V)

        @use_named_args(space)
        def obj(**params):
            scores = Parallel(n_jobs=n_inner_splits)(
                delayed(_fold_eval)(tr_idx, val_idx, params) for tr_idx, val_idx in skf.split(covs, labels)
            )
            return -float(np.mean(scores))

        res = gp_minimize(obj, space, n_calls=n_calls, n_initial_points=n_initial, random_state=random_state)
        theta = {dim.name: val for dim, val in zip(space, res.x)}
        scores = []
        for tr_idx, val_idx in skf.split(covs, labels):
            con_tr, cat_tr = _fold_confs(tr_idx)
            V = _fit_filters(covs[tr_idx], labels[tr_idx], model_key, theta, con_tr, cat_tr)
            score = _score_with_C(tr_idx, val_idx, V)
            scores.append(score)
        return {"key": model_key, "theta": theta, "cv": np.mean(np.array(scores))}


    results = Parallel(n_jobs=len(tan_model_keys))(
        delayed(_tune_model)(k) for k in tan_model_keys
    )
    cv_scores = np.array([r["cv"] for r in results])
    winner = results[np.argmax(cv_scores)]

    return {
        "winner_model": winner["key"],
        "winner_theta": winner["theta"],
    }

def TSSF(covs, labels, clf_str="logreg_en",clf_params=None, metric="riemann", deconf=True, con_confounder_train=None, cat_confounder_train=None, z_score=2, haufe=True, visualize=False,output_dir="plots"):

    # https://ieeexplore.ieee.org/abstract/document/9630144/references#references
    # https://arxiv.org/abs/1909.10567
    data, Frechet_Mean = tangent_transform(covs,metric=metric)

    if deconf:
        data = deconfound(data, con_confounder_train, cat_confounder_train, X_test=None, con_confounder_test=None, cat_confounder_test=None)

    if z_score == 1:
        scaler = StandardScaler(with_mean=True, with_std=False)
        data = scaler.fit_transform(data)
    elif z_score == 2:
        scaler = StandardScaler(with_mean=True, with_std=True)
        data = scaler.fit_transform(data)

    # Build tuned classifier
    spec = clf_dict[clf_str]
    make_clf = spec["make"]
    clf = make_clf(**(clf_params or {})).fit(data, labels)

    coef = np.atleast_2d(clf.coef_)
    if coef.shape[1] != data.shape[1]:
        coef = coef.T

    if haufe:
        coef = haufe_transform(data, coef.T, method="basic")
    
    # boundary_matrix = untangent_space(coef, Frechet_Mean, metric=metric)[0,:,:]
    # eigs, filters  = eigh(boundary_matrix, Frechet_Mean)
    # riem_eig = (np.log(eigs))**2
    # TODO Dffferent transformation functions from each paper
    # fkt_riem_eigs = np.maximum(eigs,1/eigs)
    # TODO Specific for Log Euclidean case
    # Equivalent to the 2 norm of the distance to reference eigenvalue on the tangent space
    # sqrt((log(lamda) - log(1))^2) == abs(log(lamda) - log(1)) == abs(log(lambda))
    # riem_eig = np.abs(np.log(eigs))
    
    # Here the matrix remains in the tangent space so the eigenvalues are in terms of the riemannian metric
    boundary_matrix = unupper(coef)[0,:,:]
    eigs, filters  = eigh(boundary_matrix)
    riem_eig = eigs**2

    if visualize:
        plt.scatter(range(0,riem_eig.shape[0]),riem_eig)
        plt.title("Riemannian Distance Supported by Spatial Filter")
        plt.xlabel("Max Eigenvector for Group B to Max Eigenvector for Group A")
        # plt.ylabel(r"$|log(\lambda)|$")
        plt.ylabel(r"$\log(\lambda)^2$")
        plt.savefig(os.path.join(output_dir, "fkt_scree.svg"))
        plt.close('all')

    return riem_eig, filters, boundary_matrix, Frechet_Mean

def orthonormalize_filters(W1, W2):
    # Stack the two filters into a single matrix
    W = np.concatenate((W1, W2)).T  # shape: (features x 2)
    print("Shape of concatenated filters", W.shape)
    # Perform QR decomposition to orthonormalize the filters
    Q, _ = np.linalg.qr(W)
    

    # Verify that the inner product between the two orthonormalized vectors is 0 (orthogonality)
    print(f'Inner product between Q[:, 0] and Q[:, 1]: {np.dot(Q[:, 0].T, Q[:, 1])} (should be 0)')
    
    # Verify that the inner product within each vector is 1 (normalization)
    print(f'Norm of Q[:, 0]: {np.dot(Q[:, 0].T, Q[:, 0])} (should be 1)')
    print(f'Norm of Q[:, 1]: {np.dot(Q[:, 1].T, Q[:, 1])} (should be 1)')
    
    return Q

def whiten(X,n_components, method="SVD"):
    # -1 to account for demean
    n_samples = X.shape[-1]-1
    X_mean = X.mean(axis=-1)
    X_centered = X - X_mean[:, np.newaxis] 

    if method == "SVD":
        u, d = svd(X_centered, full_matrices=False, check_finite=False)[:2]
        # Give consistent eigenvectors for both svd solvers
        # u *= np.sign(u[0])
        K = (u / d).T[:n_components]  # see (6.33) p.140
        del u, d
        whitening_matrix = np.sqrt(n_samples)*K
    elif method == "Cholesky":
    # Does not Orthogonalize, just has unit covariance
        # Step 2: Perform Cholesky decomposition
        L = np.linalg.cholesky(np.cov(X_centered,ddof=1))
        # Step 3:
        whitening_matrix = np.linalg.inv(L)
    elif method == "InvCov":
        # Calculate the covariance matrix of the centered data
        cov_matrix = np.cov(X_centered)
        # Perform eigenvalue decomposition of the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        # Calculate the whitening matrix
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        whitening_matrix = eigvecs @ D_inv_sqrt @ eigvecs.T
   
    whitened_data = whitening_matrix@X_centered

    return whitened_data, whitening_matrix

########################################################################## Functions for computing voxel level filters #################################################################################

import sys
sys.path.append('/project/3022057.01/IFA/utils')

import torch
import numpy as np
from preprocessing import load_subject
import os
import hcp_utils as hcp
from nilearn.plotting import view_surf
from sklearn.decomposition import PCA
from IPython.display import clear_output, display

def orthonormalize_and_analyze_filters(W1, W2, device='cuda', n_components=10):
    """
    Orthonormalize filters, perform PCA, and compute scree plot to measure shared information.

    Parameters:
    - W1 (torch.Tensor): The first set of filters (features x filters).
    - W2 (torch.Tensor): The second set of filters (features x filters).
    - device (str): The device to use ('cuda' for GPU, 'cpu' for CPU).
    - n_components (int): Number of PCA components to compute.

    Returns:
    - Q (torch.Tensor): Orthonormalized combined filters (features x 2*filters).
    - pca (PCA object): Fitted PCA object for shared information analysis.
    """
    # Concatenate filters into one matrix
    combined_filters = torch.cat((W1, W2), dim=1).to(device)  # features x (filters_W1 + filters_W2)
    
    # Perform QR decomposition for orthonormalization
    Q, _ = torch.linalg.qr(combined_filters)
    Q_cpu = Q.cpu().numpy()  # Move to CPU for PCA
    print(f"Rank of Q: {np.linalg.matrix_rank(Q_cpu)}")
    print(f"Q shape: {Q_cpu.shape}")

    print(f'Orthonormalized matrix shape: {Q_cpu.shape}')
    print(f'Inner product matrix (should be close to identity):\n{np.round(Q_cpu.T @ Q_cpu, decimals=4)}')
    
    # Perform PCA on the orthonormalized filters
    pca = PCA(n_components=n_components,svd_solver='arpack')
    pca.fit(Q_cpu.T)  # Use transposed data (filters as rows)

    # Scree plot
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', label='Cumulative Variance')
    plt.plot(pca.explained_variance_ratio_, marker='x', label='Individual Variance')
    plt.title('PCA Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.legend()
    plt.grid()
    plt.show()

    # Print top explained variances
    print(f"Explained Variance Ratios (Top {n_components}): {pca.explained_variance_ratio_[:n_components]}")
    print(f"Cumulative Variance Ratio: {np.sum(pca.explained_variance_ratio_[:n_components]):.4f}")

    # Return orthonormalized filters and PCA object
    return Q_cpu, pca


def analyze_svd_of_filters(W1, W2, device='cuda'):
    """
    Normalize each filter set separately (columns have unit norm), 
    perform SVD on the combined matrix, and plot raw singular values.

    Parameters:
    - W1 (torch.Tensor): First set of filters (features x filters).
    - W2 (torch.Tensor): Second set of filters (features x filters).
    - device (str): Device to use ('cuda' for GPU, 'cpu' for CPU).

    Returns:
    - singular_values (numpy.ndarray): Singular values of the combined filter matrix.
    """
    # Move data to the specified device
    W1 = W1.to(device)
    W2 = W2.to(device)

    # Normalize each filter set column-wise
    W1_normalized = W1 / torch.norm(W1, dim=0, keepdim=True)
    W2_normalized = W2 / torch.norm(W2, dim=0, keepdim=True)

    # Combine the normalized filters
    combined_filters = torch.cat((W1_normalized, W2_normalized), dim=1).cpu().numpy()  # Move to CPU for SVD

    # Perform SVD on the combined matrix
    U, S, Vt = np.linalg.svd(combined_filters, full_matrices=False)

    # Plot raw singular values
    plt.figure(figsize=(8, 5))
    plt.plot(S, marker='o', label='Singular Values')
    plt.title('Singular Values of Combined Normalized Filters')
    plt.xlabel('Singular Value Index')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid()
    plt.show()

    # Display the top singular values
    print(f"Top 10 Singular Values: {S[:10]}")
    return S

def process_and_visualize_reconstruction(reduced_A, reduced_B, reduced_all, all_filters, device='cuda'):
    """
    Function to process filters, compute reconstruction percentage for each group, and visualize.

    Parameters:
    - reduced_A: numpy.ndarray, group A dataset
    - reduced_B: numpy.ndarray, group B dataset
    - all_filters: numpy.ndarray, filters to be processed
    - hcp: HCP utility object (e.g., for mesh and surface visualization)
    - device: str, device to use ('cuda' for GPU, 'cpu' for CPU')

    Returns:
    None
    """
    # Move data to GPU
    reduced_A_torch = torch.tensor(reduced_A, device=device)
    reduced_B_torch = torch.tensor(reduced_B, device=device)
    reduced_all_torch = torch.tensor(reduced_all, device=device)

    all_filters_torch = torch.tensor(all_filters, device=device)

    diff = 0 
    # Loop through filters
    for i in range(all_filters_torch.shape[1]):
        current_filter = all_filters_torch[:, i:i+1]  # Select current filter (shape: [features, 1])

        # Reconstruction for Group A
        recon_A = current_filter @ (torch.linalg.pinv(current_filter) @ reduced_A_torch.T)  # Shape: [features, samples]
        reconstruction_A_percentage = 1 - (torch.norm(reduced_A_torch.T - recon_A, p='fro')**2 /
                                           torch.norm(reduced_A_torch.T, p='fro')**2)

        # Reconstruction for Group B
        recon_B = current_filter @ (torch.linalg.pinv(current_filter) @ reduced_B_torch.T)  # Shape: [features, samples]
        reconstruction_B_percentage = 1 - (torch.norm(reduced_B_torch.T - recon_B, p='fro')**2 /
                                           torch.norm(reduced_B_torch.T, p='fro')**2)

        # Reconstruction for Group A
        recon_all = current_filter @ (torch.linalg.pinv(current_filter) @ reduced_all_torch.T)  # Shape: [features, samples]
        reconstruction_all_percentage = 1 - (torch.norm(reduced_all_torch.T - recon_all, p='fro')**2 /
                                           torch.norm(reduced_all_torch.T, p='fro')**2)

        # Print reconstruction percentages
        print(f"Filter {i + 1}:")
        print(f"  Reconstruction Percentage (Group A): {reconstruction_A_percentage.item() * 100:.2f}%")
        print(f"  Reconstruction Percentage (Group B): {reconstruction_B_percentage.item() * 100:.2f}%\n")
        print(f"  Reconstruction Percentage (Difference A-B): {(reconstruction_A_percentage.item() - reconstruction_B_percentage.item()) * 100:.2f}%\n")
        print(f"  Reconstruction Percentage (All): {reconstruction_all_percentage.item() * 100:.2f}%\n")

        diff += torch.abs(reconstruction_A_percentage - reconstruction_B_percentage)
        # Visualize filter
        
        print("Visualize Filter")
        view_filter = view_surf(
            surf_mesh=hcp.mesh.inflated,
            surf_map=hcp.cortex_data(current_filter[:,0].cpu().numpy()),
            bg_map=hcp.mesh.sulc,
            threshold=np.percentile(np.abs(hcp.cortex_data(current_filter[:,0].cpu().numpy())), 0)
        )
        display(view_filter)

        A = (torch.linalg.pinv(reduced_A_torch@torch.linalg.pinv(current_filter.T)) @ reduced_A_torch).T
        # A /= torch.norm(A, dim=0, keepdim=True)
        B = (torch.linalg.pinv(reduced_B_torch@torch.linalg.pinv(current_filter.T)) @ reduced_B_torch).T
        # B /= torch.norm(A, dim=0, keepdim=True)
        spatial_diff = A-B
        
        print("Visulize Group A Dual Projected on the Filter")
        view_filter = view_surf(
            surf_mesh=hcp.mesh.inflated,
            surf_map=hcp.cortex_data(A[:,0].cpu().numpy()),
            bg_map=hcp.mesh.sulc,
            threshold=np.percentile(np.abs(hcp.cortex_data(A[:,0].cpu().numpy())), 0)
        )
        display(view_filter)
        
        print("Visulize Group B Dual Projected on the Filter")
        view_filter = view_surf(
            surf_mesh=hcp.mesh.inflated,
            surf_map=hcp.cortex_data(B[:,0].cpu().numpy()),
            bg_map=hcp.mesh.sulc,
            threshold=np.percentile(np.abs(hcp.cortex_data(B[:,0].cpu().numpy())), 0)
        )
        display(view_filter)
        
        print("Visulize Group A Dual Projected  - Group B Dual Projected")
        view_filter = view_surf(
            surf_mesh=hcp.mesh.inflated,
            surf_map=hcp.cortex_data(spatial_diff[:,0].cpu().numpy()),
            bg_map=hcp.mesh.sulc,
            threshold=np.percentile(np.abs(hcp.cortex_data(spatial_diff[:,0].cpu().numpy())), 0)
        )
        display(view_filter)
    
    print("Total Diff", diff.item()*100)
    print("Average Diff", diff.item()*100/all_filters_torch.shape[1])
    # Clean up GPU memory
    del reduced_A_torch, reduced_B_torch, all_filters_torch, current_filter
    torch.cuda.empty_cache()

def compute_cov(data, method='svd', log=False, shrink=None):
    """
    Compute a covariance matrix from a subject's data.

    It then computes the covariance using either:
      - SVD-based eigen-decomposition (optionally applying a log transform to the eigenvalues
        and applying shrinkage), or
      - A direct Gram matrix calculation.

    Parameters:
      data (np.ndarray): Subject data with shape (n_timepoints, n_features).
      method (str): 'svd' for SVD-based or 'gram' for direct Gram matrix computation.
      log (bool): If True, apply the logarithm to eigenvalues (SVD) 
        (note not possible if calculated via gram matrix due to size of eigendecompoistion needed)
      shrink (float or None): Shrinkage parameter (0 to 1) to apply in SVD mode.

    Returns:
      cov_cpu (torch.Tensor): The computed covariance matrix (on CPU).
    """
    # Select device: GPU if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move data to the selected device as float32
    subject_torch = torch.tensor(data, device=device, dtype=torch.float32)
    
    # Demean the data 
    col_mean = torch.mean(subject_torch, dim=0)
    demeaned = (subject_torch - col_mean)
    n = demeaned.shape[0]
    del subject_torch, col_mean
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print_memory_status("Before computing raw cov matrix")
    if method == 'svd':
        # Use SVD to compute eigenvalues/eigenvectors of the standardized data.
        _, S, Vh = torch.linalg.svd(demeaned, full_matrices=False)
        del demeaned
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        # Compute eigenvalues (variance explained) using an unbiased estimator.
        eigvals = S**2 / (n - 1)
        if shrink is not None:
            isotropic_mean = eigvals.mean()
            eigvals = (1 - shrink) * eigvals + shrink * isotropic_mean
        if log:
            eigvals = torch.log(eigvals)
        # Reconstruct the covariance matrix from eigenvalues and eigenvectors.
        cov = Vh.T @ torch.diag(eigvals) @ Vh
        del Vh, eigvals
    elif method == 'gram':
        # Directly compute the Gram matrix.
        cov = (demeaned.T @ demeaned).div_((n - 1))
        del demeaned
    else:
        raise ValueError("Unknown method: choose 'svd' or 'gram'")
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print_memory_status("Before moving raw cov to CPU")
    if cov.device.type != 'cpu':
        cov = cov.cpu()
    print_memory_status("After moving raw cov to CPU")

    return cov, n


def average_covariances(group, method='svd', log=False, shrink=None):
    """
    Load each subject from a list (paths or data), compute its covariance,
    and return the average covariance.

    Parameters:
      group (list): List of subject file paths (str)
      method (str): 'svd' or 'gram'.
      log (bool): Whether to apply the logarithm (as per compute_covariance_from_data).
      shrink (float or None): Shrinkage parameter for the SVD method.

    Returns:
      avg_cov (torch.Tensor): The average covariance matrix (on CPU).
    """
    avg_cov = None
    avg_samples = 0
    for idx, sub in enumerate(group):
        # Load subject data if a file path is provided.
        data = load_subject(sub)

        cov, samples = compute_cov(data, method=method, log=log, shrink=shrink)
        avg_samples += samples
        if avg_cov is None:
            avg_cov = cov.clone()
        else:
            avg_cov += cov
        del cov
        # Clear CUDA cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Average the summed covariance matrices
    avg_cov.div_(len(group))
    avg_samples = avg_samples/(len(group))
    return avg_cov, avg_samples

    #     cov_cpu = cov.cpu()
    #     del cov
    #     torch.cuda.empty_cache()
    
    #     if upper_triangular_indices is None:
    #         upper_triangular_indices = torch.triu_indices(cov_cpu.shape[0], cov_cpu.shape[1])

    #     # Vectorize the upper triangular part of the covariance matrix on CPU
    #     cov_vector_cpu = cov_cpu[upper_triangular_indices[0], upper_triangular_indices[1]]
    #     del cov_cpu
    #     torch.cuda.empty_cache()

    #     # Sum up vectors of upper triangular parts across subjects
    #     if sum is None:
    #         sum = cov_vector_cpu.clone()  # Initialize the sum
    #     else:
    #         # Move log_sum to GPU for addition
    #         sum_gpu = sum.to(device='cuda')
    #         sum_gpu += cov_vector_cpu.to(device='cuda')

    #         # Move log_sum back to CPU
    #         sum = sum_gpu.cpu()

    #         del cov_vector_cpu, sum_gpu
    #         torch.cuda.empty_cache()

    #     print(count)
    #     count += 1

    # # Divide the summed vector by the number of subjects in place
    # sum.div_(len(group))  # In-place division

    # return sum

def oas_shrinkage(sample_covariance, n_samples):
    """Get OAS shrinkage parameter.
    
    Arguments:
    ----------
    sample_covariance : torch.Tensor
        Sample covariance matrix.
    """
    n_dim = sample_covariance.shape[0]
    tr_cov = torch.trace(sample_covariance)
    tr_prod = torch.linalg.norm(sample_covariance, ord='fro')**2
    shrinkage = (
    (1 - 2 / n_dim) * tr_prod + tr_cov ** 2
    ) / (
    (n_samples + 1 - 2 / n_dim) * (tr_prod - tr_cov ** 2 / n_dim)
    )
    shrinkage = torch.clamp(shrinkage, max=1)
    return shrinkage

def isotropic_estimator(sample_covariance):
    """Isotropic covariance estimate with same trace as sample.
    
    Arguments:
    ---------- 
    sample_covariance : torch.Tensor
        Sample covariance matrix.
    """
    n_dim = sample_covariance.shape[0]
    trace = torch.trace(sample_covariance)  # Calculate the trace once
    isotropic = (trace / n_dim) * torch.eye(n_dim, device=sample_covariance.device, dtype=sample_covariance.dtype)
    return isotropic

def oas_estimator(sample_covariance,n_samples, shrink=None):
    """Oracle Approximating Shrinkage (OAS) covariance estimate.

    Arguments:
    ----------
    Covariance : torch.Tensor
        Data matrix with shape (n_features, n_features).
    """
    if shrink is None:
        shrinkage = oas_shrinkage(sample_covariance, n_samples) 
    else:
        shrinkage = torch.as_tensor(shrink, dtype=sample_covariance.dtype,device=sample_covariance.device)
    
    # 2) get trace & isotropic mean
    tr = torch.trace(sample_covariance)     
    iso = tr / sample_covariance.shape[0]
                               

    # 3) shrink the entire matrix
    sample_covariance.mul_(1 - shrinkage)                 

    # 4) add α·iso *only to the diagonal
    sample_covariance.diagonal().add_(shrinkage * iso)       

    # 5) move back to CPU if needed
    if sample_covariance.device.type != 'cpu':
        sample_covariance = sample_covariance.cpu()

    return sample_covariance

def Large_FKT(X1, X2, n, LOBPCG=True,num_simulations=1000, log=False,largest=True,reg=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if LOBPCG:
        try:
            if reg:
                X1_gpu = X1.clone().to(device=device)
            else:
                X1_gpu = X1.to(device=device)
            Sum_gpu = (X1 + X2).to(device=device)
            if log:
                L, Q = torch.lobpcg(A=X1_gpu - Sum_gpu.div_(2),B=None, k=n, largest=largest)
            else:
                if reg:
                    # 1) Add a small ridge ε to both A and B
                    eps = 1e-6
                    N = X1_gpu.size(0)
                    X1_gpu.add_(eps *torch.eye(N, device=device))
                    Sum_gpu.add_(eps *torch.eye(N, device=device))

                    # 2) Build Jacobi preconditioner from A_eps
                    inv_diag = 1.0 / X1_gpu.diag()     # shape (N,)
                    iK = torch.diag(inv_diag)         # shape (N, N)
                else:
                    iK = None
                L, Q = torch.lobpcg(A=X1_gpu,B=Sum_gpu, k=n, iK=iK, largest=largest)
                if iK is not None:
                    del iK
            torch.cuda.empty_cache()
            print(f"L: {L.shape}, S: {Q.shape}")
            print(f"eigenvalue {L}")
            # del X1_gpu
            del X1_gpu,Sum_gpu
            torch.cuda.empty_cache()
            L_cpu = L.to(device='cpu')
            Q_cpu = Q.to(device='cpu')
            del L,Q
            torch.cuda.empty_cache()
            return L_cpu,Q_cpu
        except Exception as e:
            print(f"Failed to compute the eigendecomposition: {e}")
            if "X1_gpu" in locals():
                del X1_gpu
            if "Sum_gpu" in locals():
                del Sum_gpu
            torch.cuda.empty_cache()
            return None, None
    else:
        # Random initialization of the vector z
        z = torch.rand(X1.shape[1], dtype=X1.dtype).to(device=device)

        for _ in range(num_simulations):
            # Move X1 to GPU, perform the multiplication, and then delete X1_gpu
            X1_gpu = X1.to(device=device)
            vec = X1_gpu @ z
            del X1_gpu
            torch.cuda.empty_cache()  # Clear GPU memory
            
            # Move X2 to GPU, solve the linear system, and then delete X2_gpu
            X2_gpu = X2.to(device=device)
            v_new = torch.linalg.solve(X2_gpu, vec)
            del X2_gpu, vec
            torch.cuda.empty_cache()  # Clear GPU memory

            # Normalize v_new
            v_new /= torch.norm(v_new)
            
            # Check for convergence (if the difference between iterations is small)
            if torch.allclose(z, v_new, atol=1e-6):
                z = v_new.to(device='cpu')  # Move to CPU before deleting
                del v_new
                torch.cuda.empty_cache()
                break
            
            z = v_new
            del v_new
            torch.cuda.empty_cache()
            print('I')
        
        # Move the final result back to CPU if not already done
        print(f"z: {z.shape}")
        z_cpu = z.to(device='cpu') if z.is_cuda else z
        del z
        torch.cuda.empty_cache()
        # The Rayleigh quotient gives an estimate of the eigenvalue
        # lambda_approx = np.dot(z.T, Mz) / np.dot(z.T, Kz)
        return None, z_cpu
    
def haufe_transform_torch(W, Sigma_avg):
    """
    Perform the Haufe transformation to convert discriminant vectors into interpretable activation patterns using PyTorch.
    
    Parameters:
    W (torch.Tensor): The matrix of discriminant vectors (shape: features x components).
    Sigma_avg (torch.Tensor): The average covariance matrix (shape: features x features).
    device (str): The device to use for computation ('cuda' for GPU, 'cpu' for CPU).
    
    Returns:
    A (torch.Tensor): The matrix of activation patterns (shape: features x components).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure the tensors are on the correct device
    W = W.to(device)
    Sigma_avg = Sigma_avg.to(device)
    
    # Calculate the covariance matrix of the discriminant scores
    Sigma_s_hat = W.T @ Sigma_avg @ W
    
    # Invert the covariance matrix of the discriminant scores
    Sigma_s_hat_inv = torch.inverse(Sigma_s_hat)
    del Sigma_s_hat
    torch.cuda.empty_cache()
    
    # Apply the transformation to obtain the activation patterns
    A = Sigma_avg @ W @ Sigma_s_hat_inv
    del Sigma_s_hat_inv
    torch.cuda.empty_cache()
    A_cpu = A.to(device='cpu')
    del A
    torch.cuda.empty_cache()
    
    return A_cpu

def orthonormalize_filters_torch(W1, W2):
    """
    Orthonormalize two filters using QR decomposition in PyTorch, then return the results on the CPU.
    
    Parameters:
    W1 (torch.Tensor): The first filter vector (shape: features x A_components).
    W2 (torch.Tensor): The second filter vector (shape: features x B_components).
    device (str): The device to use for computation ('cuda' for GPU, 'cpu' for CPU).
    
    Intermediate:
    R (torch.Tensor): The upper triangular matrix from QR decomposition (shape: (A_components + B_components) x (A_components + B_components)), on CPU.

    Returns:
    Q (torch.Tensor): The orthonormalized matrix containing the two orthonormalized filters (shape: features x (A_components + B_components)), on CPU.
    """
    
    # Stack the two filters into a single matrix
    W = torch.cat((W1, W2), dim=1)
    # Ensure the matrix is on the correct device
    # Perform QR decomposition to orthonormalize the filters
    Q, _ = torch.linalg.qr(W)
    torch.cuda.empty_cache()

    # Verify that the inner product between the two orthonormalized vectors is 0 (orthogonality)
    print(f'Inner product between Q[:, 0] and Q[:, 1]: { Q.T@Q} (should be 0)')
    
    # Verify that the inner product within each vector is 1 (normalization)
    print(f'Norm of Q[:, 0]: { Q[:, 0].T@Q[:, 0]} (should be 1)')
    print(f'Norm of Q[:, 1]: { Q[:, 1].T@Q[:, 1]} (should be 1)')
    
    return Q

def save_brain(map, title, output_dir):
        view_surf(
        surf_mesh=hcp.mesh.inflated,
        surf_map=hcp.cortex_data(map),
        bg_map=hcp.mesh.sulc,
        title=title
    ).save_as_html(os.path.join(output_dir, f"{title}.html"))


def create_dense_cov(group=None, group_paths=None,  paths=False, log=False, shrinkage=0.01 , cov_method='svd'):
        if paths:
            dense, samples = average_covariances(group_paths, method=cov_method, log=log, shrink=None)
        else:
            dense, samples = compute_cov(group, method=cov_method, log=log, shrink=None)
        
        print_memory_status("Before Regularizing Matrix")
        dense_adj = oas_estimator(dense,n_samples=samples,shrink=shrinkage)

        return dense_adj

def print_memory_status(tag=""):
    """Prints current and peak RAM usage in GB with an optional tag."""
    status = {}
    with open("/proc/self/status", "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) == 2:
                key, value = parts
                status[key] = value.strip()

    ram_current_gb = int(status.get("VmRSS", "0").split()[0]) / 1e6
    ram_peak_gb = int(status.get("VmPeak", "0").split()[0]) / 1e6

    print(f"[{tag}] Current RAM: {ram_current_gb:.2f} GB | Peak RAM: {ram_peak_gb:.2f} GB")




def voxelwise_FKT(groupA=None, groupB=None, n_filters_per_group=1, groupA_paths=None, groupB_paths=None, paths=False,log=False,shrinkage=0.01,cov_method='svd',outputfolder='Path', save=False, save_img=True):
    print(log,shrinkage)
    with torch.no_grad():
        try:
            assert (not log) or (cov_method == 'svd'), "If log is True, then method must be 'svd'."
            
            # print_memory_status("Before computing Group A dense matrix")
            A_dense_adj = create_dense_cov(group=groupA, group_paths=groupA_paths,  paths=paths, log=log, shrinkage=shrinkage , cov_method=cov_method)
            # print(f"Group A dense matrix loaded. Memory size: {A_dense_adj.element_size() * A_dense_adj.nelement() / 1e9:.4f} GB")

            # print_memory_status("Before computing Group B dense matrix")
            B_dense_adj = create_dense_cov(group=groupB, group_paths=groupB_paths,  paths=paths, log=log, shrinkage=shrinkage , cov_method=cov_method)
            # print(f"Group B dense matrix loaded. Memory size: {B_dense_adj.element_size() * B_dense_adj.nelement() / 1e9:.4f} GB")

            # print_memory_status("Before running Large_FKT for A_filters")
            _, A_filters = Large_FKT(A_dense_adj, B_dense_adj, n=n_filters_per_group, LOBPCG=True,num_simulations=1000,log=log,largest=True)
            np.save(os.path.join(outputfolder, "filtersA.npy"), A_filters.cpu().numpy())

            # print_memory_status("Before running Large_FKT for B_filters")
            _, B_filters = Large_FKT(B_dense_adj, A_dense_adj, n=n_filters_per_group, LOBPCG=True,num_simulations=1000,log=log,largest=True)
            np.save(os.path.join(outputfolder, "filtersB.npy"), B_filters.cpu().numpy())
            
            # print_memory_status("After Running FKT")

            if save_img:
                for i in range(A_filters.shape[1]):
                    save_brain(A_filters[:,i].cpu().numpy(),f"A_filter{i}",outputfolder)
                for i in range(B_filters.shape[1]):
                    save_brain(B_filters[:,i].cpu().numpy(),f"B_filter{i}",outputfolder)

            
            # If the filters were calculated in logspace then the average cov is in log space and cant be used for haufe transform
            # thus need to calcualte the normal covariance and use this for haufe
            if log:
                if save:
                    np.save(os.path.join(outputfolder, "A_avg_logcov.npy"), A_dense_adj.cpu().numpy())
                    np.save(os.path.join(outputfolder, "B_avg_logcov.npy"), B_dense_adj.cpu().numpy())
                del A_dense_adj, B_dense_adj
                gc.collect()
                torch.cuda.empty_cache()

                A_dense_adj = create_dense_cov(group=groupA, group_paths=groupA_paths,paths=paths, log=False, shrinkage=shrinkage , cov_method='svd')
                B_dense_adj = create_dense_cov(group=groupB, group_paths=groupB_paths,paths=paths, log=False, shrinkage=shrinkage , cov_method='svd')

                if save:
                    np.save(os.path.join(outputfolder, "A_avg_cov.npy"), A_dense_adj.cpu().numpy())
                    np.save(os.path.join(outputfolder, "B_avg_cov.npy"), B_dense_adj.cpu().numpy())
            else:
                if save:
                    np.save(os.path.join(outputfolder, "A_avg_cov.npy"), A_dense_adj.cpu().numpy())
                    np.save(os.path.join(outputfolder, "B_avg_cov.npy"), B_dense_adj.cpu().numpy())
            
            A_filters_haufe = haufe_transform_torch(A_filters,A_dense_adj)
            del A_dense_adj
            gc.collect()
            torch.cuda.empty_cache()
            print_memory_status("After Running Group A Haufe Transform and Deleting Covariances")
            
            np.save(os.path.join(outputfolder, "A_filters_haufe.npy"), A_filters_haufe.cpu().numpy())

            B_filters_haufe = haufe_transform_torch(B_filters,B_dense_adj)
            del B_dense_adj
            gc.collect()
            torch.cuda.empty_cache()
            print_memory_status("After Running Group B Haufe Transform and Deleting Covariances")

            np.save(os.path.join(outputfolder, "B_filters_haufe.npy"), B_filters_haufe.cpu().numpy())
            
            if save_img:
                for i in range(A_filters_haufe.shape[1]):
                    save_brain(A_filters_haufe[:,i].cpu().numpy(),f"A_filter_haufe{i}",outputfolder)
                for i in range(B_filters_haufe.shape[1]):
                    save_brain(B_filters_haufe[:,i].cpu().numpy(),f"B_filter_haufe{i}",outputfolder)


            filters = orthonormalize_filters_torch(A_filters_haufe,B_filters_haufe)
            np.save(os.path.join(outputfolder, "filters.npy"), filters.cpu().numpy())
            for i in range(filters.shape[1]):
                save_brain(filters[:,i].cpu().numpy(),f"filter_haufe_ortho{i}",outputfolder)
        except Exception as e:
            print(f"Failed During Filter Computation: {e}")
            return

########################################################################################################################################################################################################
########################################################################## Untested and in development: joint optimization #############################################################################
########################################################################################################################################################################################################

def dual_projection(X, F):
    """
    Perform dual regression (projection) using the matrix F.

    Parameters:
        X: Input data matrix (subjects x components x features).
        F: Projection matrix.

    Returns:
        Dual-projected data matrix.
    """
    F_demeaned = F - jnp.mean(F,axis=0,keepdims=True)
    proj = jnp.linalg.pinv(X @ (F_demeaned))
    proj_demeaned = proj - jnp.mean(proj,axis=1,keepdims=True)
    return jnp.linalg.pinv(proj_demeaned) @ X

def resid_calc(orig, filt):
    recon = jnp.linalg.pinv(filt.T)@filt.T@orig@filt@jnp.linalg.pinv(filt)
    residuals = (orig) - (recon)
    # Compute variances
    var_residuals = jnp.var(residuals, ddof=1)
    var_original = jnp.var(orig, ddof=1)
    # Calculate variance explained
    variance_explained = (1 - (var_residuals / var_original))
    return variance_explained

def var_calc(orig, filt):
    orig_stack = np.vstack(orig)
    recon = orig_stack@filt@jnp.linalg.pinv(filt)
    residuals = (orig_stack) - (recon)
    # Compute variances
    var_residuals = jnp.var(residuals, ddof=1)
    var_original = jnp.var(orig_stack, ddof=1)
    # Calculate variance explained
    variance_explained = (1 - (var_residuals / var_original))
    return variance_explained

def connectivity_cost(F, W, mode="trace_ratio"):
    """
    Compute the connectivity term of the cost function.

    Parameters:
        F: Projection matrix.
        C, W: Covariance matrices for connectivity differences.
        mode: 'trace_ratio' or 'log_trace_difference'.

    Returns:
        Connectivity cost value.
    """
    if mode == "trace_ratio":
        return jnp.trace(F.T @ W @ F)
    elif mode == "var_explained":
        total_cost = resid_calc(W, F)
        return total_cost
    else:
        raise ValueError("Invalid mode. Choose 'trace_ratio' or 'log_trace_difference'.")

def multitransp(A):
    """Vectorized matrix transpose for JAX."""
    if A.ndim == 2:
        return A.T
    return jnp.transpose(A, (0, 2, 1))

def grassmann_dist_jax(A, B):
    """Compute the Grassmann distance between two points using JAX."""
    A_q, _ = jnp.linalg.qr(A.T)
    B_q, _ = jnp.linalg.qr(B.T)
    product = multitransp(A_q) @ B_q
    s = jnp.linalg.svd(product, compute_uv=False)
    angles = jnp.arccos(s)  # Convert to principal angles
    print(angles)
    # return jnp.mean(angles)
    # return jnp.linalg.norm(angles,ord=-jnp.inf)
    return (jnp.linalg.norm(angles,ord=2)**2)

def within_class_similarity(subject_matrices):
    """
    Compute the average pairwise Grassmann distance for within-class similarity.

    Parameters:
        subject_matrices: 3D Array (subjectsx components x features) for each subject in a class.

    Returns:
        Average Grassmann distance (lower is better for similarity).
    """
    distances = []
    n_subjects = subject_matrices.shape[0]
    for i in range(n_subjects):
        for j in range(i + 1, n_subjects):
            dist = grassmann_dist_jax(subject_matrices[i,:,:], subject_matrices[j,:,:])
            distances.append(dist)
    return np.mean(distances)

def between_class_difference(class_1_matrices, class_0_matrices):
    """
    Compute the Grassmann distance between the mean subspaces of two classes.

    Parameters:
        class_1_matrices: 3D Array (subjectsx components x features) for each subject in class 1.
        class_0_matrices: 3D Array (subjectsx components x features) for each subject in a class 0.

    Returns:
        Grassmann distance between the mean subspaces of the two classes.
    """
    # Compute mean subspaces
    mean_class_1 = jnp.mean(class_1_matrices, axis=0)
    mean_class_0 = jnp.mean(class_0_matrices, axis=0)
    # Compute Grassmann distance
    return grassmann_dist_jax(mean_class_1, mean_class_0)

def spatial_cost(F, X, labels, mode="mean_difference"):
    """
    Compute the spatial term of the cost function.

    Parameters:
        F: Projection matrix.
        X: Input data matrix (subjects x features).
        labels: Group labels for classifier-based optimization.
        mode: 'mean_difference' or 'classifier'.

    Returns:
        Spatial cost value.
    """
    # Dual-projected data
    projected_data = dual_projection(X, F)

    if mode == "mean_difference":
        group1_mean = jnp.mean(projected_data[labels == 1], axis=0)
        group2_mean = jnp.mean(projected_data[labels == 0], axis=0)
        dist = jnp.linalg.norm(group1_mean - group2_mean) ** 2
        return dist
    elif mode == "subspace":
        within_class = within_class_similarity((projected_data[labels == 1])) + within_class_similarity((projected_data[labels == 0]))
        between_class = between_class_difference((projected_data[labels == 1]), (projected_data[labels == 0]))
        return within_class - between_class
    else:
        raise ValueError("Invalid mode. Choose 'mean_difference' or 'classifier'.")

def optimize_combined(W, X, labels, alpha=1.0, beta=1.0, c=1.0, conn_mode="var_explained", spatial_mode="mean_difference", optimizer="ConjugateGradient", n_components=5,initial_point=None):
    """
    Optimize the combined cost function on the Stiefel manifold.

    Parameters:
        C, W: Covariance matrices for connectivity differences.
        X: Input data matrix (subjects x features).
        labels: Group labels for classifier-based optimization.
        alpha, beta: Weights for connectivity and spatial terms.
        conn_mode: 'trace_ratio' or 'log_trace_difference'.
        spatial_mode: 'mean_difference' or 'classifier'.
        n_components: Number of components to optimize.

    Returns:
        Optimized projection matrix F.
    """
    n_features = W.shape[0]
    manifold = Grassmann(n_features, n_components)
    # manifold = Stiefel(n_features, n_components)
    # manifold = Sphere(n_features, n_components)


    @pymanopt.function.jax(manifold)
    def combined_cost(F):
        """
        Combined cost function for optimizing connectivity and spatial differences.
        """
        conn_term = connectivity_cost(F, W, conn_mode)
        spatial_term = spatial_cost(F, X, labels, spatial_mode)
        # variance_term = var_calc(X, F)
        print("spatial_cost:",-beta * (spatial_term), -beta*(spatial_term))
        print("conn_term:",-alpha * conn_term)
        # print("variance_term:", -c*variance_term)
        print("total:",-alpha * conn_term + -beta * (spatial_term))
        return -alpha * conn_term + -beta * (spatial_term)  # Negative for minimization


    # Set up the problem with the decorated cost function
    problem = Problem(manifold=manifold, cost=combined_cost)

    # Select optimizer
    if optimizer == "ConjugateGradient":
        opt = ConjugateGradient(max_iterations=300, min_gradient_norm=1e-05,  min_step_size=1e-10, max_time=3600)
    elif optimizer == "TrustRegions":
        opt = TrustRegions()
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Run optimization
    if initial_point is not None:
        print("Has starting point")
        result = opt.run(problem,initial_point=initial_point)
    else:
        result = opt.run(problem)
    return result.point
########################################################################################################################################################################################################
########################################################################################################################################################################################################
