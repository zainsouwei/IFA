import numpy as np
import functools
from sklearn.linear_model import LinearRegression, Lasso, MultiTaskLasso, ElasticNet
from pyriemann.estimation import Covariances
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import os

import sys
sys.path.append('/utils')
from preprocessing import load_subject #TODO change back to non PCN

def haufe_transform(data, filters, method="basic", alpha=1, beta=0, l1_ratio=0.5, lambda1=.01, lambda2=.01):
    S = (data @ filters)
    
    if method == "basic":
        proj = (np.linalg.pinv(S)@ data)
    elif method == "covs":
        cov_est_scm = Covariances(estimator='scm')
        s_cov = cov_est_scm.transform(S.T[np.newaxis,:,:])[0,:,:]
        data_cov = cov_est_scm.transform(data.T[np.newaxis,:,:])[0,:,:]
        proj = (data_cov @ filters @ np.linalg.inv(s_cov)).T
    elif method == "linreg":
        reg = LinearRegression()
        reg.fit(S, data)
        proj = reg.coef_.T
    elif method == "grouplassolinreg":
        reg = MultiTaskLasso(alpha=alpha)  # Using 5-fold cross-validation
        reg.fit(S, data)
        proj = reg.coef_.T
    elif method == "lassolinreg":
        reg = Lasso(alpha=alpha)  # Using 5-fold cross-validation
        reg.fit(S, data)
        proj = reg.coef_.T
    elif method == "elasticlinreg":
        reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        reg.fit(S, data)
        proj = reg.coef_.T
    elif method == "growl":
        # Proximal Operator for GrOWL targeting columns
        def prox_growl(V, lambda1, lambda2, tau):
            p, r = V.shape
            norms = np.linalg.norm(V, axis=0)  # Norms of columns
            indices = np.argsort(-norms)  # Sort indices by descending norms
            weights = lambda1 + lambda2 * np.linspace(1, 0, r)  # Weights decrease
            V_new = np.zeros_like(V)
            for i in range(r):
                idx = indices[i]
                if norms[idx] > weights[i] * tau:
                    V_new[:, idx] = (1 - tau * weights[i] / norms[idx]) * V[:, idx]
            return V_new
        
        # Initialization
        B = np.zeros((filters.shape[1], data.shape[1]))
        
        # Optimization Loop
        max_iter = 100
        learning_rate = 0.01
        for _ in range(max_iter):
            gradient = S.T @ (S @ B - data)
            B -= learning_rate * gradient
            B = prox_growl(B, lambda1, lambda2, tau=learning_rate)
            if np.linalg.norm(gradient) < 1e-1:
                break
        
        proj = B.T
    
    return proj

def process_subject_haufe(sub,pinv_TF):
    try:
        Xn = load_subject(sub)
        Xpf = pinv_TF@Xn
        del Xn
        return Xpf

    except Exception as e:
        print(f"Error processing subject: {e}")
        raise                      # (k, V)

def process_subject_haufe_partial(sub, pinv_TF, vt32, pinv_vt32):
    try:
    # X = T×V (float32), vt32 = K×V (float32), pinv_vt32 = V×K (float32)
        # Load raw subject data.
        Xn = load_subject(sub)
        # Partial out vt from the raw data.
        Xn_partial = Xn - (Xn @ pinv_vt32) @ vt32        # stays float32
        return pinv_TF @ Xn_partial                      # pinv_TF can be float32
    except Exception as e:
        print(f"Error processing subject {sub}: {e}")
        raise

def partial_filter_dual_regression(F, parcellated, paths, vt=None, workers=20):
    """
    Map the filters F from parcel space to vertex (CIFTI) space.
    
    Parameters:
      - F: Filters in parcel space.
      - parcellated: The parcellated data used to compute the transformation.
      - paths: List of subject file paths.
      - vt: The major eigenspace basis to partial out.
      - workers: Number of parallel workers.
    
    Returns:
      - The aggregated transformation across subjects.
    """
    # # Compute the transformation matrix using the parcellated data and F.
    # pinv_TF = np.linalg.pinv(parcellated.reshape(-1, parcellated.shape[-1]) @ np.linalg.pinv(F.T))
    
    # # Split pinv_TF along the column dimension into as many blocks as there are subjects.
    # pinv_TF_list = np.array_split(pinv_TF, len(paths), axis=1)

    stacked_parcellated = np.vstack(parcellated)  # (sum_T, Parcels)

    # Compute pseudo-inverse transformation
    # pinv_TF = np.linalg.pinv((stacked_parcellated @ np.linalg.pinv(F.T.astype(np.float64, copy=False),rcond=1e-12)).astype(np.float64, copy=False),rcond=1e-12)  # (k, sum_T)
    # pinv(F^T) in float64 → stable
    Ft_pinv64 = np.linalg.pinv(F.T.astype(np.float64, copy=False), rcond=1e-12)
    Z64 = (stacked_parcellated @ Ft_pinv64).astype(np.float64, copy=False)  # (ΣT, k)
    pinv_TF = np.linalg.pinv(Z64, rcond=1e-12).astype(np.float32, copy=False)  # (k, ΣT)

    # Compute number of timepoints per subject
    subject_lengths = [subj.shape[0] for subj in parcellated]
    cumsum_lengths = np.cumsum(subject_lengths)

    # Split pinv_TF based on subject timepoints
    pinv_TF_list = np.split(pinv_TF, cumsum_lengths[:-1], axis=1)

    if vt is None:
        func = process_subject_haufe                                     
    else:
        # precompute once in 64, then cast down
        vt32 = np.asarray(vt, dtype=np.float32, order='C')
        pinv_vt32 = np.linalg.pinv(np.asarray(vt, dtype=np.float64, order='C'), rcond=1e-12).astype(np.float32, copy=False)
        func = functools.partial(process_subject_haufe_partial, vt32=vt32, pinv_vt32=pinv_vt32)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(func, paths, pinv_TF_list))     
    
    # Aggregate the results (here, summing along the subject axis; adjust if needed)
    aggregated = np.array(results).sum(axis=0)
    return aggregated