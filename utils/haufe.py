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
        return None


def filter_dual_regression(F, parcellated,paths,workers=20):
    
    # Ensure the tensors are on the correct device
    pinv_TF = np.linalg.pinv(parcellated.reshape(-1,parcellated.shape[-1]) @ np.linalg.pinv(F.T))


    # pinv_TF_list = pinv_TF.reshape(len(paths),F.shape[1],pinv_TF.shape[0])
    pinv_TF_list = (np.array_split(pinv_TF, len(paths), axis=1))

    with ProcessPoolExecutor(max_workers=(int(workers))) as executor:
        # Use map to process subjects in parallel
        blocks = np.array(list(executor.map(process_subject_haufe, paths,pinv_TF_list)))
        return (blocks.sum(axis=0))
    

####################### For Partialing Out #######################
def process_subject_haufe_partial(sub, pinv_TF, vt):
    try:
        # Load raw subject data.
        Xn = load_subject(sub)
        # Partial out vt from the raw data.
        Xn_partial = Xn - (Xn @ np.linalg.pinv(vt)) @ vt
        # Apply the dual regression mapping.
        Xpf = pinv_TF @ Xn_partial
        return Xpf
    except Exception as e:
        print(f"Error processing subject {sub}: {e}")
        raise

def partial_filter_dual_regression(F, parcellated, paths, vt, workers=20):
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
    # Compute the transformation matrix using the parcellated data and F.
    pinv_TF = np.linalg.pinv(parcellated.reshape(-1, parcellated.shape[-1]) @ np.linalg.pinv(F.T))
    
    # Split pinv_TF along the column dimension into as many blocks as there are subjects.
    pinv_TF_list = np.array_split(pinv_TF, len(paths), axis=1)
    
    # Create a partial function so that vt is fixed for every subject.
    func = functools.partial(process_subject_haufe_partial, vt=vt)
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Map over subject paths and corresponding pinv_TF blocks.
        results = list(executor.map(func, paths, pinv_TF_list))
    
    # Aggregate the results (here, summing along the subject axis; adjust if needed)
    aggregated = np.array(results).sum(axis=0)
    return aggregated
