import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, MultiTaskLasso, ElasticNet
from pyriemann.estimation import Covariances

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