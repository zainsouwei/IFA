from skopt import gp_minimize
from skopt.utils import use_named_args
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score
from scipy.linalg import eigh
from pyriemann.estimation import Covariances
import numpy as np
from pyriemann.utils.tangentspace import untangent_space, unupper

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

        @use_named_args(space)
        def obj(**params):
            scores = []
            # TODO this should be in parallel 
            for tr_idx, val_idx in skf.split(covs, labels):
                con_tr, cat_tr = _fold_confs(tr_idx)
                V = _fit_filters(covs[tr_idx], labels[tr_idx], model_key, params, con_tr, cat_tr)
                s = _score_with_C(tr_idx, val_idx, V)
                scores.append(s)
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

    # ---- try all tangent models, pick winner on inner-CV mean ----
    results = [_tune_model(k) for k in tan_model_keys]
    winner = max(results, key=lambda r: r["cv"])

    return {
        "winner_model": winner["key"],
        "winner_theta": winner["theta"],
    }