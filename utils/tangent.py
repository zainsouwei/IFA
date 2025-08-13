import numpy as np
from skopt import gp_minimize
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from joblib import Parallel, delayed
from skopt.utils import use_named_args
from threadpoolctl import threadpool_limits
from skopt import Optimizer
from itertools import product

import sys
sys.path.append('/utils')

from classification import linear_classifier, clf_dict
from regression import deconfound

def tangent_transform(train, test=None, metric="riemann"):
    if test is not None:
        Frechet_Mean = mean_covariance(train,metric=metric)

        # Perform tangent space projection for training data
        train = tangent_space(train, Frechet_Mean, metric=metric)
        test = tangent_space(test, Frechet_Mean, metric=metric)
        return train, test, Frechet_Mean

    else:
        Frechet_Mean = mean_covariance(train,metric=metric)
        train = tangent_space(train, Frechet_Mean, metric=metric)
        return train, Frechet_Mean

def tangent_classification(covs_train, y_train, covs_test, y_test, clf_str="svc_l2_sq",
                           z_score=2, metric="riemann",
                           deconf=False, con_confounder_train=None, cat_confounder_train=None,
                           con_confounder_test=None, cat_confounder_test=None,
                           random_state=0, n_inner_splits=5, n_cpus=15, n_batches=10):

    # ---- Outer transform once (for final evaluation only)
    X_train, X_test, _ = tangent_transform(covs_train, covs_test, metric=metric)
    if deconf:
        X_train, X_test = deconfound(
            X_train, con_confounder_train, cat_confounder_train,
            X_test=X_test, con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test
        )

    # ---- Pre-split folds once
    cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
    folds = list(cv.split(covs_train, y_train))

    # ---- Precompute per-fold tangent + optional deconf ONCE
    def _prepare_one_fold(tr_idx, val_idx):
        # fold-specific tangent
        Xtr, Xval, _ = tangent_transform(covs_train[tr_idx], covs_train[val_idx], metric=metric)
        # optional deconf
        if deconf:
            ctr = None if con_confounder_train is None else con_confounder_train.iloc[tr_idx]
            cta = None if cat_confounder_train is None else cat_confounder_train.iloc[tr_idx]
            cvr = None if con_confounder_train is None else con_confounder_train.iloc[val_idx]
            cva = None if cat_confounder_train is None else cat_confounder_train.iloc[val_idx]
            Xtr, Xval = deconfound(Xtr, ctr, cta, X_test=Xval, con_confounder_test=cvr,cat_confounder_test=cva)
        return (Xtr, y_train[tr_idx], Xval, y_train[val_idx])

    # use threads to share memory; cap BLAS to avoid oversubscription
    nj = min(n_inner_splits, max(1, int(n_cpus)))
    fold_cache = Parallel(n_jobs=nj)(
        delayed(_prepare_one_fold)(tr, val) for tr, val in folds
    )

    def tune_model(model_key):
        spec     = clf_dict[model_key]
        space    = spec["space"]
        make_clf = spec["make"]

        batch_size = max(1, n_cpus // n_inner_splits)
        opt = Optimizer(
            dimensions=space,
            base_estimator="GP",
            acq_func="EI",
            acq_optimizer="lbfgs",
            random_state=random_state,
        )

        best_params, best_score = None, -np.inf

        def eval_fold(params_dict, cache_item):
            Xtr, ytr, Xval, yval = cache_item
            mdl = make_clf(**params_dict)
            out = linear_classifier(Xtr, ytr, Xval, yval, clfs_list=[mdl], z_score=z_score)
            yhat = next(iter(out.values()))['predictions']
            return balanced_accuracy_score(yval, yhat)

        for _ in range(n_batches):
            X_batch = opt.ask(n_points=batch_size)
            
            jobs = [({space[i].name: x[i] for i in range(len(space))}, fc) for x, fc in product(X_batch, fold_cache)]

            with threadpool_limits(limits=1):
                fold_scores = Parallel(n_jobs=n_cpus, backend="loky")(
                    delayed(eval_fold)(params, fc) for params, fc in jobs
                )

            y_batch = []
            for i in range(0, len(fold_scores), n_inner_splits):
                mean_acc = np.mean(fold_scores[i:i+n_inner_splits])
                y_batch.append(-mean_acc)  # skopt minimizes

            opt.tell(X_batch, y_batch)

            for x, y in zip(X_batch, y_batch):
                score = -y
                if score > best_score:
                    best_score = score
                    best_params = {space[i].name: x[i] for i in range(len(space))}

        return make_clf(**best_params)

    # Train tuned model(s)
    if clf_str == "all":
        models = [tune_model(k) for k in clf_dict.keys()]
    else:
        models = [tune_model(clf_str)]

    # Final outer eval
    summary = linear_classifier(X_train, y_train, X_test, y_test, clfs_list=models, z_score=z_score)
    return summary


# def tune_model(model_key):
#     spec     = clf_dict[model_key]
#     space    = spec["space"]
#     make_clf = spec["make"]

#     @use_named_args(space)
#     def _objective(**params):
#         # score one cached fold
#         def _score(cache_item):
#             Xtr, ytr, Xval, yval = cache_item
#             mdl = make_clf(**params)
#             out = linear_classifier(Xtr, ytr, Xval, yval, clfs_list=[mdl], z_score=z_score)
#             yhat = next(iter(out.values()))['predictions']
#             return balanced_accuracy_score(yval, yhat)

#         # Use threads to avoid pickling big arrays; cap BLAS threads to 1
#         scores = Parallel(n_jobs=n_inner_splits)(
#             delayed(_score)(cache) for cache in fold_cache
#         )
#         return -float(np.mean(scores))

#     res = gp_minimize(_objective, space, n_calls=n_calls, n_initial_points=n_initial,
#                         random_state=random_state, acq_optimizer="lbfgs")
#     best_params = {dim.name: val for dim, val in zip(space, res.x)}
#     return make_clf(**best_params)
