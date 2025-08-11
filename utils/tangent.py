import numpy as np
from skopt import gp_minimize
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from joblib import Parallel, delayed
from skopt.utils import use_named_args

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


def tangent_classification(covs_train, y_train, covs_test, y_test, clf_str="logreg_en", z_score=2, metric="riemann", 
                           deconf=False, con_confounder_train=None, cat_confounder_train=None,
                           con_confounder_test=None, cat_confounder_test=None,
                           random_state=0, n_inner_splits=10,n_cpus=15, n_calls=25, n_initial=6):

    # Outer transform
    X_train, X_test, _ = tangent_transform(covs_train, covs_test, metric=metric)
    if deconf:
        X_train, X_test = deconfound(
            X_train, con_confounder_train, cat_confounder_train,
            X_test=X_test, con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test
        )

    def tune_model(model_key):
        spec     = clf_dict[model_key]
        space    = spec["space"]
        make_clf = spec["make"]
        cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)

        @use_named_args(space)
        def _objective(**params):
            def _score_fold(tr_idx, val_idx):
                # RAW → fold-specific tangent (+ optional deconf)
                covs_in_tr, covs_in_val = covs_train[tr_idx], covs_train[val_idx]
                y_in_tr,   y_in_val     = y_train[tr_idx],   y_train[val_idx]
                X_in_tr, X_in_val, _ = tangent_transform(covs_in_tr, covs_in_val, metric=metric)

                if deconf:
                    X_in_tr, X_in_val = deconfound(
                        X_in_tr,
                        None if con_confounder_train is None else con_confounder_train.iloc[tr_idx],
                        None if cat_confounder_train is None else cat_confounder_train.iloc[tr_idx],
                        X_test=X_in_val,
                        con_confounder_test=None if con_confounder_train is None else con_confounder_train.iloc[val_idx],
                        cat_confounder_test=None if cat_confounder_train is None else cat_confounder_train.iloc[val_idx],
                    )

                tuned = make_clf(**params)
                fold_metrics = linear_classifier(X_in_tr, y_in_tr, X_in_val, y_in_val,
                                                clfs_list=[tuned], z_score=z_score)
                summ = next(iter(fold_metrics.values()))
                y_pred = summ['predictions']

                return balanced_accuracy_score(y_in_val, y_pred)
                # return accuracy_score(y_in_val,  y_pred)
                # return f1_score(y_in_val,  y_pred, average="macro")
                # return f1_score(y_in_val,  y_pred, average="micro")

            scores = Parallel(n_jobs=n_inner_splits)(
                delayed(_score_fold)(tr, val) for tr, val in cv.split(covs_train, y_train)
            )
            return -float(np.mean(scores))  # minimize

        res = gp_minimize(_objective, space, n_calls=n_calls, n_initial_points=n_initial, random_state=random_state)
        best_params = {dim.name: val for dim, val in zip(space, res.x)}
        return make_clf(**best_params)

    # Build tuned model list
    # TODO come back to this here (parallel  cpus/n_inner pslots if < 0 then 1 amnd alays a hwole number )
    if clf_str == "all":
        nj = n_cpus // max(1, n_inner_splits)
        nj = max(1, int(nj))
        keys = list(clf_dict.keys())
        models = Parallel(n_jobs=nj)(delayed(tune_model)(k) for k in keys)
    else:
        models = [tune_model(clf_str)]

    # Final outer evaluation using your helper
    summary = linear_classifier(X_train, y_train, X_test, y_test, clfs_list=models, z_score=z_score)
    return summary