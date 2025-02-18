import numpy as np
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space

import sys
sys.path.append('/utils')

from classification import linear_classifier
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

def tangent_classification(covs_train, y_train, covs_test, y_test, clf_str="SVM (C=1)", z_score=2, metric="riemann", deconf=False, con_confounder_train=None, cat_confounder_train=None, con_confounder_test=None, cat_confounder_test=None):
    X_train, X_test, _ = tangent_transform(covs_train, covs_test, metric=metric)
    if deconf:
        X_train, X_test = deconfound(X_train, con_confounder_train, cat_confounder_train, X_test=X_test, con_confounder_test=con_confounder_test, cat_confounder_test=cat_confounder_test)

    summary = linear_classifier(X_train, y_train, X_test, y_test, clf_str=clf_str, z_score=z_score)

    return summary
