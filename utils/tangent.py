import numpy as np
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space

import sys
sys.path.append('/utils')

from classification import clf_dict, linear_classifier

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

def tangent_classification(covs_train, y_train, covs_test, y_test, clf=clf_dict["SVM (C=1)"], z_score=2, metric="riemann"):
    X_train, X_test, _ = tangent_transform(covs_train, covs_test, metric=metric)
    summary = linear_classifier(X_train, y_train, X_test, y_test, clf=clf, z_score=z_score)

    return summary
