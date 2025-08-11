import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.base import clone
from sklearn.utils.multiclass import unique_labels
from functools   import partial
from skopt.space import Real

clf_dict = {
    #  LINEAR SVC variants
    "svc": {
        "make": partial(SVC,kernel='linear', class_weight='balanced'),
        "space": [Real(1e-6, 1e3, name="C", prior="log-uniform")]

    },
    "svc_l2_sq": {                         #  L2-penalty, squared-hinge (default)
        "make": partial(LinearSVC,
                        penalty="l2",
                        loss="squared_hinge",
                        dual="auto",
                        class_weight="balanced"),
        "space": [Real(1e-6, 1e3, name="C", prior="log-uniform")]
    },
    "svc_l2_hinge": {                      #  L2-penalty, classic hinge
        "make": partial(LinearSVC,
                        penalty="l2",
                        loss="hinge",
                        dual=True,              # hinge ⇒ dual must be True
                        class_weight="balanced"),
        "space": [Real(1e-6, 1e3, name="C", prior="log-uniform")]
    },
    "svc_l1": {                            #  L1-penalty (sparse weights)
        "make": partial(LinearSVC,
                        penalty="l1",
                        loss="squared_hinge",
                        dual=False,             # L1 ⇒ dual must be False
                        class_weight="balanced"),
        "space": [Real(1e-6, 1e3, name="C", prior="log-uniform")]
    },

    #  LOGISTIC-REGRESSION variants
    "logreg_l2": {                         #  pure L2
        "make": partial(LogisticRegression,
                        penalty="l2",
                        solver="saga",
                        class_weight="balanced",
                        ),
        "space": [Real(1e-6, 1e3, name="C", prior="log-uniform")]
    },
    "logreg_l1": {                         #  pure L1
        "make": partial(LogisticRegression,
                        penalty="l1",
                        solver="saga",
                        class_weight="balanced",
                        ),
        "space": [Real(1e-6, 1e3, name="C", prior="log-uniform")]
    },
    # TODO Decide on whether or not to use class balance weights 
    "logreg_en": {                         #  elastic-net (tune C & l1_ratio)
        "make": partial(LogisticRegression,
                        penalty="elasticnet",
                        solver="saga",
                        class_weight="balanced",
                        ),
        "space": [
            Real(1e-6, 1e3, name="C",        prior="log-uniform"),
            Real(1e-3,   1.0, name="l1_ratio", prior="uniform")
        ]
    },
}

def name_estimator(est, keys=('C','penalty','loss','alpha','l1_ratio','kernel')):
    # if Pipeline, look at final step
    if hasattr(est, "steps"):
        est = est.steps[-1][1]
    base = est.__class__.__name__
    params = est.get_params(deep=False)
    picks = []
    for k in keys:
        if k in params:
            v = params[k]
            if isinstance(v, float):
                v = f"{v:.3g}"
            picks.append(f"{k}={v}")
    return f"{base}[{', '.join(picks)}]" if picks else base

def linear_classifier(X_train, y_train, X_test, y_test, clfs_list, z_score=2):
    # Clone to avoid modifying the original object    
    if z_score == 1:
        scaler = StandardScaler(with_mean=True, with_std=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif z_score == 2:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif z_score == 3:
        # Robust scaling
        scaler = RobustScaler(with_centering=True)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    metrics_dict = {}
    for model in clfs_list:
        clf = clone(model)

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        correct_predictions = np.sum(predictions == y_test)
        total_predictions = len(y_test)
        # Confusion matrix to get per-class accuracy
        labels = unique_labels(y_test, predictions)
        cm = confusion_matrix(y_test, predictions, labels=labels)
        per_class_correct = np.diag(cm)
        per_class_total = np.sum(cm, axis=1)
        per_class_accuracy = per_class_correct / per_class_total

        summary = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'per_class_correct': per_class_correct,
            'per_class_total': per_class_total,
            'per_class_accuracy': per_class_accuracy,
            'labels': labels,
            'predictions': predictions,
        }
        
        clf_name = name_estimator(clf)
        metrics_dict[clf_name] = summary
    return metrics_dict