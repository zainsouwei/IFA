import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

clf_dict = {
        "SVM (C=1)": SVC(kernel='linear', C=1, class_weight='balanced'),
        "SVM (C=0.1)": SVC(kernel='linear', C=0.1, class_weight='balanced'),
        "SVM (C=0.01)": SVC(kernel='linear', C=0.01, class_weight='balanced'),
        # "SVM (C=0.001)": SVC(kernel='linear', C=0.001, class_weight='balanced'),
        # "L2 SVM (C=1)": LinearSVC(penalty='l2',loss='squared_hinge',C=1,class_weight='balanced'),
        "L2 SVM (C=0.1)": LinearSVC(penalty='l2',loss='squared_hinge',C=.1,class_weight='balanced'),
        "L2 SVM (C=0.01)":  LinearSVC(penalty='l2',loss='squared_hinge',C=.01,class_weight='balanced'),
        # "L2 SVM (C=0.001)":  LinearSVC(penalty='l2',loss='squared_hinge',C=.001,class_weight='balanced'),
        # "L2 SVM Hinge (C=1)": LinearSVC(penalty='l2',loss='hinge',C=1,class_weight='balanced'),
        # "L2 SVM Hinge (C=0.1)": LinearSVC(penalty='l2',loss='hinge',C=.1,class_weight='balanced'),
        # "L2 SVM Hinge (C=0.01)":  LinearSVC(penalty='l2',loss='hinge',C=.01,class_weight='balanced'),
        # "L1 SVM (C=1)": LinearSVC(penalty='l1',loss='squared_hinge',dual=False,C=1,class_weight='balanced'),
        # "L1 SVM (C=0.1)": LinearSVC(penalty='l1',loss='squared_hinge',dual=False,C=.1,class_weight='balanced'),
        # "L1 SVM (C=0.01)":  LinearSVC(penalty='l1',loss='squared_hinge',dual=False,C=.01,class_weight='balanced'),
        "LDA": LDA(),
        "Logistic Regression": LogisticRegression(),
        "Logistic Regression (l2 C=1)": LogisticRegression(penalty='l2', C=1, class_weight='balanced'),
        "Logistic Regression (l2 C=10)": LogisticRegression(penalty='l2', C=10, class_weight='balanced'),
        # "Logistic Regression (l1)": LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced'),
        "Logistic Regression (elasticnet C=1)": LogisticRegression(penalty='elasticnet', C=1, solver='saga', l1_ratio=0.1, class_weight='balanced'),
        "Logistic Regression (elasticnet C=10)": LogisticRegression(penalty='elasticnet', C=10, solver='saga', l1_ratio=0.1, class_weight='balanced')
    }

def linear_classifier(X_train, y_train, X_test, y_test, clf_str='SVM (C=1)', z_score=2):
    if clf_str == 'all':
        clf_strs = [key for key, _ in clf_dict.items()]
    else:
        clf_strs = [clf_str]
    
    metrics_dict = {}
    for clf_name in clf_strs:
        clf = clf_dict[clf_name]
        if z_score == 1:
            scaler = StandardScaler(with_mean=True, with_std=False)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        elif z_score == 2:
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        correct_predictions = np.sum(predictions == y_test)
        total_predictions = len(y_test)
        # Confusion matrix to get per-class accuracy
        unique_labels = np.unique(y_test)
        cm = confusion_matrix(y_test, predictions, labels=[unique_labels[1], unique_labels[0]])
        per_class_correct = np.diag(cm)
        per_class_total = np.sum(cm, axis=1)
        per_class_accuracy = per_class_correct / per_class_total

        summary = {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'per_class_correct': per_class_correct,
            'per_class_total': per_class_total,
            'per_class_accuracy': per_class_accuracy
        }
        metrics_dict[clf_name] = summary
    return metrics_dict