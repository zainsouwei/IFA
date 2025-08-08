import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.svm import LinearSVC
from pathlib import Path
from scipy.stats import entropy, iqr, differential_entropy
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import gp_minimize
from skopt.space import Real
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, recall_score
from joblib import Parallel, delayed
from scipy.stats import pearsonr  # local import to avoid clutter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space
from skopt.utils import use_named_args

import sys
sys.path.append('/project/3022057.01/IFA/utils')

import regression
import tangent
from importlib import reload

# after editing regression.py:
reload(regression)

# now pull in your updated functions
from regression import deconfound
# ────────────────────────────────────────────────────────────────
#  MODEL_REGISTRY  ── one stop for every classifier you want to try
# ────────────────────────────────────────────────────────────────
from functools   import partial
from skopt.space import Real
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

MODEL_REGISTRY = {
    #  LINEAR SVC variants
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


def tangent_transform(train, test=None, *, metric="riemann", labels=None, balance=False):
    """
    Parameters
    ----------
    train  : (N, P, P) array            SPD covariances for TRAIN samples
    test   : (M, P, P) array | None     optional TEST covariances
    labels : (N,) array | None          class labels for TRAIN samples
    balance: bool                       if True → equal-class Frechét mean

    Returns
    -------
    X_train , X_test (if test not None), Mean
        Tangent-space projections and the reference SPD mean.
    """
    # 1) choose the reference SPD matrix
    if balance and (labels is not None):
        # (a) compute one class-specific Frechét mean each
        class_means = [
            mean_covariance(train[labels == c], metric=metric)
            for c in np.unique(labels)
        ]
        class_means = np.stack(class_means)           # shape = (n_classes, P, P)

        # (b) *equal-weight* Frechét mean of those class means
        Frechet_Mean = mean_covariance(class_means, metric=metric)
    else:
        # original behaviour – single mean over all samples
        Frechet_Mean = mean_covariance(train, metric=metric)

    # 2) project TRAIN (and optionally TEST) into tangent space
    X_train = tangent_space(train, Frechet_Mean, metric=metric)
    if test is None:
        return X_train, Frechet_Mean

    X_test  = tangent_space(test,  Frechet_Mean, metric=metric)
    return X_train, X_test, Frechet_Mean


def sliding_window_transformation(data, *, window, overlap_perc=0.1, cov_meth='oas'):
    """
    Return a list of (n_windows_i, P, P) covariance stacks, one per subject.
    `window` is **in samples**. `overlap_perc` is a fraction (e.g. 0.2 → 20 %).
    """
    stride = max(1, int(round(window * (1 - overlap_perc))))
    cov_est = Covariances(estimator=cov_meth)

    out = []
    for ts in data:                       # ts : (T, P)
        T, _ = ts.shape
        win_starts = range(0, T - window + 1, stride)
        covs = [cov_est.transform(ts[s:s+window].T[np.newaxis])[0] for s in win_starts]
        out.append(np.stack(covs, axis=0))
    return out

def balance_windows(cov_list, labels, *, con_confounders=None, cat_confounders=None, random_state=42, method='undersample'):
    rng = np.random.RandomState(random_state)

    win_counts = np.fromiter((c.shape[0] for c in cov_list), int)
    X = np.concatenate(cov_list, axis=0)

    def _repeat(df):
        return (df.loc[df.index.repeat(win_counts)].reset_index(drop=True)) if df is not None else None

    con_rep = _repeat(con_confounders)
    cat_rep = _repeat(cat_confounders)

    y = np.repeat(labels, win_counts)
    cls, cnts = np.unique(y, return_counts=True)
    tgt_n, replace = (cnts.min(), False) if method=='undersample' else (cnts.max(), True)

    sel = np.hstack([resample(np.where(y==c)[0], replace=replace, n_samples=tgt_n, random_state=rng) for c in cls])
    rng.shuffle(sel)

    return (X[sel],
            y[sel],
            con_rep.iloc[sel] if con_rep is not None else None,
            cat_rep.iloc[sel] if cat_rep is not None else None)


def prepare_fold(
        subj_df,               # the 1‑row‑per‑subject dataframe
        tr_idx, te_idx,        # indices from KFold / GroupKFold …
        *,                     # keyword‑only from here on
        deconf=True,
        metric="riemann",
        tangent_balanced=False, 
        group_col=None,
        normalize=True       
):
    # ────────────────────────────── predictors (brain) ──────────────
    covs_tr = np.stack(subj_df["cov"].iloc[tr_idx])
    covs_te = np.stack(subj_df["cov"].iloc[te_idx])
    
    y_tr = None
    if tangent_balanced:
        if group_col is None:
            raise ValueError("Need `group_col` when tangent_balanced=True")
        y_tr = subj_df[group_col].to_numpy()[tr_idx]

    # 1) tangent projection
    X_tr, X_te, _ = tangent_transform(covs_tr, covs_te, metric=metric,labels=y_tr, balance=tangent_balanced)
    
    if deconf:
        X_tr, X_te = deconfound(
            X_tr,
            pd.DataFrame(subj_df["con_base"].tolist()).iloc[tr_idx],
            pd.DataFrame(subj_df["cat_base"].tolist()).iloc[tr_idx],
            X_test=X_te,
            con_confounder_test=pd.DataFrame(subj_df["con_base"].tolist()).iloc[te_idx],
            cat_confounder_test=pd.DataFrame(subj_df["cat_base"].tolist()).iloc[te_idx]
        )
    
    if normalize:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    return X_tr, X_te


def augementer(
    subj_df, tr_idx, te_idx,
    *,
    augment,
    window=None,
    overlap_perc=0.1,
    balance="oversample",
    deconf=False,
    metric="riemann",
    tangent_balanced=False,
    group_col=None,
    normalize=False,
    random_state=42
):
    """
    Unified data preparation for a single train/test split.

    Returns:
      X_tr, X_te, y_tr, con_rep, cat_rep
    """
    # CASE 1: no window augmentation
    if augment in ("none", "smote", "undersample"):
        # subject-level tangent projection (+ optional deconf + normalize)
        X_tr, X_te = prepare_fold(
            subj_df, tr_idx, te_idx,
            deconf=deconf,
            metric=metric,
            tangent_balanced=tangent_balanced,
            group_col=group_col,
            normalize=normalize
        )
        # labels for train
        y_tr = subj_df[group_col].to_numpy()[tr_idx]

        # optional resampling on train only
        if augment == "smote":
            # TODO switch to SVM smote
            X_tr, y_tr = SMOTE(random_state=random_state).fit_resample(X_tr, y_tr)
        elif augment == "undersample":
            X_tr, y_tr = RandomUnderSampler(random_state=random_state).fit_resample(X_tr, y_tr)

        return X_tr, X_te, y_tr, None, None

    # CASE 2: sliding-window augmentation
    elif augment == "windows":
        # 1) sliding-window covariances for train subjects
        win_covs = sliding_window_transformation(
            subj_df["data"].iloc[tr_idx].tolist(),
            window=window,
            overlap_perc=overlap_perc,
            cov_meth='oas'
        )
        # 2) subject-level confounders
        con_subj = pd.DataFrame(subj_df["con_base"].tolist()).iloc[tr_idx]
        cat_subj = pd.DataFrame(subj_df["cat_base"].tolist()).iloc[tr_idx]
        y_tr_subj = subj_df[group_col].to_numpy()[tr_idx]

        # 3) balance windows & repeat confounders
        X_cov_bal, y_tr, con_rep, cat_rep = balance_windows(
            win_covs,
            labels=y_tr_subj,
            con_confounders=con_subj,
            cat_confounders=cat_subj,
            method=balance,
            random_state=random_state
        )
        # 4) tangent-space projection using window-level X
        covs_te = np.stack(subj_df["cov"].iloc[te_idx])
        X_tr, X_te, _ = tangent_transform(
            X_cov_bal,
            covs_te,
            metric=metric,
            labels=y_tr,
            balance=tangent_balanced
        )
        # 5) optional deconfounding on windows
        if deconf and (con_rep is not None or cat_rep is not None):
            con_test = pd.DataFrame(subj_df["con_base"].tolist()).iloc[te_idx]
            cat_test = pd.DataFrame(subj_df["cat_base"].tolist()).iloc[te_idx]
            X_tr, X_te = deconfound(
                X_tr, con_rep, cat_rep,
                X_test=X_te,
                con_confounder_test=con_test,
                cat_confounder_test=cat_test
            )
        # 6) optional normalization
        if normalize:
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_te = scaler.transform(X_te)

        return X_tr, X_te, y_tr, con_rep, cat_rep

    else:
        raise ValueError(f"augment must be 'none','smote','undersample' or 'windows', got {augment!r}")


def tune_model_on_subset(
        train_df, model_key,
        *,                         # data-pipeline kwargs
        augment, window, overlap_perc, balance,
        deconf, metric, tangent_balanced, group_col,
        normalize, random_state,
        n_calls=25, n_initial=6,
        n_inner_splits=3):
    """
    Bayesian hyper-parameter search on *train_df* only.
    "Returns (best_params_dict)"
    """
    spec       = MODEL_REGISTRY[model_key]
    space      = spec["space"]
    make_clf   = spec["make"]

    y_train = train_df[group_col].to_numpy()
    n_train = len(train_df)
    inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True,random_state=random_state)

    @use_named_args(space)
    def _objective(**params):
        def _score_fold(tr_idx, val_idx):
            X_tr, X_val, y_tr, _, _ = augementer(
                train_df, tr_idx, val_idx,
                augment=augment, window=window,
                overlap_perc=overlap_perc, balance=balance,
                deconf=deconf, metric=metric,
                tangent_balanced=tangent_balanced,
                group_col=group_col, normalize=normalize,
                random_state=random_state
            )
            clf = make_clf(**params)
            clf.fit(X_tr, y_tr)
            # return balanced_accuracy_score(y_train[val_idx], clf.predict(X_val))
            # return accuracy_score(y_train[val_idx],  clf.predict(X_val))
            return f1_score(y_train[val_idx],  clf.predict(X_val), average="macro")
            # return f1_score(y_train[val_idx],  clf.predict(X_val), average="micro")

        scores = Parallel(n_jobs=n_inner_splits)(
            delayed(_score_fold)(tr, val)
            for tr, val in inner_cv.split(np.arange(n_train), y=y_train)
        )
        return -np.mean(scores)          # negate → minimise
    
    x0 = None
    if model_key == "logreg_en":
        x0 = [[0.1, 0.1],[1, 1]]          #  C=0.1, l1_ratio=0.1
    if model_key == "logreg_l1":
        x0 = [[1.0]]
    res = gp_minimize(
        _objective, space, x0=x0, y0=None,
        n_calls=n_calls, n_initial_points=n_initial,
        acq_optimizer="lbfgs",
        random_state=random_state, verbose=False
    )

    best_params = {d.name: v for d, v in zip(space, res.x)}

    return best_params

def stringify_label_filter(label_filter):
    if label_filter is None:
        return "all"
    elif all(isinstance(el, int) for el in label_filter):
        return "_".join(map(str, label_filter))
    elif all(isinstance(el, (list, tuple)) for el in label_filter):
        return "__".join(["_".join(map(str, group)) for group in label_filter])
    else:
        raise ValueError("label_filter must be list of ints or list of lists/tuples of ints")


def class_analysis(
        df,
        covs,
        data,
        *,
        deconf=False,
        con_cols=None,
        cat_cols=None,
        group_col="group_numeric_label",
        subject_col="Subject",
        tp_col="TimepointNr",
        label_filter=None,
        tp0=0,
        tp1=1,
        dpi=110,
        save_dir=None,
        n_splits=4,
        random_state=0,
        alpha=None,
        metric="riemann",
        augment="none",   # "none" | "smote" | "windows" | "undersample"
        window=None,             # used only if augment=="windows"
        overlap_perc=0.1,
        balance="oversample",     # "oversample" | "undersample"  (windows mode)
        tangent_balanced=False,
        normalize=False,
        model_key="svc_l2_sq",    

):

    # 0) start with “keep everything”
    keep_mask = np.ones(len(df), dtype=bool)

    # 1) label filter  (does NOT shrink df yet)
    if label_filter is not None:
        groups  = (label_filter if any(isinstance(el, (list, tuple))
                                       for el in label_filter)
                   else [[lbl] for lbl in label_filter])
        mapping = {old: new for new, grp in enumerate(groups) for old in grp}

        mask_lbl = df[group_col].isin(mapping)
        keep_mask &= mask_lbl.values

    # 2) literal “na/NAN/…” → np.nan
    df = df.replace(r"(?i)^\s*na[n]?\s*$", np.nan, regex=True)

    # 3) subject‑level QC
    con_cols = con_cols or []
    cat_cols = cat_cols or []

    mask_subj = np.zeros(len(df), dtype=bool)   # will be True for good rows
    for subj, grp in df.groupby(subject_col):
        ok  =  set(grp[tp_col]) == {tp0, tp1}
        if deconf:
            ok &= not grp[con_cols + cat_cols].isna().any().any()
        if ok:
            mask_subj[grp.index] = True

    keep_mask &= mask_subj

    # 4) apply the single, final mask
    df   = df[keep_mask].reset_index(drop=True)
    covs = covs[keep_mask]                 # rows still match 1‑to‑1
    # --  mask the list ‘data’ robustly ----------------------------
    if isinstance(data, (np.ndarray, pd.Series)):
        data = data[keep_mask]
    else:                                          # plain Python list
        data = [d for d, keep in zip(data, keep_mask) if keep]
    print(f"Kept {len(df)} sessions  covs shape = {covs.shape} len data = {len(data)}")

    if df.empty:
        print("No subjects left after filtering."); return

    # 5)  remap group labels now that df is final
    if label_filter is not None:
        df[group_col] = df[group_col].map(mapping).astype(int)

    df["covs"] = list(covs)
    df["data"] = list(data)

    # 6) collapse to one row per subject (aligned arrays)
    rows = []
    for subj, grp in df.groupby(subject_col):
        base = grp.loc[grp[tp_col] == tp0].iloc[0]   # TP0
        foll = grp.loc[grp[tp_col] == tp1].iloc[0]   # TP1

        rows.append({
            "subj"   : subj,
            "cov"    : base["covs"],                       # ← baseline connectome
            "data"   : base["data"],                        # ← baseline data
            "con_base" : (base[con_cols] if con_cols else None),
            "cat_base" : (base[cat_cols] if cat_cols else None),
            "con_foll" : (foll[con_cols] if con_cols else None),
            "cat_foll" : (foll[cat_cols] if cat_cols else None),
            f"{group_col}"  : int(base[group_col]),
        })

    subj_df = pd.DataFrame(rows)          # one row per subject
    print(f"Kept {len(subj_df)} subjects")
   

    y_strat = subj_df[group_col]
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    label_str = stringify_label_filter(label_filter)
    log_dir = Path(f"/project/3022057.01/POM/POM_Analysis/best_params_logs/{label_str}/{model_key}")
    log_dir.mkdir(parents=True, exist_ok=True)



    def _process_fold_mode(subj_df, tr_idx, te_idx, *, deconf, metric, group_col, alpha,
                                   augment, window, overlap_perc, balance, tangent_balanced, normalize, model_key, log_dir, fold_id, random_state): 
        # 1) prepare data
        X_tr, X_te, y_tr, con_rep, cat_rep = augementer(
            subj_df, tr_idx, te_idx,
            augment=augment,
            window=window,
            overlap_perc=overlap_perc,
            balance=balance,
            deconf=deconf,
            metric=metric,
            tangent_balanced=tangent_balanced,
            group_col=group_col,
            normalize=normalize,
            random_state=random_state
        )

        # 2) ground truth
        Y_true = subj_df[group_col].to_numpy()[te_idx]
        Y_pred = np.full(X_te.shape[0], np.nan)

        if alpha is None:                                    # let BayesOpt decide
            train_df = subj_df.iloc[tr_idx].reset_index(drop=True)

            best_params = tune_model_on_subset(
                train_df,
                model_key=model_key,          #  pick any key from MODEL_REGISTRY
                augment=augment, window=window, overlap_perc=overlap_perc,
                balance=balance, deconf=deconf, metric=metric,
                tangent_balanced=tangent_balanced, group_col=group_col,
                normalize=normalize, random_state=random_state,
                n_calls=25, n_initial=6, n_inner_splits=10
            )

            log_file =  Path(log_dir,f"fold_{fold_id}_best_params.txt")
            with open(log_file, "w") as f:
                for k, v in best_params.items():
                    f.write(f"{k}: {v}\n")

            clf = MODEL_REGISTRY[model_key]["make"](**best_params)
        else:                                  
            clf = MODEL_REGISTRY[model_key]["make"](**(alpha if isinstance(alpha, dict) else {"C": alpha}))
            
        clf.fit(X_tr, y_tr)
        Y_pred = clf.predict(X_te)

        # 3) return everything the caller needs
        out = {
            "te_idx"   : te_idx,
            "Y_true"   : Y_true,        # whatever ground truth we just used
            "Y_pred"   : Y_pred,
        }
        return out

    # build list of jobs
    jobs = []
    for fold_id, (tr_idx, te_idx) in enumerate(kf.split(subj_df,y=y_strat)):
        jobs.append(delayed(_process_fold_mode)(
            subj_df, tr_idx, te_idx,
            deconf=deconf,
            metric=metric,
            group_col=group_col,
            alpha=alpha,
            augment=augment,
            window=window,
            overlap_perc=overlap_perc,
            balance=balance,
            tangent_balanced=tangent_balanced,
            normalize= normalize,
            model_key=model_key,
            log_dir=log_dir,
            fold_id=fold_id,           
            random_state=random_state          # pass it down so SMOTE & resample agree

        ))

    # run in parallel  (use n_jobs=-1 for all cores)
    results_list = Parallel(n_jobs=n_splits, verbose=1)(jobs)
        
    # 9) aggregate outputs  ─────────────────────────────────────────────
    n_subj = len(subj_df)

    store = {"true": np.full((n_subj), np.nan), "pred": np.full((n_subj), np.nan)}

    for res in results_list:
        # subject-ordered containers
        store["true"][res["te_idx"]] = res["Y_true"]
        store["pred"][res["te_idx"]] = res["Y_pred"]

    y_true_cls  = store["true"]
    y_pred_cls  = store["pred"]

    # ── confusion-matrix heat map ────────────────────────────────────
    cm     = confusion_matrix(y_true_cls, y_pred_cls, labels=np.unique(y_true_cls))
    cm_pct = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
    im = ax.imshow(cm_pct, cmap=plt.cm.Blues)

    acc  = accuracy_score(y_true_cls, y_pred_cls)
    f1   = f1_score(y_true_cls, y_pred_cls, average="macro")
    f1_mic   = f1_score(y_true_cls, y_pred_cls, average="micro")
    bacc = balanced_accuracy_score(y_true_cls, y_pred_cls)

    ax.set(title=f"Acc {acc:.2%}  |  Bal Acc {bacc:.2%}  |  F1 (macro) {f1:.2f} |  F1 (micro) {f1_mic:.2f}", xlabel="Predicted label", ylabel="True label", xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(np.unique(y_true_cls), rotation=45)
    ax.set_yticklabels(np.unique(y_true_cls))

    thresh = 0.5 * np.nanmax(cm_pct)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            val, pct = cm[row, col], cm_pct[row, col]
            ax.text(col, row, f"{val}\n({pct:.1%})",
                    ha="center", va="center",
                    color="white" if pct > thresh else "black")
    fig.colorbar(im, ax=ax, label="Fraction of true class")
    fig.tight_layout(); plt.show()


