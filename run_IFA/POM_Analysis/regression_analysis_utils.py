import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.svm import LinearSVC, LinearSVR
from pathlib import Path
from scipy.stats import entropy, iqr, differential_entropy
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score
from joblib import Parallel, delayed
from scipy.stats import pearsonr 
from functools   import partial

import sys
sys.path.append('/project/3022057.01/IFA/utils')

import regression
import tangent
from importlib import reload

# after editing regression.py:
reload(regression)
reload(tangent)

# now pull in your updated functions
from regression import deconfound
from tangent    import tangent_transform



# ---------- which models can be tuned -------------
CLS_REG = {
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

# TODO should i tune epsilon?
REG_REG = {                         # regression
    "ridge":     dict(make=partial(Ridge),        # Ridge α to be tuned
                       space=[Real(1e-6,1e2,"log-uniform",name="alpha")]),
    "enet":      dict(make=partial(ElasticNet),
                     space=[Real(1e-6, 1e3, name="alpha", prior="log-uniform"),
                            Real(1e-3,   1.0, name="l1_ratio", prior="uniform")]),
    "svr_l1":      dict(make=partial(LinearSVR, loss="epsilon_insensitive"),
                       space=[Real(1e-6,1e3,"log-uniform",name="C")]),
    "svr_l2":      dict(make=partial(LinearSVR, loss="squared_epsilon_insensitive"),
                       space=[Real(1e-6,1e3,"log-uniform",name="C")]),
}


def prepare_fold(
        subj_df,               # the 1‑row‑per‑subject dataframe
        tr_idx, te_idx,        # indices from KFold / GroupKFold …
        mode,                  # 0 = t0 , 1 = t1 , 2 = Δ(/week)
        *,                     # keyword‑only from here on
        deconf=True,
        target_latent="raw",   # "raw" | "pca" | "lda" | "plsc"
        n_components=3,
        metric="riemann",
        per_week=True,
        group_col="group_numeric_label",
        target_cols=None,
        quantile=None,
        normalize=False
):
    # ────────────────────────────── predictors (brain) ──────────────
    covs_tr = np.stack(subj_df["cov"].iloc[tr_idx])
    covs_te = np.stack(subj_df["cov"].iloc[te_idx])

    # 1) tangent projection
    X_tr, X_te, _ = tangent_transform(covs_tr, covs_te, metric=metric)

    # 2) de‑confound brain (baseline confounds only)
    
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
    # ────────────────────────────── targets  ────────────────────────
    Y0_tr_raw = np.vstack(subj_df["Y0"].iloc[tr_idx])
    Y1_tr_raw = np.vstack(subj_df["Y1"].iloc[tr_idx])
    Y0_te_raw = np.vstack(subj_df["Y0"].iloc[te_idx])
    Y1_te_raw = np.vstack(subj_df["Y1"].iloc[te_idx])

    if deconf:
        Y0_tr_raw, Y0_te_raw = deconfound(
            Y0_tr_raw,
            pd.DataFrame(subj_df["con_base"].tolist()).iloc[tr_idx],
            pd.DataFrame(subj_df["cat_base"].tolist()).iloc[tr_idx],
            X_test=Y0_te_raw,
            con_confounder_test=pd.DataFrame(subj_df["con_base"].tolist()).iloc[te_idx],
            cat_confounder_test=pd.DataFrame(subj_df["cat_base"].tolist()).iloc[te_idx]
        )
        Y1_tr_raw, Y1_te_raw = deconfound(
            Y1_tr_raw,
            pd.DataFrame(subj_df["con_foll"].tolist()).iloc[tr_idx],
            pd.DataFrame(subj_df["cat_foll"].tolist()).iloc[tr_idx],
            X_test=Y1_te_raw,
            con_confounder_test=pd.DataFrame(subj_df["con_foll"].tolist()).iloc[te_idx],
            cat_confounder_test=pd.DataFrame(subj_df["cat_foll"].tolist()).iloc[te_idx]
        )

    # pick the right raw block for t0 / t1
    if   mode == 0:  Y_tr_raw, Y_te_raw = Y0_tr_raw, Y0_te_raw
    elif mode == 1:  Y_tr_raw, Y_te_raw = Y1_tr_raw, Y1_te_raw
    else:            # Δ will be built later
            Y_tr_raw = Y_te_raw = None   # placeholders

    # ───────── optional latent basis fitted on *baseline‑train* ───────
    if target_latent.lower() == "raw":
        Y_tr, Y_te = Y_tr_raw, Y_te_raw
        names = target_cols.copy() if target_cols else []
        load = None
        dr = None                     # avoid any accidental reference later
        scaler = None
    else:
        if mode == 1: base_tr = Y1_tr_raw
        else: base_tr = Y0_tr_raw

        scaler   = StandardScaler().fit(base_tr)

        if target_latent.lower() == "pca":
            k   = min(n_components, base_tr.shape[1])
            dr = PCA(
                n_components=k,
                svd_solver="full",       # full SVD, deterministic
                random_state=None        # ignored by 'full'
            ).fit(scaler.transform(base_tr))
            names, load = [f"PC{i+1}" for i in range(k)], dr.components_
        elif target_latent.lower() == "plsc":               # ← NEW
            k   = min(n_components, base_tr.shape[1])
            dr  = PLSCanonical(n_components=k, scale=False).fit(X_tr,scaler.transform(base_tr))
            names = [f"PLSC{i+1}" for i in range(k)]
            load  = dr.y_loadings_.T                        # (k × q)
        else:  # "lda"
            lbl_tr = subj_df[group_col].iloc[tr_idx].astype(int).to_numpy()
            k   = min(n_components, len(np.unique(lbl_tr))-1)
            dr = LDA(n_components=k).fit(scaler.transform(base_tr), lbl_tr)
            names, load = [f"LD{i+1}" for i in range(k)], dr.scalings_[:, :k]

        # transform whichever block we have (t0 or t1)
        if mode in (0,1):
            if target_latent.lower() == "plsc":
                _, Y_tr = dr.transform(X_tr,scaler.transform(Y_tr_raw))
                _ ,Y_te = dr.transform(X_te,scaler.transform(Y_te_raw))
            else:
                Y_tr = dr.transform(scaler.transform(Y_tr_raw))
                Y_te = dr.transform(scaler.transform(Y_te_raw))

    # ───────────── Δ in latent space (change‑of‑scores) ───────────────
    if mode == 2:
        if target_latent.lower() !="raw":
            if target_latent.lower() == "plsc":
                _, Y0_tr_raw = dr.transform(X_tr,scaler.transform(Y0_tr_raw))
                _, Y1_tr_raw = dr.transform(X_tr, scaler.transform(Y1_tr_raw))
                _, Y0_te_raw = dr.transform(X_te, scaler.transform(Y0_te_raw)) 
                _, Y1_te_raw = dr.transform(X_te, scaler.transform(Y1_te_raw))
            else:
                Y0_tr_raw = dr.transform(scaler.transform(Y0_tr_raw))
                Y1_tr_raw = dr.transform(scaler.transform(Y1_tr_raw))
                Y0_te_raw = dr.transform(scaler.transform(Y0_te_raw)) 
                Y1_te_raw = dr.transform(scaler.transform(Y1_te_raw))

        Y_tr = (Y1_tr_raw - Y0_tr_raw) / subj_df["weeks"].iloc[tr_idx].to_numpy()[:,None] if per_week else (Y1_tr_raw - Y0_tr_raw)
        Y_te = (Y1_te_raw - Y0_te_raw) / subj_df["weeks"].iloc[te_idx].to_numpy()[:,None] if per_week else (Y1_te_raw - Y0_te_raw)
        names = [f"Δ_{c}" for c in names] if names else []
    
    Y_tr_cls, Y_te_cls = None, None
    if quantile is not None:
        if not (0.0 < quantile < 0.5):
            raise ValueError("quantile must be between 0 and 0.5")

        n_targets = 1 if Y_tr.ndim == 1 else Y_tr.shape[1]
        Y_tr_cls = np.empty((Y_tr.shape[0], n_targets), dtype=int)
        Y_te_cls = np.empty((Y_te.shape[0], n_targets), dtype=int)

        for j in range(n_targets):
            y_tr_vec = Y_tr[:, j]
            y_te_vec = Y_te[:, j]

            cuts = np.quantile(y_tr_vec, [quantile, 1.0 - quantile])
            Y_tr_cls[:, j] = np.digitize(y_tr_vec, cuts, right=False)
            Y_te_cls[:, j] = np.digitize(y_te_vec, cuts, right=False)

        
        # for j in range(n_targets):
        #     plt.figure(figsize=(10, 6))
        #     plt.hist(Y_tr[:,j], alpha=0.5, label='All true values')
        #     plt.hist(Y_tr[Y_tr_cls[:,j] == 0,j], alpha=0.5, label='Top quantiles')
        #     plt.hist(Y_tr[Y_tr_cls[:,j] == np.max(Y_tr_cls[:,j]),j], alpha=0.5, label='Bottom quantiles')
        #     plt.xlabel('True phenotype values')
        #     plt.ylabel('Frequency')
        #     plt.title(f'Histogram of True {names[j]} Values with Extreme Tails')
        #     plt.legend()
        #     plt.tight_layout()
        #     plt.show()

    return X_tr, X_te, Y_tr, Y_te, names, load, Y_tr_cls, Y_te_cls


# ===============================================================
#  Hyper-param search that works on *pre-processed* (X, y) arrays
# ===============================================================
def select_hp_preprocessed(
        X, y, *,
        task,              # "reg" | "cls"
        model_key,         # e.g. "ridge" | "svc_l2_sq"
        random_state=0,
        n_splits=3,
        n_calls=25,
        n_initial=6):

    reg      = REG_REG if task == "reg" else CLS_REG
    space    = reg[model_key]["space"]
    make_est = reg[model_key]["make"]

    cv = (StratifiedKFold if task == "cls" else KFold)(n_splits=n_splits, shuffle=True, random_state=random_state)

    @use_named_args(space)
    def _objective(**params):
        def score_one(tr_idx, val_idx):                      # ← helper
            est = make_est(**params)
            est.fit(X[tr_idx], y[tr_idx])
            preds = est.predict(X[val_idx])
            return (f1_score(y[val_idx], preds, average="macro")
                    if task == "cls"
                    else r2_score(y[val_idx], preds))

        # ===========================
        # run inner-folds in parallel
        # ===========================
        scores = Parallel(n_jobs=n_splits)(
                    delayed(score_one)(tr, val)
                    for tr, val in cv.split(X, y if task == "cls" else None)
                 )

        return -np.mean(scores)          # gp_minimize minimises

    res = gp_minimize(_objective, space,
                      n_calls=n_calls,
                      n_initial_points=n_initial,
                      random_state=random_state,
                      acq_optimizer="lbfgs",
                      verbose=False)

    return {dim.name: val for dim, val in zip(space, res.x)}


def regression_analysis(
        df,
        covs,
        target_cols,
        *,
        target_latent="pca",
        n_components=3,
        deconf=False,
        con_cols=None,
        cat_cols=None,
        per_week=True,
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
        stratify=False,
        quantile=None,
        global_pre=False,
        normalize=False,
        model_key="svc_l2_sq"
):

    # ------------------------------------------------------------------
    # 0) start with “keep everything”
    # ------------------------------------------------------------------
    keep_mask = np.ones(len(df), dtype=bool)

    # ------------------------------------------------------------------
    # 1) label filter  (does NOT shrink df yet)
    # ------------------------------------------------------------------
    if label_filter is not None:
        groups  = (label_filter if any(isinstance(el, (list, tuple))
                                       for el in label_filter)
                   else [[lbl] for lbl in label_filter])
        mapping = {old: new for new, grp in enumerate(groups) for old in grp}

        mask_lbl = df[group_col].isin(mapping)
        keep_mask &= mask_lbl.values

    # ------------------------------------------------------------------
    # 2) literal “na/NAN/…” → np.nan
    # ------------------------------------------------------------------
    df = df.replace(r"(?i)^\s*na[n]?\s*$", np.nan, regex=True)

    # ------------------------------------------------------------------
    # 3) subject‑level QC
    # ------------------------------------------------------------------
    con_cols = con_cols or []
    cat_cols = cat_cols or []

    mask_subj = np.zeros(len(df), dtype=bool)   # will be True for good rows
    for subj, grp in df.groupby(subject_col):
        ok  =  set(grp[tp_col]) == {tp0, tp1}
        ok &= not grp[target_cols].isna().any().any()
        ok &= not pd.isna(grp.loc[grp[tp_col]==tp1, "WeeksToFollowUp"]).any()
        if deconf:
            ok &= not grp[con_cols + cat_cols].isna().any().any()
        if ok:
            mask_subj[grp.index] = True

    keep_mask &= mask_subj

    # ------------------------------------------------------------------
    # 4) apply the single, final mask
    # ------------------------------------------------------------------
    df   = df[keep_mask].reset_index(drop=True)
    covs = covs[keep_mask]                 # rows still match 1‑to‑1

    print(f"Kept {len(df)} sessions  "
          f"covs shape = {covs.shape}")

    if df.empty:
        print("No subjects left after filtering."); return

    # ------------------------------------------------------------------
    # 5)  remap group labels now that df is final
    # ------------------------------------------------------------------
    if label_filter is not None:
        df[group_col] = df[group_col].map(mapping).astype(int)

    df["covs"] = list(covs)

    # ------------------------------------------------------------------
    # 6) collapse to one row per subject (aligned arrays)
    # ------------------------------------------------------------------
    rows = []
    for subj, grp in df.groupby(subject_col):
        base = grp.loc[grp[tp_col] == tp0].iloc[0]   # TP0
        foll = grp.loc[grp[tp_col] == tp1].iloc[0]   # TP1

        rows.append({
            "subj"   : subj,
            "cov"    : base["covs"],                       # ← baseline connectome
            "Y0"     : base[target_cols].astype(float).values,
            "Y1"     : foll[target_cols].astype(float).values,
            "weeks"  : float(foll["WeeksToFollowUp"]),
            "con_base" : (base[con_cols] if con_cols else None),
            "cat_base" : (base[cat_cols] if cat_cols else None),
            "con_foll" : (foll[con_cols] if con_cols else None),
            "cat_foll" : (foll[cat_cols] if cat_cols else None),
            f"{group_col}"  : int(base[group_col]),
        })

    subj_df = pd.DataFrame(rows)          # one row per subject
    print(f"Kept {len(subj_df)} subjects")
    # print(subj_df.head())
    # print(pd.DataFrame(subj_df["con_base"].tolist()).head())
    # print(pd.DataFrame(subj_df["cat_base"].tolist()).head())

    if target_latent == 'raw':
        n_components = len(target_cols)
    else:
        n_components = min(n_components, len(target_cols))
        if target_latent == 'lda':
            lbls = subj_df[group_col].astype(int).to_numpy()
            n_components = min(n_components, len(np.unique(lbls))-1)
    
    print(n_components)

    if global_pre:
        idx_all  = np.arange(len(subj_df))          # every subject, once
        GLOBAL   = {}

        for mode, tag in enumerate(("t0", "t1", "delta")):
            _, _, Y_tr, Y_te, names, load, Y_tr_cls, Y_te_cls = prepare_fold(
                subj_df, idx_all, idx_all, mode,      # ← HERE: idx_all, idx_all, mode
                deconf          = deconf,
                target_latent   = target_latent,
                n_components    = n_components,
                metric          = metric,
                per_week        = per_week,
                group_col       = group_col,
                target_cols     = target_cols,
                quantile        = quantile
            )
            GLOBAL[tag]      = Y_tr                  # (Y_tr == Y_te)
            if Y_tr_cls is not None:                 # only if you asked for quantiles
                GLOBAL[f"{tag}_cls"] = Y_tr_cls
            if mode == 0:
                GLOBAL["names"] = names
        if load is not None:
            GLOBAL["load"]  = load
    else:
        GLOBAL = None
   
    if stratify:
        if global_pre and quantile is not None:
            # take just one column of the global class labels
            cls_all = GLOBAL["t0_cls"]
            y_strat = cls_all[:, 0] if cls_all.ndim == 2 else cls_all
        else:
            y_strat = subj_df[group_col]

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        y_strat = None
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


    def _process_fold_mode(subj_df, tr_idx, te_idx, mode_i, tag,
                        *, deconf, target_latent, n_components,
                        metric, per_week, group_col, target_cols, alpha,quantile,model_key,
                        global_pre=False, GLOBAL=None, normalize=False,random_state=0): 
        # 1) preprocessing
        X_tr, X_te, Y_tr, Y_te, names, load, Y_tr_cls, Y_te_cls = prepare_fold(
            subj_df, tr_idx, te_idx, mode_i,
            deconf=deconf,
            target_latent=target_latent,
            n_components=n_components,
            metric=metric,
            per_week=per_week,
            group_col=group_col,
            target_cols=target_cols,
            quantile=quantile,
            normalize=normalize
        )

        if global_pre:
            tag_name = {0: "t0", 1: "t1", 2: "delta"}[mode_i]

            # continuous targets
            Y_tr = GLOBAL[tag_name][tr_idx]
            Y_te = GLOBAL[tag_name][te_idx]

            # optional class labels
            if quantile is not None:
                cls = GLOBAL[f"{tag_name}_cls"]
                Y_tr_cls = cls[tr_idx]
                Y_te_cls = cls[te_idx]

            names = GLOBAL["names"]        # same for every fold
            load  = GLOBAL.get("load")     # may be absent for 'raw'

        # Decide once which targets we’ll treat as “ground truth”
        is_class = quantile is not None
        Y_true   = Y_te_cls if is_class else Y_te
        n_out    = Y_true.shape[1] if Y_true.ndim == 2 else 1

        Y_pred = np.empty((X_te.shape[0], n_out))
        Y_pred.fill(np.nan)

        for j in range(n_out):
            if is_class:
                m_tr = (Y_tr_cls[:,j] == 0) | (Y_tr_cls[:,j] == 2)
                m_te = (Y_te_cls[:,j] == 0) | (Y_te_cls[:,j] == 2)
                # TODO change so no leakage between validation and train set
                if alpha is None:
                    y_train   = Y_tr_cls[m_tr, j]
                    best_hp = select_hp_preprocessed(
                        X_tr[m_tr,:], y_train,
                        task        = "cls" if is_class else "reg",
                        model_key   = model_key,
                        random_state= random_state,
                        n_splits=10, n_calls=25, n_initial=6)
                    alpha = best_hp
            else:
                # TODO change so no leakage between validation and train set
                if alpha is None:
                    y_train   = Y_tr[:,j]
                    best_hp = select_hp_preprocessed(
                        X_tr, y_train,
                        task        = "cls" if is_class else "reg",
                        model_key   = model_key,
                        random_state= random_state,
                        n_splits=10, n_calls=25, n_initial=6)
                    alpha = best_hp

            est = (CLS_REG if is_class else REG_REG)[model_key]["make"](**alpha)

            # ------------ train & predict ------------
            
            if is_class:
                est.fit(X_tr[m_tr,:], Y_tr_cls[m_tr, j])
                Y_pred[m_te, j] = est.predict(X_te[m_te,:])
            else:
                est.fit(X_tr, Y_tr[:, j])
                Y_pred[:, j] = est.predict(X_te)

                
        # 3) return everything the caller needs
        out = {
            "tag"      : tag,
            "te_idx"   : te_idx,
            "Y_true"   : Y_true,        # whatever ground truth we just used
            "Y_pred"   : Y_pred,
            "names"    : names,
            "load"     : load,
            "mode_i"   : mode_i,
            # keep continuous test targets for plotting even in class mode
            "y_true_cont": Y_te
        }
        return out

    # ------------------------------------------------------------
    # build list of jobs
    # ------------------------------------------------------------
    jobs = []
    for _, (tr_idx, te_idx) in enumerate(kf.split(subj_df,y=y_strat)):
        for mode_i, tag in zip([0, 1, 2], ("t0", "t1", "delta")):
            jobs.append(delayed(_process_fold_mode)(
                subj_df, tr_idx, te_idx, mode_i, tag,
                deconf=deconf,
                target_latent=target_latent,
                n_components=n_components,
                metric=metric,
                per_week=per_week,
                group_col=group_col,
                target_cols=target_cols,
                alpha=alpha,
                quantile=quantile,
                model_key=model_key,
                global_pre= global_pre,   
                GLOBAL=GLOBAL,
                normalize=normalize,
                random_state=random_state
            ))

    # ------------------------------------------------------------
    # run in parallel  (use n_jobs=-1 for all cores)
    # ------------------------------------------------------------
    results_list = Parallel(n_jobs=-1, verbose=1)(jobs)
        
    # 9) aggregate outputs  ─────────────────────────────────────────────
    n_subj = len(subj_df)
    n_out  = (results_list[0]["Y_true"].shape[1]    # 1 → scalar targets
            if results_list[0]["Y_true"].ndim == 2 else 1)

    store = {k: {"true": np.full((n_subj, n_out), np.nan),
                "pred": np.full((n_subj, n_out), np.nan),
                "cont": np.full((n_subj, n_out), np.nan)}
            for k in ("t0", "t1", "delta")}

    fold_meta      = []                       # loadings + row indices per fold
    latent_scores  = None                     # allocate only if needed
    if target_latent.lower() != "raw":
        latent_scores = np.zeros((n_subj, n_components))

    for res in results_list:
        tag, rows = res["tag"], res["te_idx"]

        # subject-ordered containers
        store[tag]["true"][rows] = res["Y_true"]
        store[tag]["pred"][rows] = res["Y_pred"]
        store[tag]["cont"][rows] = res["y_true_cont"]

        # extras for sign-alignment – only once per fold (mode_i==0 ≡ TP0)
        if target_latent.lower() != "raw" and res["mode_i"] == 0:
            latent_scores[rows, : res["Y_true"].shape[1]] = res["Y_true"]
            fold_meta.append({"load": res["load"], "rows": rows})

    # 10) sign-align loadings  *and* every per-subject value  ───────────
    if target_latent.lower() != "raw" and fold_meta:
        loads = np.stack([m["load"] for m in fold_meta])   # shape depends on method
        ref   = loads[0].copy()

        if target_latent.lower() in ("pca", "plsc"):       # loads: (F, k, q)
            k = loads.shape[1]
            flips = np.ones((loads.shape[0], k), dtype=int)

            for f in range(1, loads.shape[0]):
                for k_i in range(k):
                    if np.dot(ref[k_i], loads[f, k_i]) < 0:
                        flips[f, k_i]  = -1
                        loads[f, k_i] *= -1
            avg_load = loads.mean(axis=0)                  # (k, q)

        else:                                             # LDA  loads: (F, q, k)
            k = loads.shape[2]
            flips = np.ones((loads.shape[0], k), dtype=int)

            for f in range(1, loads.shape[0]):
                for k_i in range(k):
                    if np.dot(ref[:, k_i], loads[f, :, k_i]) < 0:
                        flips[f, k_i]  = -1
                        loads[f, :, k_i] *= -1
            avg_load = loads.mean(axis=0)                  # (q, k)

        # propagate flips to latent scores and all store-arrays
        for f, meta in enumerate(fold_meta):
            rows = meta["rows"]
            for k_i in range(k):
                if flips[f, k_i] == -1:
                    latent_scores[rows, k_i] *= -1

                    # always flip the continuous copy (needed for plots & scatter)
                    store["t0"]["cont"][rows, k_i] *= -1
                    store["t1"]["cont"][rows, k_i] *= -1
                    store["delta"]["cont"][rows, k_i] *= -1

                    # flip the regression targets **only** when we are in regression mode
                    if quantile is None:
                        for tag in ("t0", "t1", "delta"):
                            store[tag]["true"][rows, k_i] *= -1
                            store[tag]["pred"][rows, k_i] *= -1


    def spread(y):
        if np.nanstd(y, ddof=1) < 1e-10:
            return 0.0
        z = (y - np.nanmean(y)) / np.nanstd(y, ddof=1)
        return iqr(z)

    group_labels = subj_df[group_col].to_numpy()
    cmap         = plt.get_cmap("tab10")


    # ---------------------------------------------------------------
    # 11) ordered plots  (one latent/target at a time)
    # ---------------------------------------------------------------
    plot_names = results_list[0]["names"]
    tags_order = ("t0", "t1", "delta")
  
    for j, nm in enumerate(plot_names):

        # ── A) bar plot of loadings (latent only) ───────────────────
        if target_latent.lower() != "raw":
            bar_vals = (avg_load[j] if target_latent.lower() in ("pca", "plsc") else avg_load[:, j])
            plt.figure(dpi=dpi)
            plt.barh(target_cols, bar_vals, color="steelblue")
            plt.axvline(0, color="grey", lw=1)
            plt.title(f"{nm}  –  mean loading"); plt.tight_layout(); plt.show()

        for tag in tags_order:
            y_true_cls  = store[tag]["true"]
            y_pred_cls  = store[tag]["pred"]
            y_true_cont = store[tag]["cont"] 

            # ❷ choose the array used for the subject-ordered scatter
            if quantile is None:
                y_true_plot = y_true_cls[:, j]   # they’re already continuous
                y_pred_plot = y_pred_cls[:, j]
            else:
                y_true_plot = y_true_cont[:, j]  # continuous spread
                y_pred_plot = y_pred_cls[:, j]   # NaNs for mid-quantile subjects

            # ❸ keep the *discrete* version under the old names so the confusion-matrix
            #    code below still works unchanged  ↓↓↓
            y_true = y_true_cls[:, j]    # ← add these two lines
            y_pred = y_pred_cls[:, j]    # ←


            # subject-ordered scatter (continuous values give it vertical spread)
            order = np.argsort(group_labels)
            x     = np.arange(len(order))

            plt.figure(figsize=(10, 4), dpi=dpi)
            sc = plt.scatter(x, y_true_plot[order],
                            c=group_labels[order], cmap=cmap, s=30)
            handles, labels = sc.legend_elements(prop="colors", alpha=1.0)
            if quantile is not None:
                # recompute the cutoffs on the continuous true values for this target
                # y_true_cont is shape (n_subjects, n_targets)
                lower_q, upper_q = np.quantile(y_true_cont[:, j], [quantile, 1 - quantile])
                # draw dashed red lines at those values
                plt.axhline(lower_q, color='red', linestyle='--', linewidth=1)
                plt.axhline(upper_q, color='red', linestyle='--', linewidth=1)
            plt.legend(handles, labels, title="Group", loc="best")

            plt.title(f"{nm} – {tag.upper()}  | spread={spread(y_true_plot):.2g}")
            plt.xlabel("subject");  plt.ylabel(nm)
            plt.tight_layout(); plt.show()

            # -----------------------------------------------------------------
            #  regression scatter  OR  classification diagnostics
            # -----------------------------------------------------------------
            if quantile is None:
                r  = pearsonr(y_true_plot, y_pred_plot)[0]
                r2 = r2_score(y_true_plot, y_pred_plot)

                plt.figure(dpi=dpi)
                plt.scatter(y_true_plot, y_pred_plot, alpha=0.6)
                plt.grid(True)
                plt.title(f"{nm} – {tag.upper()}  | r={r:.2f}, R²={r2:.2f}")
                plt.xlabel("True"); plt.ylabel("Predicted")
                plt.tight_layout(); plt.show()
            else:
                # ── 1) histogram of continuous phenotype with coloured tails ──────────
                if tag == "t0":                      # draw it only once per target
                    phen = store["t0"]["cont"][:, j]
                    lower_q, upper_q = np.quantile(phen, [quantile, 1.0 - quantile])
                    bins = np.linspace(phen.min(), phen.max(), 30)
                    plt.figure(figsize=(10, 6), dpi=dpi)
                    plt.hist(phen, bins=bins, alpha=0.5, label=f"All", color="green")
                    plt.hist(phen[phen >= upper_q], bins=bins, alpha=0.5, label=f"Top {quantile:.0%}", color="blue")
                    plt.hist(phen[phen <= lower_q], bins=bins, alpha=0.5, label=f"Bottom {quantile:.0%}", color="red")
                    plt.xlabel(nm)
                    plt.ylabel("Frequency")
                    plt.title(f"Histogram of {nm} (deconf. {deconf}) with extreme quantile classes")
                    plt.legend()
                    plt.tight_layout()
                    plt.show()

                # ── 2) confusion-matrix heat map ────────────────────────────────────
                # predictions are NaN for the mid-quantile subjects → drop them
                valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
                y_true_cm  = y_true[valid_mask].astype(int)
                y_pred_cm  = y_pred[valid_mask].astype(int)

                cm     = confusion_matrix(y_true_cm, y_pred_cm, labels=np.unique(y_true_cm))
                cm_pct = cm / cm.sum(axis=1, keepdims=True)

                fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
                im = ax.imshow(cm_pct, cmap=plt.cm.Blues)

                acc = accuracy_score(y_true_cm, y_pred_cm)
                ax.set(title=f"{nm} - {tag.upper()} | Acc {acc:.2%}",
                    xlabel="Predicted label", ylabel="True label",
                    xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
                ax.set_xticklabels(np.unique(y_true_cm), rotation=45)
                ax.set_yticklabels(np.unique(y_true_cm))

                thresh = 0.5 * np.nanmax(cm_pct)
                for row in range(cm.shape[0]):
                    for col in range(cm.shape[1]):
                        val, pct = cm[row, col], cm_pct[row, col]
                        ax.text(col, row, f"{val}\n({pct:.1%})",
                                ha="center", va="center",
                                color="white" if pct > thresh else "black")


                fig.colorbar(im, ax=ax, label="Fraction of true class")
                fig.tight_layout(); plt.show()
