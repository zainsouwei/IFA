from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from typing import List, Dict, Optional, Tuple
from itertools import combinations
import pandas as pd
from matplotlib.colors import ListedColormap
from nilearn import plotting
import hcp_utils as hcp
import os
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from PIL import Image
import cairosvg
import io
import json

import math

def _json_safe(obj):
    """Recursively convert scientific-Python objects to JSON-safe types."""
    # numpy scalars
    if isinstance(obj, np.generic):
        return obj.item()
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # pathlib Path
    if isinstance(obj, Path):
        return str(obj)
    # pandas
    if isinstance(obj, pd.DataFrame):
        return {c: _json_safe(obj[c].values) for c in obj.columns}
    if isinstance(obj, pd.Series):
        return _json_safe(obj.values)
    # containers
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    # floats that might be nan/inf
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj

def plot_reconstruction_kde(
    condition_paths: List[str],
    condition_labels: List[str],
    pipelines: List[str] = ("GICA", "parcel_IFA", "voxel_IFA"),
    pipeline_labels: List[str] = ("GICA", "Parcellated IFA", "Grayordinate IFA"),
    palette: Dict[str, str] = None,
    nPCA: int = 8,
    folds: List[int] = (0, 1, 2, 3, 4),
    figsize: Tuple[int, int] = (16, 12),
    sharey: bool = True,
    save_path: Optional[str] = None,
):
    if palette is None:
        palette = {
            "GICA":             "#1b9e77",  # Teal
            "Parcellated IFA":  "#d95f02",  # Burnt orange
            "Grayordinate IFA": "#7570b3",  # Purple
        }

    assert len(condition_paths) == len(condition_labels), "Paths and labels must align."
    assert len(pipelines) == len(pipeline_labels), "pipelines and pipeline_labels must align."

    # Decide grid if not provided
    n_cond = len(condition_paths)
    # simple square-ish layout
    ncols = int(np.ceil(np.sqrt(n_cond)))
    nrows = int(np.ceil(n_cond / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()

    # To return the raw pooled values if you want to reuse them
    pooled_errors = {}

    for ax, cond_label, cond_root in zip(axes, condition_labels, condition_paths):
        pooled_errors[cond_label] = {lbl: [] for lbl in pipeline_labels}

        # Collect test-set reconstruction errors across folds per pipeline
        for pipe, pipe_lbl in zip(pipelines, pipeline_labels):
            for fold in folds:
                pkl = Path(cond_root) / f"fold_{fold}" / f"nPCA_{nPCA}" / "Results" / pipe / "results.pkl"
                if not pkl.exists():
                    continue
                data = pickle.loads(pkl.read_bytes())
                recon = data.get("recon")[1]  # test recon at index 1
                # Flatten scalar or array
                if np.ndim(recon) == 0:
                    pooled_errors[cond_label][pipe_lbl].append(float(recon))
                else:
                    pooled_errors[cond_label][pipe_lbl].extend(np.ravel(recon))

        # Plot KDE per pipeline
        for pipe_lbl, vals in pooled_errors[cond_label].items():
            arr = np.asarray(vals, dtype=float)
            kde = gaussian_kde(arr)
            x_grid = np.linspace(arr.min(), arr.max(), 500)
            y_grid = kde(x_grid)

            ax.plot(x_grid, y_grid, lw=2.5, label=pipe_lbl, color=palette.get(pipe_lbl, None))
            ax.fill_between(x_grid, y_grid, alpha=0.25, color=palette.get(pipe_lbl, None))

        # Styling
        ax.set_title(cond_label, fontsize=16, fontweight='bold')
        ax.set_xlabel("% Variance Explained", fontsize=14)
        if ax is axes[0]:
            ax.set_ylabel("Density", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=12, frameon=False)

    # Hide any extra axes if grid > number of conditions
    for ax in axes[len(condition_paths):]:
        ax.axis("off")

    plt.tight_layout(pad=4.0)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig, axes, pooled_errors


def _extract_accuracy(class_result: dict):
    """
    class_result: dict like {"Logistic Regression": {"accuracy": ...}, "svc": {...}, ...}
    priority: ordered classifier names to try
    fallback: "best" -> pick the key with highest accuracy if priorities missing
              "none" -> return None if priorities missing
    """
    best = None
    best_acc = -np.inf
    for k, v in class_result.items():
        acc = v.get("accuracy")
        if acc is not None and np.isfinite(acc) and acc > best_acc:
            best_acc = acc
            best = acc
    return float(best)


def plot_accuracy_vs_model_order_robust(
    condition_paths: List[str],
    condition_labels: List[str],
    nPCA_all: List[int],
    pipelines: List[str] = ("GICA", "parcel_IFA", "voxel_IFA"),
    pipeline_labels: List[str] = ("GICA", "Parcellated IFA", "Grayordinate IFA"),
    folds: List[int] = (0, 1, 2, 3, 4),
    figsize: Tuple[int, int] = (12, 5),
    sharey: bool = True,
    save_path: Optional[str] = None,
):
    """
    Aggregate classifier accuracy across folds per nPCA and plot mean ± std for each pipeline.
    Robust to missing classifier keys (e.g., no 'Logistic Regression').
    """
    palette = {
        "GICA":             "#264653",  # deep slate
        "Parcellated IFA":  "#2a9d8f",  # teal
        "Grayordinate IFA": "#e9c46a",  # mustard
    }

    assert len(condition_paths) == len(condition_labels), "Paths and labels must align."
    assert len(pipelines) == len(pipeline_labels), "pipelines and pipeline_labels must align."

    x_ticks = list(nPCA_all)

    fig, axes = plt.subplots(1, len(condition_labels), figsize=figsize, sharey=sharey)
    axes = np.atleast_1d(axes)

    stats = {}

    for ax, cond_label, cond_path in zip(axes, condition_labels, condition_paths):
        stats[cond_label] = {label: {"mean": [], "std": []} for label in pipeline_labels}

        for nPCA in nPCA_all:
            for folder, plabel in zip(pipelines, pipeline_labels):
                accs = []
                for fold in folds:
                    pkl = Path(cond_path) / f"fold_{fold}" / f"nPCA_{nPCA}" / "Results" / folder / "results.pkl"
                    if not pkl.exists():
                        continue
                    data = pickle.loads(pkl.read_bytes())

                    class_res = data.get("Class_Result", {})
                    if not isinstance(class_res, dict):
                        continue

                    accs.append(_extract_accuracy(class_res))

                stats[cond_label][plabel]["mean"].append(float(np.mean(accs)) if accs else np.nan)
                stats[cond_label][plabel]["std"].append(float(np.std(accs)) if accs else np.nan)

        for plabel in pipeline_labels:
            ax.errorbar(
                x_ticks, stats[cond_label][plabel]["mean"], yerr=stats[cond_label][plabel]["std"],
                marker='o', linestyle='-', capsize=4, color=palette[plabel], label=plabel
            )

        ax.set_title(cond_label)
        ax.set_xlabel("Model order (nPCA)")
        ax.set_xticks(x_ticks)

    axes[0].set_ylabel("Accuracy")
    for ax in axes:
        ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig, axes, stats


def summarize_spatial_tests(
    condition_paths: List[str],
    condition_labels: List[str],
    pipelines: List[str] = ("GICA", "parcel_IFA", "voxel_IFA"),
    nPCA: int = 8,
    folds: List[int] = (0,1,2,3,4),
    alpha: float = 0.05,
    compute_min_p_per_vertex: bool = False,
) -> Tuple[Dict, Optional[Dict]]:
    """
    Prints per-condition/pipeline summary lines:
      Standard:     #Sig = mean ± std,  -log(p) = mean ± std
      Discriminant: #Sig = mean ± std,  -log(p) = mean ± std

    Returns:
      summary[(cond_label, pipe)] -> {
        'std':     {'n_sig': list, 'logp_sum': list},
        'discrim': {'n_sig': list, 'logp_sum': list},
        'folds_used': int
      }
      min_p[(cond_label, pipe)] -> {
        'std_min_p': np.ndarray [n_vertices],
        'discrim_min_p': np.ndarray [n_vertices]
      } or None
    """
    assert len(condition_paths) == len(condition_labels), "Paths and labels must align."


    summary: Dict[Tuple[str, str], Dict] = {}
    min_p: Optional[Dict[Tuple[str, str], Dict[str, np.ndarray]]] = {} if compute_min_p_per_vertex else None

    for cond_label, cond_path in zip(condition_labels, condition_paths):
        print(f"\n==== {cond_label} ====")
        for p in pipelines:

            sig, psum, d_sig, d_psum = [], [], [], []
            std_min_stack, dis_min_stack = [], []

            for fold in folds:
                pkl = Path(cond_path) / f"fold_{fold}" / f"nPCA_{nPCA}" / "Results" / p / "results.pkl"
                if not pkl.exists():
                    continue
                data = pickle.loads(pkl.read_bytes())

                # Standard spatial t-test
                st = data.get("Spatial_t_test")
                if isinstance(st, (list, tuple)) and len(st) >= 3:
                    p_std = np.asarray(st[1])
                    logp_std = np.asarray(st[2])
                    sig.append(int(np.sum(p_std < alpha)))
                    psum.append(float(np.sum(logp_std)))
                    if compute_min_p_per_vertex:
                        std_min_stack.append(p_std)

                # Discriminant spatial t-test
                sd = data.get("Spatial_t_test_discrim")
                if isinstance(sd, (list, tuple)) and len(sd) >= 3:
                    p_dis = np.asarray(sd[1])
                    logp_dis = np.asarray(sd[2])
                    d_sig.append(int(np.sum(p_dis < alpha)))
                    d_psum.append(float(np.sum(logp_dis)))
                    if compute_min_p_per_vertex:
                        dis_min_stack.append(p_dis)

            # Store raw lists
            summary[(cond_label, p)] = {
                "std":     {"n_sig": sig,  "logp_sum": psum},
                "discrim": {"n_sig": d_sig, "logp_sum": d_psum},
                "folds_used": max(len(sig), len(d_sig), len(psum), len(d_psum)),
            }

            # Pretty print in your exact format
            def m_s(x): 
                return (np.mean(x), np.std(x)) if len(x) else (np.nan, np.nan)
            m_sig, s_sig   = m_s(sig)
            m_ps,  s_ps    = m_s(psum)
            m_dsig, s_dsig = m_s(d_sig)
            m_dps,  s_dps  = m_s(d_psum)

            print(f"\n--- {p} ---")
            print(f"Standard:     #Sig = {m_sig:.1f} ± {s_sig:.1f},  -log(p) = {m_ps:.2f} ± {s_ps:.2f}")
            print(f"Discriminant: #Sig = {m_dsig:.1f} ± {s_dsig:.1f},  -log(p) = {m_dps:.2f} ± {s_dps:.2f}")

            # Optional per-vertex min p across folds
            if compute_min_p_per_vertex:
                out = {}
                if std_min_stack:
                    out["std_min_p"] = np.min(np.stack(std_min_stack, axis=0), axis=0)
                if dis_min_stack:
                    out["discrim_min_p"] = np.min(np.stack(dis_min_stack, axis=0), axis=0)
                min_p[(cond_label, p)] = out

    return summary, (min_p if compute_min_p_per_vertex else None)



# # once, near the top of your script
# mmp_labels = hcp.mmp["labels"]
# pd.DataFrame({"id": list(mmp_labels.keys()), "name": list(mmp_labels.values())}) \
#   .sort_values("id").to_csv(Path(out_dir) / "mmp_labels_reference.csv", index=False)

def custom_mode(vertices):
    values, counts = np.unique(vertices, return_counts=True)
    if len(values) == 3:
        if 3 in values and 1 in values and 0 in values:
            return 1  # A only
        elif 3 in values and 2 in values and 0 in values:
            return 2  # B only
        elif 3 in values and 1 in values and 2 in values:
            return 3  # Shared
        else:
            return 0  # Neither
    return values[np.argmax(counts)]


def load_results_pvals(cond_path, pipe, fold, nPCA):
    pkl = Path(cond_path) / f"fold_{fold}" / f"nPCA_{nPCA}" / "Results" / pipe / "results.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"Missing results: {pkl}")
    data = pickle.loads(pkl.read_bytes())
    # arrays shape: (n_maps, n_vertices)
    reg_p = np.asarray(data["Spatial_t_test"][1])
    dis_p = np.asarray(data["Spatial_t_test_discrim"][1])
    # collapse maps with min across maps (vertexwise)
    reg_min = np.min(reg_p, axis=0)
    dis_min = np.min(dis_p, axis=0)
    return reg_min, dis_min


def all_across_folds(pipelines,folds,nPCA,cond_path,min=False):
    # returns dict: pipe -> {"regular": min_p_vec, "discrim": min_p_vec} across folds
    combs = {}
    for pipe in pipelines:
        reg_stack, dis_stack = [], []
        for f in folds:
            r, d = load_results_pvals(cond_path, pipe, f, nPCA)
            reg_stack.append(r)
            dis_stack.append(d)
        if min:
            combs[pipe] = {
                "regular": np.min(np.stack(reg_stack, 0), axis=0),
                "discrim": np.min(np.stack(dis_stack, 0), axis=0)
            }
        else:
            combs[pipe] = {
                "regular": np.max(np.stack(reg_stack, 0), axis=0),
                "discrim": np.max(np.stack(dis_stack, 0), axis=0)
            }
    return combs


def sig_composite(sig_a, sig_b):
    comp = np.zeros_like(sig_a, dtype=int)
    comp[sig_a & ~sig_b] = 1  # A only
    comp[~sig_a & sig_b] = 2  # B only
    comp[sig_a & sig_b]  = 3  # shared
    return comp


# --- ADD THIS HELPER (anywhere above compare_pipelines_spatial) ---
def fold_aggregate_composite(A_sig_folds: np.ndarray,
                             B_sig_folds: np.ndarray,
                             ambiguous_policy: str = "neither") -> np.ndarray:
    """
    Aggregate significance across folds into a single per-vertex composite:
      0 = neither, 1 = A-only, 2 = B-only, 3 = both

    Rules per vertex across folds:
      - neither (0) if it's EVER neither across folds
      - both    (3) if it's EVER both AND NEVER neither
      - A-only  (1) if it's ALWAYS A-only (and never neither/both)
      - B-only  (2) if it's ALWAYS B-only (and never neither/both)
      - Mixed A-only/B-only with no both/neither -> ambiguous_policy
        ("neither" | "both" | "majority"), default "neither".
    """
    # shapes: (n_folds, n_vertices) booleans
    neither_any = np.any((~A_sig_folds) & (~B_sig_folds), axis=0)
    both_any    = np.any(A_sig_folds & B_sig_folds, axis=0)
    a_only_all  = np.all(A_sig_folds & ~B_sig_folds, axis=0)
    b_only_all  = np.all(~A_sig_folds & B_sig_folds, axis=0)

    comp = np.zeros(A_sig_folds.shape[1], dtype=int)  # start as 0 (neither)

    # both if ever both and never neither
    idx_both = (~neither_any) & both_any
    comp[idx_both] = 3

    # A-only if always A-only
    idx_aonly = (~neither_any) & (~idx_both) & a_only_all
    comp[idx_aonly] = 1

    # B-only if always B-only
    idx_bonly = (~neither_any) & (~idx_both) & (~idx_aonly) & b_only_all
    comp[idx_bonly] = 2

    # ambiguous: mixed A-only/B-only with no both/neither
    amb = (~neither_any) & (~idx_both) & (~a_only_all) & (~b_only_all)
    if np.any(amb):
        if ambiguous_policy == "both":
            comp[amb] = 3
        elif ambiguous_policy == "majority":
            Aonly_counts = np.sum(A_sig_folds[:, amb] & ~B_sig_folds[:, amb], axis=0)
            Bonly_counts = np.sum(~A_sig_folds[:, amb] & B_sig_folds[:, amb], axis=0)
            to_A = Aonly_counts >= Bonly_counts
            amb_idx = np.where(amb)[0]
            comp[amb_idx[to_A]] = 1
            comp[amb_idx[~to_A]] = 2
        # else "neither": leave as 0
    return comp

# add these helpers (above compare_pipelines_spatial)

def make_legend_handles(label_a: str, label_b: str):
    return [
        mpatches.Patch(color="#cccccc", label="Neither"),
        mpatches.Patch(color="#f4a582", label=f"{label_a} only"),
        mpatches.Patch(color="#92c5de", label=f"{label_b} only"),
        mpatches.Patch(color="#8073ac", label="Shared"),
    ]


def append_parcel_rows_from_comp(csv_rows: list,
                                 comp: np.ndarray,
                                 parcel_vec: np.ndarray,
                                 labels: dict,
                                 mode: str,
                                 condition: str,
                                 comparison: str,
                                 kind: str,
                                 fold):
    """
    Use the composite vector (0=neither, 1=A-only, 2=B-only, 3=shared) to
    append parcel-level ABR percentages into csv_rows.
    """
    for pid in np.unique(parcel_vec):
        if pid == 0:
            continue
        mask = (parcel_vec == pid)
        n = mask.sum()
        if n == 0:
            continue

        pct_A = 100.0 * np.mean(comp[mask] == 1)
        pct_B = 100.0 * np.mean(comp[mask] == 2)
        pct_S = 100.0 * np.mean(comp[mask] == 3)

        if (pct_A + pct_B + pct_S) > 0:
            csv_rows.append({
                "mode": str(mode),
                "condition": str(condition),
                "comparison": str(comparison),
                "type": str(kind),
                "fold": fold,
                "parcel_id": int(pid),
                "label": str(labels.get(pid, "")),
                "pct_A_only": round(pct_A, 2),
                "pct_B_only": round(pct_B, 2),
                "pct_shared": round(pct_S, 2),
            })


def build_summary_with_metrics(df: pd.DataFrame, epsilon_full: float = 0.01) -> pd.DataFrame:
    """
    Aggregate across folds (mean/std) and add the derived metrics you used before.
    Returns a DataFrame with columns:
      mean_pct_*, std_pct_*, Total_Discriminant_Signal_Pct,
      Method_*_Unique_Contribution_Ratio, Method_*_Total_Attribution_Ratio,
      Fully_Method_A / Fully_Method_B (boolean flags).
    """
    group_cols = ["mode", "condition", "comparison", "type", "parcel_id", "label"]

    # mean/std across folds
    summary = (
        df
        .groupby(group_cols, as_index=False)[["pct_A_only","pct_B_only","pct_shared"]]
        .agg(['mean','std'])
    )
    # flatten columns
    summary.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0]
        for c in summary.columns.to_flat_index()
    ]
    # rename means/stds
    summary = summary.rename(columns={
        "pct_A_only_mean": "mean_pct_A_only",
        "pct_B_only_mean": "mean_pct_B_only",
        "pct_shared_mean": "mean_pct_shared",
        "pct_A_only_std":  "std_pct_A_only",
        "pct_B_only_std":  "std_pct_B_only",
        "pct_shared_std":  "std_pct_shared",
    })

    # derived metrics
    total = (
        summary["mean_pct_A_only"]
        + summary["mean_pct_B_only"]
        + summary["mean_pct_shared"]
    ).replace(0, np.nan)
    summary["Total_Discriminant_Signal_Pct"] = total

    summary["Method_B_Unique_Contribution_Ratio"] = summary["mean_pct_B_only"] / total
    summary["Method_A_Unique_Contribution_Ratio"] = summary["mean_pct_A_only"] / total

    summary["Method_A_Total_Attribution_Ratio"] = 1 - summary["Method_B_Unique_Contribution_Ratio"]
    summary["Method_B_Total_Attribution_Ratio"] = 1 - summary["Method_A_Unique_Contribution_Ratio"]

    summary["Fully_Method_A"] = summary["Method_A_Unique_Contribution_Ratio"] >= (1 - epsilon_full)
    summary["Fully_Method_B"] = summary["Method_B_Unique_Contribution_Ratio"] >= (1 - epsilon_full)

    return summary


def compare_pipelines_spatial(
    condition_paths,
    condition_labels,
    pipelines=("GICA","parcel_IFA","voxel_IFA"),
    pipeline_labels=None,              # pretty names, same order/len as pipelines
    nPCA=8,
    folds=(0,1,2,3,4),
    alpha=0.05,
    compare_mode="global",             # "global" or "foldwise"
    aggregation="aggregate_before_compare",  # only used if compare_mode="global":
                                             #   "aggregate_before_compare"  or  "aggregate_after_compare"
    min_join=False,
    views=("lateral","ventral","posterior","lateral"),
    out_dir="ifa_gica_outputs",
    abr_min_total_pct=1.0,            # threshold for showing bars (sum A/B/shared >= this %)
    dpi=300):
    """
    One-stop function:
      - Pairwise brain overlays of significant vertices (A only / B only / shared)
      - Parcel-level ABR bars (A-only / B-only / shared %)
      - CSV with parcel percentages (optionally per fold or aggregated)
    
    compare_mode:
      - "foldwise": do comparisons separately for each fold (overlays/rows per fold)
      - "global": combine folds first according to 'aggregation':
           * "aggregate_before_compare": min p across folds within each pipeline, then compare
           * "aggregate_after_compare": compare per fold, then average parcel % across folds (ABR & CSV)
    
    Raises FileNotFoundError if any required results.pkl is missing.
    """

    if pipeline_labels is None:
        default_map = {"GICA":"GICA", "parcel_IFA":"IFA (Parcel)", "voxel_IFA":"IFA (Voxel)"}
        pipeline_labels = [default_map.get(p, p) for p in pipelines]
    pretty = dict(zip(pipelines, pipeline_labels))

    cmap = ListedColormap(["#cccccc","#f4a582","#92c5de","#8073ac"])  # neither, A, B, shared
    parcel_vec = hcp.mmp["map_all"]
    labels = hcp.mmp["labels"]

    overlays_dir = Path(out_dir) / "overlays"
    bars_dir     = Path(out_dir) / "abr_plots"
    Path(overlays_dir).mkdir(parents=True, exist_ok=True)
    Path(bars_dir).mkdir(parents=True, exist_ok=True)

    csv_rows = []

    # -------- MAIN LOOP OVER CONDITIONS --------
    for cond_label, cond_path in zip(condition_labels, condition_paths):

        if compare_mode == "global" and aggregation == "aggregate_before_compare":
            # Aggregate within pipeline across folds first (min p), then compare.
            combs = all_across_folds(pipelines,folds,nPCA,cond_path,min=min_join)

            # overlays + parcel % for all pairs & kinds
            for (pa, pb) in combinations(pipelines, 2):
                for kind in ("regular","discrim"):
                    A = combs[pa][kind] < alpha
                    B = combs[pb][kind] < alpha

                    # --- Brain overlay (once per condition/pair/kind)
                    comp = sig_composite(A,B)
                    fig = plt.figure(figsize=(4*len(views), 5))
                    for i, view in enumerate(views):
                        hemi = "right" if (view == "lateral" and i == len(views)-1) else "left"
                        ax = fig.add_subplot(1, len(views), i+1, projection="3d")
                        plotting.plot_surf(
                            hcp.mesh.inflated,
                            hcp.cortex_data(comp),
                            bg_map=hcp.mesh.sulc,
                            hemi=hemi, view=view,
                            cmap=cmap, colorbar=False, axes=ax,
                            threshold=0.01, symmetric_cmap=False,
                            avg_method=custom_mode
                        )
                        ax.set_title(view, fontsize=10)
                    title = f"{cond_label}: {pretty[pa]} vs {pretty[pb]} [{kind}]"
                    fig.suptitle(title, fontsize=12)
                    plt.tight_layout()
                    join_tag = "MIN" if min_join else "MAX"
                    fname = f"{cond_label}_{pretty[pa]}_vs_{pretty[pb]}_{kind}_GLOBAL_{join_tag}.png".replace(" ","_").replace("(","").replace(")","")
                    fig.legend(
                        handles=make_legend_handles(pretty[pa], pretty[pb]),
                        loc="lower center", ncol=4, frameon=False
                    )
                    fig.savefig(overlays_dir/fname, dpi=dpi, bbox_inches="tight")
                    plt.close(fig)

                    # CSV rows
                    append_parcel_rows_from_comp(
                        csv_rows=csv_rows,
                        comp=comp,
                        parcel_vec=parcel_vec,
                        labels=labels,
                        mode="global_before",
                        condition=cond_label,
                        comparison=f"{pretty[pa]} vs {pretty[pb]}",
                        kind=kind,
                        fold="global_min" if min_join else "global_max",
                    )

        elif compare_mode == "global" and aggregation == "aggregate_after_compare":
            # Compare per fold (A vs B), then aggregate categories across folds
            for (pa, pb) in combinations(pipelines, 2):
                for kind in ("regular", "discrim"):
                    A_sig_folds, B_sig_folds = [], []

                    # 1) build per-fold significance (bool) for each pipeline
                    for f in folds:
                        rA, dA = load_results_pvals(cond_path, pa, f, nPCA)
                        rB, dB = load_results_pvals(cond_path, pb, f, nPCA)
                        vecA = rA if kind == "regular" else dA
                        vecB = rB if kind == "regular" else dB
                        A_sig_folds.append(vecA < alpha)
                        B_sig_folds.append(vecB < alpha)

                    A_sig_folds = np.stack(A_sig_folds, axis=0).astype(bool)  # (n_folds, n_vertices)
                    B_sig_folds = np.stack(B_sig_folds, axis=0).astype(bool)

                    # 2) aggregate the *comparison* across folds into a single composite {0,1,2,3}
                    comp = fold_aggregate_composite(
                        A_sig_folds, B_sig_folds, ambiguous_policy="neither"
                    )

                    # --- Brain overlay (single figure per condition/pair/kind)
                    fig = plt.figure(figsize=(4*len(views), 5))
                    for i, view in enumerate(views):
                        hemi = "right" if (view == "lateral" and i == len(views)-1) else "left"
                        ax = fig.add_subplot(1, len(views), i+1, projection="3d")
                        plotting.plot_surf(
                            hcp.mesh.inflated,
                            hcp.cortex_data(comp),
                            bg_map=hcp.mesh.sulc,
                            hemi=hemi, view=view,
                            cmap=cmap,
                            colorbar=False, axes=ax,
                            threshold=0.01, symmetric_cmap=False,
                            avg_method=custom_mode
                        )
                        ax.set_title(view, fontsize=10)

                    title = f"{cond_label}: {pretty[pa]} vs {pretty[pb]} [{kind}] | GLOBAL_COMPARE_THEN_AGG"
                    fig.suptitle(title, fontsize=12)
                    plt.tight_layout()
                    fname = (
                        f"{cond_label}_{pretty[pa]}_vs_{pretty[pb]}_{kind}_GLOBAL_COMPARE_THEN_AGG.png"
                        .replace(" ", "_").replace("(", "").replace(")", "")
                    )
                    fig.legend(
                        handles=make_legend_handles(pretty[pa], pretty[pb]),
                        loc="lower center", ncol=4, frameon=False
                    )                    
                    fig.savefig(overlays_dir / fname, dpi=dpi, bbox_inches="tight")
                    plt.close(fig)


                    append_parcel_rows_from_comp(
                        csv_rows=csv_rows,
                        comp=comp,
                        parcel_vec=parcel_vec,
                        labels=labels,
                        mode="global_compare_then_aggregate",
                        condition=cond_label,
                        comparison=f"{pretty[pa]} vs {pretty[pb]}",
                        kind=kind,
                        fold="compare_then_agg",
                    )

        elif compare_mode == "foldwise":
            # pure per-fold comparisons (CSV + overlays per fold)
            for (pa, pb) in combinations(pipelines, 2):
                for kind in ("regular","discrim"):
                    for f in folds:
                        rA, dA = load_results_pvals(cond_path, pa, f, nPCA)
                        rB, dB = load_results_pvals(cond_path, pb, f, nPCA)
                        vecA = rA if kind=="regular" else dA
                        vecB = rB if kind=="regular" else dB
                        A = vecA < alpha
                        B = vecB < alpha

                        # Brain overlay per fold
                        comp = sig_composite(A,B)
                        fig = plt.figure(figsize=(4*len(views), 5))
                        for i, view in enumerate(views):
                            hemi = "right" if (view == "lateral" and i == len(views)-1) else "left"
                            ax = fig.add_subplot(1, len(views), i+1, projection="3d")
                            plotting.plot_surf(
                                hcp.mesh.inflated,
                                hcp.cortex_data(comp),
                                bg_map=hcp.mesh.sulc,
                                hemi=hemi, view=view,
                                cmap=cmap, colorbar=False, axes=ax,
                                threshold=0.01, symmetric_cmap=False,
                                avg_method=custom_mode
                            )
                            ax.set_title(view, fontsize=10)
                        title = f"{cond_label}: {pretty[pa]} vs {pretty[pb]} [{kind}] | fold {f}"
                        fig.suptitle(title, fontsize=12)
                        plt.tight_layout()
                        fname = f"{cond_label}_{pretty[pa]}_vs_{pretty[pb]}_{kind}_F{f}.png".replace(" ","_").replace("(","").replace(")","")
                        fig.legend(
                            handles=make_legend_handles(pretty[pa], pretty[pb]),
                            loc="lower center", ncol=4, frameon=False
                        )
                        fig.savefig(overlays_dir/fname, dpi=dpi, bbox_inches="tight")
                        plt.close(fig)

                        append_parcel_rows_from_comp(
                            csv_rows=csv_rows,
                            comp=comp,
                            parcel_vec=parcel_vec,
                            labels=labels,
                            mode="foldwise",
                            condition=cond_label,
                            comparison=f"{pretty[pa]} vs {pretty[pb]}",
                            kind=kind,
                            fold=int(f),
                        )
        else:
            raise ValueError("compare_mode must be 'global' or 'foldwise'")

    # -------- Write CSV --------
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(csv_rows)
    csv_path = Path(out_dir) / "ifa_gica_overlap_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ wrote {csv_path}")

    # (optional) also write an aggregated-by-parcel summary with extra metrics
    summary_df = build_summary_with_metrics(df, epsilon_full=0.01)
    summary_csv = Path(out_dir) / "ifa_gica_overlap_summary_by_parcel.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ wrote {summary_csv}")

    # -------- Make ABR stacked bar plots --------
    if df.empty:
        print("No rows to plot.")
        return df

    if compare_mode == "foldwise":
        # one bar plot *per fold*
        key_cols = ["condition", "comparison", "type", "fold"]
        keys = df[key_cols].drop_duplicates()

        for _, key in keys.iterrows():
            sub = df[
                (df["condition"]  == key["condition"]) &
                (df["comparison"] == key["comparison"]) &
                (df["type"]       == key["type"]) &
                (df["fold"]       == key["fold"])
            ].copy()

            sub["total"] = sub["pct_A_only"] + sub["pct_B_only"] + sub["pct_shared"]
            # sub = sub[sub["total"] >= abr_min_total_pct].sort_values("pct_B_only", ascending=False)
            # sort by Method_B_Unique_Contribution_Ratio = pct_B_only / total
            sub = sub[sub["total"] >= abr_min_total_pct].copy()
            sub["Method_B_Unique_Contribution_Ratio"] = (sub["pct_B_only"] / sub["total"]).replace([np.inf, -np.inf], np.nan).fillna(0)
            sub = sub.sort_values("Method_B_Unique_Contribution_Ratio", ascending=False)
            if sub.empty:
                continue

            plt.figure(figsize=(16, 6))
            plt.bar(sub["label"], sub["pct_shared"],                   label="Shared",  color="#8073ac")
            plt.bar(sub["label"], sub["pct_B_only"], bottom=sub["pct_shared"],          label="B only", color="#92c5de")
            plt.bar(sub["label"], sub["pct_A_only"], bottom=sub["pct_shared"]+sub["pct_B_only"], label="A only", color="#f4a582")
            plt.xticks(rotation=75, ha="right")
            plt.ylabel("% of Significant Vertices")
            plt.title(f"{key['condition']} | {key['comparison']} | {key['type']} | fold {key['fold']}")
            plt.legend()
            plt.tight_layout()
            fname = f"{key['condition']}__{key['comparison']}__{key['type']}__F{key['fold']}.png" \
                    .replace(" ", "_").replace("(", "").replace(")", "")
            plt.savefig(Path(bars_dir) / fname, dpi=dpi)
            plt.close()

    else:  # compare_mode == "global"
        # exactly one bar plot per (condition, comparison, type)
        key_cols = ["condition", "comparison", "type"]
        keys = df[key_cols].drop_duplicates()

        for _, key in keys.iterrows():
            sub = df[
                (df["condition"]  == key["condition"]) &
                (df["comparison"] == key["comparison"]) &
                (df["type"]       == key["type"])
            ].copy()

            sub["total"] = sub["pct_A_only"] + sub["pct_B_only"] + sub["pct_shared"]
            # sub = sub[sub["total"] >= abr_min_total_pct].sort_values("pct_B_only", ascending=False)
            sub = sub[sub["total"] >= abr_min_total_pct].copy()
            sub["Method_B_Unique_Contribution_Ratio"] = (sub["pct_B_only"] / sub["total"]).replace([np.inf, -np.inf], np.nan).fillna(0)
            sub = sub.sort_values("Method_B_Unique_Contribution_Ratio", ascending=False)
            if sub.empty:
                continue

            plt.figure(figsize=(16, 6))
            plt.bar(sub["label"], sub["pct_shared"],                   label="Shared",  color="#8073ac")
            plt.bar(sub["label"], sub["pct_B_only"], bottom=sub["pct_shared"],          label="B only", color="#92c5de")
            plt.bar(sub["label"], sub["pct_A_only"], bottom=sub["pct_shared"]+sub["pct_B_only"], label="A only", color="#f4a582")
            plt.xticks(rotation=75, ha="right")
            plt.ylabel("% of Significant Vertices")
            # include mode tag in title if you want it visible:
            mode_tag = df["mode"].iloc[0] if "mode" in df.columns and df["mode"].nunique()==1 else "global"
            plt.title(f"{key['condition']} | {key['comparison']} | {key['type']} | {mode_tag}")
            plt.legend()
            plt.tight_layout()
            fname = f"{key['condition']}__{key['comparison']}__{key['type']}__GLOBAL.png" \
                    .replace(" ", "_").replace("(", "").replace(")", "")
            plt.savefig(Path(bars_dir) / fname, dpi=dpi)
            plt.close()

    return df


mpl.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
})

def bland_altman_data_with_sem(acc_baseline_folds, acc_method_folds, mode="raw"):
    mode = mode.lower()
    acc1 = np.array(acc_baseline_folds)
    acc2 = np.array(acc_method_folds)
    n_folds = acc1.shape[0]
    
    if mode == "log_odds":
        eps = 1e-6
        base = np.clip(acc1, eps, 1 - eps)
        meth = np.clip(acc2, eps, 1 - eps)
        base_trans = np.log(base / (1 - base))
        meth_trans = np.log(meth / (1 - meth))
        fold_means = (base_trans + meth_trans) / 2
        fold_diffs = meth_trans - base_trans
        xlabel = "Mean Log-Odds Accuracy"
        ylabel = "Log-Odds Difference"
    elif mode == "raw":
        base = acc1 * 100
        meth = acc2 * 100
        fold_means = (base + meth) / 2
        fold_diffs = meth - base
        xlabel = "Mean Accuracy (%)"
        ylabel = "Difference (Percentage Points)"
    elif mode == "sym_diff":
        fold_diffs = 200 * (acc2 - acc1) / (acc2 + acc1)
        fold_means = 100 * (acc1 + acc2) / 2
        xlabel = "Mean Accuracy (%)"
        ylabel = "Symmetric Relative Difference (%)"
    elif mode == "sym_error":
        e_baseline = 1 - acc1
        e_method = 1 - acc2
        fold_diffs = 200 * (e_method - e_baseline) / (e_method + e_baseline)
        fold_means = 100 * (e_baseline + e_method) / 2
        xlabel = "Mean Error Rate (%)"
        ylabel = "Symmetric Relative Error Difference (%)"
    else:
        raise ValueError("Invalid mode.")

    mean_vals_mean = np.mean(fold_means, axis=0)
    diff_mean = np.mean(fold_diffs, axis=0)
    mean_vals_sem = np.std(fold_means, axis=0, ddof=1) / np.sqrt(n_folds)
    diff_sem = np.std(fold_diffs, axis=0, ddof=1) / np.sqrt(n_folds)
    return mean_vals_mean, mean_vals_sem, diff_mean, diff_sem, xlabel, ylabel


def plot_ifa_bland_altman_grid(
    condition_paths,
    condition_labels,
    *,
    pipelines=("GICA", "parcel_IFA", "voxel_IFA"),
    pipeline_labels=("GICA", "IFA (Parcellated)", "IFA (Grayordinate)"),
    folds=(0, 1, 2, 3, 4),
    nPCA=8,
    mode="log_odds",
    share_axes=False,
    fig_title=None,
    save_svg_path=None,
    verbose=True,
):
    """
    Grid of Bland–Altman plots comparing GICA vs IFA variants across conditions.
    Looks for results at: <cond>/fold_{k}/nPCA_{nPCA}/Results/{PIPE}/results.pkl

    Auto-detects the (single) classifier in each map result dict.
    """
    condition_paths = [Path(p) for p in condition_paths]
    n_conditions = len(condition_paths)

    comp_titles = [
        f"{pipeline_labels[1]} - {pipeline_labels[0]} (Basic)",
        f"{pipeline_labels[1]} - {pipeline_labels[0]} (Discrim)",
        f"{pipeline_labels[1]}: Discrim - Basic",
        f"{pipeline_labels[2]} - {pipeline_labels[0]} (Basic)",
        f"{pipeline_labels[2]} - {pipeline_labels[0]} (Discrim)",
        f"{pipeline_labels[2]}: Discrim - Basic",
    ]
    n_cols = len(comp_titles)

    if share_axes:
        fig, axs = plt.subplots(
            n_conditions, n_cols, sharex=True, sharey=True,
            figsize=(4 * n_cols, 3.5 * n_conditions), squeeze=False
        )
    else:
        fig, axs = plt.subplots(
            n_conditions, n_cols,
            figsize=(4 * n_cols, 3.5 * n_conditions), squeeze=False
        )

    global_means_list, global_diffs_list = [], []

    def _load_one_results(cpath, pipe, fold):
        fpath = cpath / f"fold_{fold}" / f"nPCA_{nPCA}" / "Results" / pipe / "results.pkl"
        if not fpath.exists():
            if verbose:
                print(f"[missing] {fpath}")
            return None
        try:
            with open(fpath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            if verbose:
                print(f"[error] reading {fpath}: {e}")
            return None

    def _only_clf_acc(d):
        """Return the accuracy from the only classifier present in dict d."""
        # d: {"ClassifierName": {"accuracy": ...}, ... } — assume exactly one key in practice
        keys = list(d.keys())
        if not keys:
            raise ValueError("Empty classifier dict in results.")
        if len(keys) > 1 and verbose:
            print(f"[warn] multiple classifiers {keys}; using '{keys[0]}'")
        return d[keys[0]]["accuracy"]

    for i, cpath in enumerate(condition_paths):
        if verbose:
            print(f"[condition] {cpath}")

        methods = []
        for pipe in pipelines:
            map_acc, discrim_map_acc = [], []
            if verbose:
                print(f"  [pipeline] {pipe}")
            for fold in folds:
                res = _load_one_results(cpath, pipe, fold)
                if res is None:
                    continue
                try:
                    # Baseline map accuracies
                    map_acc.append(np.sort([_only_clf_acc(d) for d in res["Spatial_discrim"][0]]))
                    # Discriminative direction accuracies (your index [1][2])
                    discrim_map_acc.append(np.sort([_only_clf_acc(d) for d in res["Spatial_discrim"][1][2]]))
                except Exception as e:
                    if verbose:
                        print(f"    [warn] parsing accuracies (fold {fold}): {e}")
            methods.append((map_acc, discrim_map_acc))

        comparisons = [
            (methods[0][0], methods[1][0], comp_titles[0]),
            (methods[0][1], methods[1][1], comp_titles[1]),
            (methods[1][0], methods[1][1], comp_titles[2]),
            (methods[0][0], methods[2][0], comp_titles[3]),
            (methods[0][1], methods[2][1], comp_titles[4]),
            (methods[2][0], methods[2][1], comp_titles[5]),
        ]

        row_means, row_diffs, comp_data = [], [], []
        for acc1_folds, acc2_folds, title in comparisons:
            if len(acc1_folds) == 0 or len(acc2_folds) == 0:
                mean_vals_mean = np.array([np.nan])
                mean_vals_sem  = np.array([np.nan])
                diff_mean      = np.array([np.nan])
                diff_sem       = np.array([np.nan])
                xlabel = ylabel = ""
            else:
                mean_vals_mean, mean_vals_sem, diff_mean, diff_sem, xlabel, ylabel = \
                    bland_altman_data_with_sem(acc1_folds, acc2_folds, mode=mode)

            comp_data.append(dict(
                mean=mean_vals_mean, mean_err=mean_vals_sem,
                diff=diff_mean, diff_err=diff_sem,
                title=title, xlabel=xlabel, ylabel=ylabel
            ))
            row_means.append(mean_vals_mean)
            row_diffs.append(diff_mean)
            if share_axes and mean_vals_mean.size:
                global_means_list.append(mean_vals_mean)
                global_diffs_list.append(diff_mean)

        if not share_axes:
            try:
                all_means = np.concatenate(row_means); all_diffs = np.concatenate(row_diffs)
                all_means = all_means[~np.isnan(all_means)]; all_diffs = all_diffs[~np.isnan(all_diffs)]
                mx = 0.05 * (np.ptp(all_means) if all_means.size else 1.0)
                my = 0.05 * (np.ptp(all_diffs) if all_diffs.size else 1.0)
                row_xlim = (np.min(all_means) - mx, np.max(all_means) + mx) if all_means.size else (0,1)
                row_ylim = (np.min(all_diffs) - my, np.max(all_diffs) + my) if all_diffs.size else (-1,1)
            except ValueError:
                row_xlim, row_ylim = (0,1), (-1,1)

        for j, data in enumerate(comp_data):
            ax = axs[i, j]
            ax.errorbar(
                data["mean"], data["diff"],
                xerr=data["mean_err"], yerr=data["diff_err"],
                fmt='o', markersize=4, alpha=0.8, capsize=3
            )
            if not share_axes:
                ax.set_xlim(row_xlim); ax.set_ylim(row_ylim)

            if data["diff"].size and not np.all(np.isnan(data["diff"])):
                overall_mean_diff = np.nanmean(data["diff"])
                ax.axhline(overall_mean_diff, color='blue', linestyle='--', linewidth=1)
                xlim = ax.get_xlim()
                ax.text(xlim[1], overall_mean_diff, f"Mean = {overall_mean_diff:.2f}",
                        ha='right', va='bottom', color='blue', fontsize=9)
            ax.axhline(0, color='red', linestyle='--', linewidth=1)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.set_xlabel(data["xlabel"] if data["xlabel"] else "Mean")

            if j == 0:
                ax.set_ylabel(data["ylabel"] if data["ylabel"] else "Difference")
                ax.text(-0.4, 0.5, condition_labels[i],
                        transform=ax.transAxes, rotation=90, fontsize=12,
                        va='center', ha='center', fontweight='bold')
            elif not share_axes:
                ax.set_yticklabels([])

            if i == 0:
                ax.set_title(data["title"], fontsize=11, fontweight='bold')

    if share_axes and global_means_list and global_diffs_list:
        gm = np.concatenate(global_means_list); gd = np.concatenate(global_diffs_list)
        gm = gm[~np.isnan(gm)]; gd = gd[~np.isnan(gd)]
        if gm.size:
            mxg = 0.05 * np.ptp(gm); myg = 0.05 * np.ptp(gd)
            global_xlim = (np.min(gm) - mxg, np.max(gm) + mxg)
            global_ylim = (np.min(gd) - myg, np.max(gd) + myg)
        else:
            global_xlim, global_ylim = (0,1), (-1,1)
        for ax in axs.flat:
            ax.set_xlim(global_xlim); ax.set_ylim(global_ylim)

    if fig_title is None:
        fig_title = f"Bland–Altman of Ranked Spatial Map {mode.replace('_',' ').title()} Accuracies (Mean ± SEM Across Folds)"
    fig.suptitle(fig_title, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_svg_path is not None:
        save_svg_path = Path(save_svg_path)
        save_svg_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_svg_path, format="svg", bbox_inches="tight")
        if verbose:
            print(f"[saved] {save_svg_path}")

    return fig, axs



def summarize_connectivity_and_accuracy(
    condition_paths,
    *,
    pipelines=("GICA", "parcel_IFA", "voxel_IFA"),
    folds=(0, 1, 2, 3, 4),
    nPCA=8,
    verbose=True,
    as_percent=True
):
    condition_paths = [Path(p) for p in condition_paths]
    summary = {}

    def pick_classifier_acc(class_result):
        """Auto-pick the only classifier present and return its accuracy."""
        if not isinstance(class_result, dict) or not class_result:
            raise ValueError("Class_Result is empty or not a dict.")
        keys = list(class_result.keys())
        if len(keys) > 1 and verbose:
            print(f"[warn] multiple classifiers found {keys}; using '{keys[0]}'")
        key = keys[0]
        return class_result[key]["accuracy"]

    for cond_path in condition_paths:
        if verbose:
            print(cond_path)
        summary[str(cond_path)] = {}

        for pipe in pipelines:
            scores = []
            significant_connections = []
            possible = []

            for fold in folds:
                res_path = cond_path / f"fold_{fold}" / f"nPCA_{nPCA}" / "Results" / pipe / "results.pkl"
                if not res_path.exists():
                    if verbose:
                        print(f"Missing: {res_path}")
                    continue
                try:
                    with open(res_path, "rb") as f:
                        res = pickle.load(f)
                except Exception as e:
                    if verbose:
                        print(f"[error] reading {res_path}: {e}")
                    continue

                # thresholded t-values matrix
                try:
                    t_matrix = res["t_test"][1]
                except Exception as e:
                    if verbose:
                        print(f"[warn] t_test missing/malformed in {res_path}: {e}")
                    continue

                n = t_matrix.shape[0]
                triu_with_diag = np.triu(np.ones((n, n), dtype=bool), k=0)
                num_sig = int(np.sum((np.abs(t_matrix) > 0) & triu_with_diag))
                total_possible = int(np.sum(triu_with_diag))

                # classifier accuracy
                try:
                    acc = pick_classifier_acc(res["Class_Result"])
                except Exception as e:
                    if verbose:
                        print(f"[warn] Class_Result missing/malformed in {res_path}: {e}")
                    continue

                scores.append(acc)
                significant_connections.append(num_sig)
                possible.append(total_possible)

            # aggregate
            if scores:
                acc_mean, acc_std = float(np.mean(scores)), float(np.std(scores))
                if as_percent:
                    acc_mean *= 100
                    acc_std *= 100
                sig_mean, sig_std = float(np.mean(significant_connections)), float(np.std(significant_connections))
                pos_mean, pos_std = float(np.mean(possible)), float(np.std(possible))
                n_used = len(scores)
            else:
                acc_mean = acc_std = sig_mean = sig_std = pos_mean = pos_std = float("nan")
                n_used = 0

            summary[str(cond_path)][pipe] = {
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "sig_conn_mean": sig_mean,
                "sig_conn_std": sig_std,
                "possible_mean": pos_mean,
                "possible_std": pos_std,
                "n_folds_used": n_used,
            }

            if verbose:
                print("----------------", pipe, "----------------")
                print(f"Tangent Accuracy: {acc_mean:.2f}% ± {acc_std:.2f}%")
                print(f"Number of Connections: {sig_mean:.2f} ± {sig_std:.2f}")
                print(f"Total Possible Number of Connections: {pos_mean:.2f} ± {pos_std:.2f}")
                print()

    return summary


def plot_tangent_tvalues_grid(condition_paths, condition_labels, *, pipelines=("GICA", "parcel_IFA", "voxel_IFA"), pipeline_labels=("GICA", "IFA (Parcellated)", "IFA (Grayordinate)"),
    fold=0, nPCA=8, vlim_percentile=99, figsize=(20, 13), show_ticks=True, grid_linewidth=0.5, grid_linecolor="lightgray", figure_bg="#f7f7f7", cmap_colors=((0, "#2c7bb6"), (0.5, "white"), (1, "#d7191c")),
    title="Thresholded Tangent T-values (Lower Triangle Only)", save_path=None, verbose=True,):
    # --- style ---
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("white")
    mpl.rcParams["figure.facecolor"] = figure_bg
    mpl.rcParams["axes.facecolor"] = "white"

    condition_paths = [Path(p) for p in condition_paths]
    if len(pipelines) != len(pipeline_labels):
        raise ValueError("pipelines and pipeline_labels must have the same length.")
    if len(condition_paths) != len(condition_labels):
        raise ValueError("condition_paths and condition_labels must have the same length.")

    # --- load matrices ---
    tval_mats = {}
    missing = []
    for pipe in pipelines:
        for cond_path in condition_paths:
            res_path = cond_path / f"fold_{fold}" / f"nPCA_{nPCA}" / "Results" / pipe / "results.pkl"
            if not res_path.exists():
                missing.append(str(res_path))
                if verbose:
                    print(f"[missing] {res_path}")
                continue
            try:
                with open(res_path, "rb") as f:
                    res = pickle.load(f)
                # tangent_t_test returns: (diff_thresholded_matrix, t_values_thresholded_matrix, groupA, groupB)
                tval = res["t_test"][1]
                tval_mats[(pipe, str(cond_path))] = tval
            except Exception as e:
                if verbose:
                    print(f"[error] Failed to read {res_path}: {e}")

    if not tval_mats:
        raise RuntimeError("No T-value matrices were loaded. Check paths and inputs.")

    # --- color scale from global percentile of |T| ---
    all_vals = np.concatenate([np.abs(m).ravel() for m in tval_mats.values()])
    vlim = float(np.nanpercentile(all_vals, vlim_percentile))
    cmap = LinearSegmentedColormap.from_list("custom_cmap", list(cmap_colors))

    # --- tick labels from the first available matrix ---
    first_mat = next(iter(tval_mats.values()))
    n_connections = first_mat.shape[0]
    tick_labels = [str(i) for i in range(n_connections)] if show_ticks else False

    # --- figure grid ---
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        nrows=len(pipelines) + 1,
        ncols=len(condition_paths),
        height_ratios=[0.15] + [1] * len(pipelines),
        hspace=0.35,
        wspace=0.1
    )

    # top row: condition labels
    for j, cond_label in enumerate(condition_labels):
        ax_label = fig.add_subplot(outer[0, j])
        ax_label.axis("off")
        ax_label.text(
            0.5, 0.5, cond_label, fontsize=12, ha="center", va="center", fontweight="bold"
        )

    # heatmaps
    axes = []
    for i, (pipe, pipe_label) in enumerate(zip(pipelines, pipeline_labels)):
        row_axes = []
        for j, cond_path in enumerate(condition_paths):
            ax = fig.add_subplot(outer[i + 1, j])
            key = (pipe, str(cond_path))
            if key not in tval_mats:
                # if missing, annotate
                ax.axis("off")
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", fontsize=12, color="gray")
                row_axes.append(ax)
                continue

            mat = tval_mats[key]
            hm = sns.heatmap(
                mat,
                cmap=cmap,
                center=0,
                vmin=-vlim,
                vmax=vlim,
                square=True,
                xticklabels=tick_labels,
                yticklabels=tick_labels,
                linewidths=grid_linewidth,
                linecolor=grid_linecolor,
                cbar=False,
                ax=ax
            )

            # ticks
            ax.tick_params(axis="x", labelsize=8, rotation=90, bottom=True)
            ax.tick_params(axis="y", labelsize=8, left=True)

            # left labels: pipeline name
            if j == 0:
                ax.set_ylabel(pipe_label, fontsize=12, fontweight="bold", labelpad=10)

            # title: sum of |T|
            ax.set_title(f"∑|T| = {np.sum(np.abs(mat)):.2f}", fontsize=10, pad=8)
            row_axes.append(ax)
        axes.append(row_axes)

    # shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.30, 0.015, 0.4])
    norm = plt.Normalize(vmin=-vlim, vmax=vlim)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Tangent T-values", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(title, fontsize=18, y=0.99)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig, axes, tval_mats



def compose_filter_2d_panels(
    condition_paths,
    condition_labels,
    pipelines=("GICA", "parcel_IFA", "voxel_IFA"),
    pipeline_labels=("GICA", "IFA (Parcellated)", "IFA (Grayordinate)"),
    fold=0,
    nPCA=8,
    feature_kind="log-var",
    panel_width_px=400,
    panel_height_px=350,
    dpi=150,
    save_path="composite_2d.svg",
):
    condition_paths = [Path(p) for p in condition_paths]
    n_rows = len(pipelines)
    n_cols = len(condition_paths)

    # Load and rasterize SVGs
    images = []
    missing = []
    for pipe in pipelines:
        row_imgs = []
        for cond_path in condition_paths:
            file_path = (
                Path(cond_path)
                / f"fold_{fold}"
                / f"nPCA_{nPCA}"
                / "Results"
                / pipe
                / f"{pipe}_{feature_kind}_2d.svg"
            )
            if not file_path.exists():
                row_imgs.append(None)
                missing.append((pipe, str(cond_path)))
            else:
                with open(file_path, "rb") as f:
                    png = cairosvg.svg2png(
                        bytestring=f.read(),
                        output_width=panel_width_px,
                        output_height=panel_height_px,
                    )
                img = Image.open(io.BytesIO(png))
                row_imgs.append(img)
        images.append(row_imgs)

    # Plot grid
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(n_cols * 4.5, n_rows * 3.8),
        squeeze=False,
    )

    for i, pipe in enumerate(pipelines):
        for j, cond_label in enumerate(condition_labels):
            ax = axes[i, j]
            ax.axis("off")
            if images[i][j] is not None:
                ax.imshow(images[i][j])
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center", color="red")
            if i == 0:
                ax.set_title(cond_label, fontsize=12, fontweight="bold")

    # Row labels (pipelines) on the left
    for i, label in enumerate(pipeline_labels):
        axes[i, 0].text(
            -0.15, 0.5, label,
            transform=axes[i, 0].transAxes,
            fontsize=12, fontweight="bold",
            rotation=90, ha="center", va="center"
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg", dpi=dpi, bbox_inches="tight")
    plt.show()

    return fig, axes, missing

def flatten_spatial_summary_for_io(spat_summary):
    """
    Convert { (cond, pipe): {...} } -> list-of-rows with plain fields,
    so it’s safe for JSON/CSV.
    """
    rows = []
    for (cond_label, pipeline), block in spat_summary.items():
        rows.append({
            "condition": cond_label,
            "pipeline": pipeline,
            "folds_used": block.get("folds_used", None),
            # keep the raw lists so you still have fold-level values
            "std_n_sig": block.get("std", {}).get("n_sig", []),
            "std_logp_sum": block.get("std", {}).get("logp_sum", []),
            "discrim_n_sig": block.get("discrim", {}).get("n_sig", []),
            "discrim_logp_sum": block.get("discrim", {}).get("logp_sum", []),
        })
    return rows


def run_full_ifa_report(
    *,
    condition_paths,
    condition_labels,
    pipelines=("GICA", "parcel_IFA", "voxel_IFA"),
    pipeline_labels=("GICA", "IFA (Parcellated)", "IFA (Grayordinate)"),
    folds=(0,1,2,3,4),
    nPCA=8,
    # optional extras
    nPCA_all=None,                  # e.g. [3, 8, 13, 23, 33, 48]; if None, skip model-order plot
    feature_kind="log-var",         # for compose_filter_2d_panels
    single_fold=0,              # which fold to show in the T grid figure
    bland_altman_mode="log_odds",   # "raw", "log_odds", "sym_diff", "sym_error"
    abr_min_total_pct=5.0,
    alpha=0.05,
    out_dir="ifa_full_report"):
    """
    Run the complete visualization + summary pipeline and keep every artifact
    under one root output directory.

    Returns
    -------
    outputs: dict  # paths and returns for all called functions
    """
    root = Path(out_dir)
    figs_dir = root / "figs"
    csv_dir = root / "csv"
    figs_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    outputs = {"root": str(root), "figs": {}, "csv": {}, "returns": {}}

    # 1) Filter 2D panels (FKT 2D)
    comp2d_path = figs_dir / f"composite_{feature_kind}_2d.svg"
    fig, axes, missing = compose_filter_2d_panels(
        condition_paths=condition_paths,
        condition_labels=condition_labels,
        pipelines=pipelines,
        pipeline_labels=pipeline_labels,
        fold=single_fold,              # you can change to a loop if you want per-fold grids
        nPCA=nPCA,
        feature_kind=feature_kind,
        panel_width_px=400,
        panel_height_px=350,
        dpi=150,
        save_path=str(comp2d_path)
    )
    outputs["figs"]["composite_2d"] = str(comp2d_path)
    outputs["returns"]["compose_filter_2d_panels"] = {"missing": missing}

    # 2) Tangent T-values grid (one fold)
    tvals_grid_path = figs_dir / "tangent_tvals.svg"
    fig, axes, tval_mats = plot_tangent_tvalues_grid(
        condition_paths=condition_paths,
        condition_labels=condition_labels,
        pipelines=pipelines,
        pipeline_labels=pipeline_labels,
        fold=single_fold,
        nPCA=nPCA,
        save_path=str(tvals_grid_path)
    )
    outputs["figs"]["tangent_tvals_grid"] = str(tvals_grid_path)
    outputs["returns"]["plot_tangent_tvalues_grid"] = {"keys": list(tval_mats.keys())}

    # 3) Connectivity + accuracy summary (and CSV)
    summary = summarize_connectivity_and_accuracy(
        condition_paths=condition_paths,
        pipelines=pipelines,
        folds=folds,
        nPCA=nPCA,
        verbose=False,
        as_percent=True
    )
    conn_csv = csv_dir / "connectivity_accuracy_summary.json"
    conn_flat_csv = csv_dir / "connectivity_accuracy_summary.csv"
    # write JSON
    conn_csv.write_text(json.dumps(summary, indent=2))
    # and a flat CSV for convenience
    flat_rows = []
    for cond, pdict in summary.items():
        for pipe, stats in pdict.items():
            row = {"condition_path": cond, "pipeline": pipe, **stats}
            flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(conn_flat_csv, index=False)
    outputs["csv"]["connectivity_accuracy_json"] = str(conn_csv)
    outputs["csv"]["connectivity_accuracy_csv"] = str(conn_flat_csv)

    # 4) Bland–Altman grid
    ba_svg = figs_dir / f"ifa_bland_altman_{bland_altman_mode}.svg"
    fig, axs = plot_ifa_bland_altman_grid(
        condition_paths=condition_paths,
        condition_labels=condition_labels,
        pipelines=pipelines,
        pipeline_labels=pipeline_labels,
        folds=folds,
        nPCA=nPCA,
        mode=bland_altman_mode,
        share_axes=False,
        save_svg_path=str(ba_svg),
        verbose=False,
    )
    outputs["figs"]["bland_altman"] = str(ba_svg)

    # 5) Spatial comparisons (3 modes) — each gets its own subdir within root
    out_agg_first = root / "compare_aggregate_first"
    out_comp_first = root / "compare_compare_first"
    out_foldwise  = root / "compare_foldwise"

    _ = compare_pipelines_spatial(
        condition_paths=condition_paths,
        condition_labels=condition_labels,
        pipelines=pipelines,
        pipeline_labels=pipeline_labels,
        nPCA=nPCA,
        folds=folds,
        compare_mode="global",
        aggregation="aggregate_before_compare",
        min_join=False,
        abr_min_total_pct=abr_min_total_pct,
        out_dir=str(out_agg_first))
    outputs["figs"]["compare_aggregate_first"] = str(out_agg_first)

    _ = compare_pipelines_spatial(
        condition_paths=condition_paths,
        condition_labels=condition_labels,
        pipelines=pipelines,
        pipeline_labels=pipeline_labels,
        nPCA=nPCA,
        folds=folds,
        compare_mode="global",
        aggregation="aggregate_after_compare",
        min_join=False,
        abr_min_total_pct=abr_min_total_pct,
        out_dir=str(out_comp_first))
    outputs["figs"]["compare_compare_first"] = str(out_comp_first)

    _ = compare_pipelines_spatial(
        condition_paths=condition_paths,
        condition_labels=condition_labels,
        pipelines=pipelines,
        pipeline_labels=pipeline_labels,
        nPCA=nPCA,
        folds=folds,
        compare_mode="foldwise",
        abr_min_total_pct=abr_min_total_pct,
        out_dir=str(out_foldwise)
    )
    outputs["figs"]["compare_foldwise"] = str(out_foldwise)

    # 6) Spatial tests summary (prints + returns); also write JSON+CSV safely
    spat_summary, _ = summarize_spatial_tests(
        condition_paths=condition_paths,
        condition_labels=condition_labels,
        pipelines=pipelines,
        nPCA=nPCA,
        folds=list(folds),
        alpha=alpha,
    )

    spat_rows = flatten_spatial_summary_for_io(spat_summary)

    spat_json = csv_dir / "spatial_tests_summary.json"
    spat_csv  = csv_dir / "spatial_tests_summary.csv"
    spat_json.write_text(json.dumps(spat_rows, indent=2))
    pd.DataFrame(spat_rows).to_csv(spat_csv, index=False)

    outputs["csv"]["spatial_tests_summary_json"] = str(spat_json)
    outputs["csv"]["spatial_tests_summary_csv"]  = str(spat_csv)

    # 7) Accuracy vs model order (optional; only if nPCA_all provided)
    if nPCA_all is not None:
        acc_model_svg = figs_dir / "accuracy_vs_model_order.svg"
        fig, axes, stats = plot_accuracy_vs_model_order_robust(
            condition_paths=condition_paths,
            condition_labels=condition_labels,
            nPCA_all=nPCA_all,
            pipelines=pipelines,
            pipeline_labels=pipeline_labels,
            folds=list(folds),
            save_path=str(acc_model_svg),
        )
        outputs["figs"]["accuracy_vs_model_order"] = str(acc_model_svg)
        outputs["returns"]["accuracy_vs_model_order_stats"] = stats

    # 8) Reconstruction KDE (per condition, pooled across folds/pipelines)
    recon_svg = figs_dir / "reconstruction_kde.svg"
    fig, axes, pooled = plot_reconstruction_kde(
        condition_paths=condition_paths,
        condition_labels=condition_labels,
        pipelines=pipelines,
        pipeline_labels=pipeline_labels,
        nPCA=nPCA,
        folds=list(folds),
        save_path=str(recon_svg),
    )
    outputs["figs"]["reconstruction_kde"] = str(recon_svg)
    outputs["returns"]["reconstruction_pooled"] = pooled

    # spatial tests summary
    index_json = root / "artifact_index.json"
    spat_json.write_text(json.dumps(_json_safe(spat_rows), indent=2))
    index_json.write_text(json.dumps(_json_safe(outputs), indent=2))

    return outputs