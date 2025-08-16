#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Make sure we can import visualize.py
UTILS_DIR = "/project/3022057.01/IFA/utils"
if UTILS_DIR not in sys.path:
    sys.path.append(UTILS_DIR)

from visualize import run_full_ifa_report  # noqa

def main():
    if len(sys.argv) < 2:
        print("Usage: run_fold_final.py <CONDITION_PATH>", file=sys.stderr)
        sys.exit(2)

    condition_path = sys.argv[1]
    if not os.path.exists(condition_path):
        print(f"ERROR: condition path does not exist: {condition_path}", file=sys.stderr)
        sys.exit(3)

    # Derive a human-ish label from the path (adjust if you want something specific)
    cond_name = Path(condition_path).name
    condition_labels = [cond_name.replace("_", " ")]

    # Where to write all artifacts
    out_dir = f"ifa_full_report_{cond_name}"

    print(f"[INFO] Running report for: {condition_path}")
    print(f"[INFO] Output dir       : {out_dir}")

    # Fixed knobs (change here if needed)
    pipelines = ("GICA", "parcel_IFA", "voxel_IFA")
    pipeline_labels = ("GICA", "IFA (Parcellated)", "IFA (Grayordinate)")
    folds = (0, 1, 2, 3, 4)
    nPCA = 8

    # Execute
    outputs = run_full_ifa_report(
        condition_paths=[condition_path],
        condition_labels=condition_labels,
        pipelines=pipelines,
        pipeline_labels=pipeline_labels,
        folds=folds,
        nPCA=nPCA,
        nPCA_all=None,          # set a list if you want the model-order plot
        feature_kind="log-var",
        single_fold=1,
        bland_altman_mode="log_odds",
        abr_min_total_pct=5.0,
        alpha=0.05,
        out_dir=out_dir,
    )

    idx = Path(outputs["root"]) / "artifact_index.json"
    print(f"[DONE] Artifacts index -> {idx}")

if __name__ == "__main__":
    main()