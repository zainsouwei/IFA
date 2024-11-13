import sys
import pickle
import argparse

sys.path.append('/project/3022057.01/IFA/utils')

import os
import json
# End of save block
from PCA import migp, PPCA
import numpy as np


def run_pca(outputfolder, fold_output_dir, nPCA):
    settings_filepath = os.path.join(outputfolder, "settings.json")
    # Load the settings from the JSON file
    with open(settings_filepath, "r") as f:
        settings = json.load(f)
    a_label = int(settings["a_label"])
    b_label = int(settings["b_label"])

    labels = np.load(os.path.join(outputfolder,"labels.npy"))
    with open(os.path.join(outputfolder, "paths.pkl"), "rb") as f:
        paths = pickle.load(f)      
    indices_dir = os.path.join(fold_output_dir, "Indices")
    train_idx = np.load(os.path.join(indices_dir, "train_idx.npy"))
    train_labels = labels[train_idx]
    train_paths = paths[train_idx]


    migp_dir = os.path.join(fold_output_dir, "MIGP")
    if not os.path.exists(migp_dir):
        os.makedirs(migp_dir)
    reducedsubsA = migp(train_paths[train_labels == a_label])
    reducedsubsB = migp(train_paths[train_labels == b_label])
    reducedsubs = np.concatenate((reducedsubsA, reducedsubsB), axis=0)
    _, vt = PPCA(reducedsubs.copy(), threshold=0.0, niters=1, n=nPCA)
    np.save(os.path.join(migp_dir, "reducedsubsA.npy"), reducedsubsA)
    np.save(os.path.join(migp_dir, "reducedsubsB.npy"), reducedsubsB)
    np.save(os.path.join(migp_dir, "reducedsubs.npy"), reducedsubs)
    np.save(os.path.join(migp_dir, "vt.npy"), vt)

# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCA analysis for a given fold")
    parser.add_argument("outputfolder", type=str, help="Path to the output folder")
    parser.add_argument("fold_output_dir", type=str, help="Path to the output folder")
    parser.add_argument("nPCA", type=int, help="Fold number")
    args = parser.parse_args()
    run_pca(args.outputfolder, args.fold_output_dir, args.nPCA)