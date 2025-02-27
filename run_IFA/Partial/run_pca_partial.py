import sys
import pickle
import argparse

sys.path.append('/project/3022057.01/IFA/utils')

import os
import json
# End of save block
from PCA import migp, PPCA
import numpy as np
from filters import voxelwise_FKT


def run_pca(outputfolder, fold_output_dir, voxel_filters_dir, batch_size=5):
    try:
        with open(os.path.join(outputfolder, "settings.json"), "r") as f:
            settings = json.load(f)
        

        n_filters_per_group = settings["n_filters_per_group"]
        # a_label = settings["a_label"]
        # b_label = settings["b_label"]

        # labels = np.load(os.path.join(outputfolder, "labels.npy"))
        # with open(os.path.join(outputfolder, "paths.pkl"), "rb") as f:
        #     paths = pickle.load(f)

        # indices_dir = os.path.join(fold_output_dir, "Indices")
        # train_idx = np.load(os.path.join(indices_dir, "train_idx.npy"))
        # train_labels = labels[train_idx]
        # train_paths = paths[train_idx]


        migp_dir = os.path.join(fold_output_dir, "MIGP")
        # if not os.path.exists(migp_dir):
        #     os.makedirs(migp_dir)
        
        ## last element in path list is number of timepoints, see load_subject in preprocessing
        # m = train_paths[0][-1]
        # print("Keep this many pseudotime points",m, flush=True)
        # reducedsubsA = migp(train_paths[train_labels == a_label],batch_size=batch_size,m=m)
        # reducedsubsB = migp(train_paths[train_labels == b_label],batch_size=batch_size,m=m)
        # np.save(os.path.join(migp_dir, "reducedsubsA.npy"), reducedsubsA)
        # np.save(os.path.join(migp_dir, "reducedsubsB.npy"), reducedsubsB)
        # reducedsubs = np.concatenate((reducedsubsA, reducedsubsB), axis=0)
        # np.save(os.path.join(migp_dir, "reducedsubs.npy"), reducedsubs)
        
        reducedsubsA = np.load(os.path.join(migp_dir, "reducedsubsA.npy"))
        reducedsubsB = np.load(os.path.join(migp_dir, "reducedsubsB.npy"))
        
        filters_dir = os.path.dirname(voxel_filters_dir)
        vt = np.load(os.path.join(filters_dir, "vt.npy"))


        A_partial = reducedsubsA - (reducedsubsA@np.linalg.pinv(vt))@vt
        B_partial = reducedsubsB - (reducedsubsB@np.linalg.pinv(vt))@vt

        voxelwise_FKT(groupA=A_partial, groupB=B_partial, 
                        n_filters_per_group=n_filters_per_group, 
                        groupA_paths=None, groupB_paths=None, 
                        paths=False,log=False,shrinkage=0.01,
                        cov_method='svd',outputfolder=voxel_filters_dir, save=False)

    except Exception as e:
        print(f"Error in run_pca: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Exit with non-zero code so SLURM knows the job failed
        sys.exit(1)
        
# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PCA analysis for a given fold")
    parser.add_argument("outputfolder", type=str, help="Path to the output folder")
    parser.add_argument("fold_output_dir", type=str, help="Path to the fold output folder")
    parser.add_argument("voxel_filters_dir", type=str, help="Path to the fold voxel filter output folder")
    args = parser.parse_args()
    run_pca(args.outputfolder, args.fold_output_dir, args.voxel_filters_dir)
