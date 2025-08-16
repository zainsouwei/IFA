import os, sys, json, pickle
import numpy as np

# os.environ["LD_PRELOAD"] = os.path.join(os.environ["CONDA_PREFIX"], "lib", "libstdc++.so.6")
import warnings, logging, sklearn
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
warnings.filterwarnings("ignore", message=r"pixdim\[1,2,3\].*", category=UserWarning)
logging.getLogger("nibabel").setLevel(logging.ERROR)

# Add the path to custom modules
sys.path.append('/project/3022057.01/IFA/utils')

# Import necessary modules
from PCA import migp
from filters import voxelwise_FKT


def highdim_fkt(outputfolder, voxel_filters_dir, train_paths, train_labels, a_label, b_label, mA, mB, batch_size=5,cifti=False):
    try:
        with open(os.path.join(outputfolder, "settings.json"), "r") as f:
            settings = json.load(f)
        

        n_filters_per_group = settings["n_filters_per_group"]
        cov_log = settings["cov_log"]
        shrink = settings["shrinkage"]
        # TODO changed here (no more partial)
        A_partial = migp(train_paths[train_labels == a_label], m=mA, n_jobs=20,batch_size=batch_size,vt=None)
        B_partial = migp(train_paths[train_labels == b_label], m=mB, n_jobs=20,batch_size=batch_size,vt=None)
        
        print("Finished MIGP prior to Voxel FKT; Running Voxel FKT on GPU")
        voxelwise_FKT(groupA=A_partial, groupB=B_partial, 
                        n_filters_per_group=n_filters_per_group, 
                        groupA_paths=None, groupB_paths=None, 
                        paths=False,log=cov_log,shrinkage=shrink,
                        cov_method='svd',outputfolder=voxel_filters_dir, save=False, save_img=cifti)

    except Exception as e:
        print(f"Error in run_voxel_fkt: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Exit with non-zero code so SLURM knows the job failed
        sys.exit(1)

def main(outputfolder, fold_dir, voxel_filters_dir):
    # Load settings and training split
    with open(os.path.join(outputfolder, "settings.json"), "r") as f:
        settings = json.load(f)
    a_label = settings["a_label"]
    b_label = settings["b_label"]
    cifti   = settings["cifti"]

    with open(os.path.join(outputfolder, "paths.pkl"), "rb") as f:
        paths = pickle.load(f)

    labels = np.load(os.path.join(outputfolder, "labels.npy"))
    time_points = np.load(os.path.join(outputfolder, "time_points.npy"))

    indices_dir = os.path.join(fold_dir, "Indices")
    train_idx = np.load(os.path.join(indices_dir, "train_idx.npy"))

    train_paths  = paths[train_idx]
    train_labels = labels[train_idx]

    # match your main computation of mA/mB
    mA = 2 * time_points[train_idx][train_labels == a_label].max()
    mB = 2 * time_points[train_idx][train_labels == b_label].max()

    os.makedirs(voxel_filters_dir, exist_ok=True)

    # Compute and save voxel filters (writes filters.npy inside voxel_filters_dir)
    highdim_fkt(outputfolder, voxel_filters_dir, train_paths, train_labels, a_label, b_label, int(mA), int(mB), batch_size=1, cifti=cifti)

    # Sanity check
    out_file = os.path.join(voxel_filters_dir, "filters.npy")
    if not os.path.exists(out_file):
        raise RuntimeError(f"Voxel FKT finished but {out_file} not found.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: run_voxel_fkt.py <outputfolder> <fold_output_dir> <voxel_filters_dir>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])