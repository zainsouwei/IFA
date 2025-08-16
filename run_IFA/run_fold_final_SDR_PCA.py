import sys
import os
import time

# os.environ["LD_PRELOAD"] = os.path.join(os.environ["CONDA_PREFIX"], "lib", "libstdc++.so.6")
import warnings, logging, sklearn
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
warnings.filterwarnings("ignore", message=r"pixdim\[1,2,3\].*", category=UserWarning)
logging.getLogger("nibabel").setLevel(logging.ERROR)

import json
import pandas as pd
import numpy as np
import argparse
import subprocess
import pickle
import hcp_utils as hcp
from pyriemann.estimation import Covariances
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import functools
import gc
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from nilearn.input_data import NiftiMapsMasker
from nilearn.datasets import fetch_atlas_basc_multiscale_2015, fetch_atlas_smith_2009

# Add the path to custom modules
sys.path.append('/project/3022057.01/IFA/utils')

# Import necessary modules
from analysis import evaluate, compare
from PCA import PPCA, migp
from filters import orthonormalize_filters, save_brain
from ICA import ICA
from DualRegression import DualRegress
from filters import TSSF_select, TSSF, FKT, evaluate_filters
from tangent import tangent_classification
from haufe import partial_filter_dual_regression
from preprocessing import load_subject


# End of save block
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


def save_text_results(text, filepath):
    """Save text results to a file."""
    with open(filepath, "a") as f:  # Using 'a' to append results to the file
        f.write(text + "\n")

def check_job_completion(job_id):
    """Poll the status of a job and wait until it reaches a final state."""
    while True:
        job_status = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader"],
            capture_output=True, text=True
        )
        # Split lines and take the first non-empty line as the status
        state = job_status.stdout.splitlines()[0].strip()
        
        if "COMPLETED" in state:
            return True
        elif any(status in state for status in ["FAILED", "CANCELLED", "TIMEOUT"]):
            return False
        
        # Sleep for a bit before checking again
        time.sleep(120)  # Poll every 120 seconds

# TODO changed here (no more partial)
def parcellate_subject(sub, cifti=True):
    try:
        # Load subject data using our load_subject function.
        data = load_subject(sub)
        # Parcellate the residual data.
        if cifti:
            Xp = hcp.parcellate(data, hcp.mmp)
            del data  # free memory
        else:
            _, gm_mask_file = sub
            masker = NiftiMasker(mask_img=gm_mask_file, dtype=np.float32)
            masker.fit()
            data_full = masker.inverse_transform(data)
            del data
            basc = fetch_atlas_basc_multiscale_2015(resolution=122, version='asym')
            basc_labels = basc.maps
            # TODO Correlation vs Covariance
            labels_masker = NiftiLabelsMasker(labels_img=basc_labels, mask_img=gm_mask_file, standardize=False,  dtype=np.float32,resampling_target='data')
            # TODO decide if I should normalize 
            # labels_masker = NiftiLabelsMasker(labels_img=basc_labels, mask_img=gm_mask_file, standardize='zscore_sample',  dtype=np.float32,resampling_target='data')
            Xp = labels_masker.fit_transform(data_full)
            del data_full

        # If the subject is simulated (i.e. file path ends with '.npy'), return Xp without extra normalization.
        if isinstance(sub, str) and sub.endswith('.npy'):
            return Xp
        else:
            # TODO Correlation vs Covariance
            return Xp
            # TODO decide if I should normalize 
            # Otherwise, apply the final demeaning and normalization.
            # return hcp.normalize(Xp - Xp.mean(axis=1, keepdims=True))
    except Exception as e:
        print(f"Error processing subject {sub}: {e}")
        traceback.print_exc()
        raise

# TODO changed here (no more partial)
def parcellate_subjects(paths, output_dir, n_workers=20, cifti=True):
    try:
        
        os.makedirs(output_dir, exist_ok=True)

        # run subjects in parallel
        func = functools.partial(parcellate_subject, cifti=cifti)
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            partiallated = list(ex.map(func, paths))

        # validate outputs
        if any(x is None for x in partiallated):
            bad = [i for i, x in enumerate(partiallated) if x is None]
            raise RuntimeError(f"{len(bad)} subjects failed: indices {bad}")

        # check same number of parcels across subjects
        parcel_counts = {x.shape[1] for x in partiallated}
        if len(parcel_counts) != 1:
            sizes = [x.shape for x in partiallated]
            raise ValueError(f"Inconsistent parcel counts across subjects: {sizes}")
        P = parcel_counts.pop()

        # persist time series (flexible: variable T per subject)
        ts_path = os.path.join(output_dir, "parcellated_data.pkl")
        with open(ts_path, "wb") as f:
            pickle.dump(partiallated, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Parcellated data saved to {ts_path}")
        
        # covariances per subject (handles variable T)
        cov_est = Covariances(estimator='oas')
        covs = np.empty((len(partiallated), P, P), dtype=np.float32)
        for i, X in enumerate(partiallated):
            # X shape: (T, P) -> (1, P, T)
            cov = cov_est.transform(X.T[np.newaxis, :, :])[0]
            covs[i] = cov.astype(np.float32, copy=False)
        
        cov_path = os.path.join(output_dir, "parcellated_covs.npy")
        np.save(cov_path, covs)
        print(f"Covariances saved to {cov_path}")

        return partiallated, covs

    except Exception as e:
        print(f"Error in parcellation process: {e}")
        traceback.print_exc()
        raise  # Re-raise the error so the process crashes.

def major_recon_discrim(discrim_basis, major_space,output_folder):
    try:
        # Compute reconstruction
        ms64 = np.asarray(major_space, dtype=np.float64, order='C')
        pinv_ms = np.linalg.pinv(ms64, rcond=1e-12)
        reconstructed = discrim_basis.T @ pinv_ms @ ms64
        numerator = np.linalg.norm(discrim_basis.T - reconstructed, 'fro') ** 2
        denominator = np.linalg.norm(discrim_basis.T, 'fro') ** 2
        reconstruction_percentage = (1 - numerator / denominator)
        print("Reconstruction Percentage:", reconstruction_percentage)
        
        # Save the reconstruction percentage
        recon_file = os.path.join(output_folder, "discriminant_reconstruction_percentage_vt.txt")
        with open(recon_file, "w") as f:
            f.write(str(reconstruction_percentage) + "\n")
    except Exception as e:
        print("Failed to compute reconstruction percentage:", e)
        reconstruction_percentage = None

def PPCA_ICA(reducedsubs,basis=None, n_components=None,random_state=42, output_folder=None):
    if basis is None:
        _, basis = PPCA(reducedsubs.copy(), threshold=0.0, niters=1, n=n_components)
    
    os.makedirs(output_folder, exist_ok=True)

    spatial_maps = ICA(basis, output_dir=output_folder,random_state=random_state)
    spatial_maps = spatial_maps.T
    for i in range(spatial_maps.shape[1]):
        save_brain(spatial_maps[:,i], f"s_map_{i}", output_folder)

    np.save(os.path.join(output_folder, "basis.npy"), basis)
    np.save(os.path.join(output_folder, "spatial_maps.npy"), spatial_maps)

    return spatial_maps

def run_comparisons(results_list, base_output_folder, pairs, alpha=0.05):
    """
    Run pairwise comparisons for a list of evaluation results.
    
    Parameters:
    - results_list: list of evaluation results (e.g., normalized or unnormalized).
    - base_output_folder: base directory where comparison subfolders will be created.
    - pairs: list of tuples (i, j, label_one, label_two) indicating the indices in results_list and their labels.
    """
    if not os.path.exists(base_output_folder):
        os.makedirs(base_output_folder)
        
    for i, j, label_one, label_two in pairs:
        pair_dir = os.path.join(base_output_folder, f"{label_one}_vs_{label_two}")
        if not os.path.exists(pair_dir):
            os.makedirs(pair_dir)
        compare(
            results_list[i], results_list[j],
            label_one=label_one, label_two=label_two,
             alpha=alpha, output_dir=pair_dir
        )

def MIGP_wrapper(sub_paths, file_path,m=4800, n_jobs=4, batch_size=3, vt=None):
    if os.path.exists(file_path):
        print(f"Loading existing reducedsubs from {file_path}")
        reducedsubs = np.load(file_path)
    else:
        print(f"Reducedsubs {file_path} not found — running MIGP")
        reducedsubs = migp(sub_paths, m=m, n_jobs=n_jobs, batch_size=batch_size,vt=vt)
        np.save(file_path, reducedsubs)
    return reducedsubs


def run_fold(outputfolder, fold):
    
    # Read the settings from the JSON file
    with open(os.path.join(outputfolder, "settings.json"), "r") as f:
        settings = json.load(f)
        
    random_state = settings["random_state"]
    n_filters_per_group = settings["n_filters_per_group"]
    nPCA_levels = settings["nPCA_levels"]
    tangent_class = settings["tangent_class"]
    metric = settings["metric"]
    a_label = settings["a_label"]
    b_label = settings["b_label"]
    deconfound = settings["deconfound"]
    paired = settings["paired"]
    cifti = settings["cifti"]
    use_gpu = settings["gpu"]

    # Load pickle files
    with open(os.path.join(outputfolder, "paths.pkl"), "rb") as f:
        paths = pickle.load(f)

    group_path = os.path.join(outputfolder, "group_ID.pkl")
    if os.path.exists(group_path):
        with open(group_path, "rb") as f:
            group_ID = pickle.load(f)

    # Load numpy files
    sub_ID = np.load(os.path.join(outputfolder, "Sub_ID.npy"), allow_pickle=True)
    labels = np.load(os.path.join(outputfolder, "labels.npy"))
    time_points = np.load(os.path.join(outputfolder, "time_points.npy"))

    # Load Fold Specific Vairables
    fold_output_dir = os.path.join(outputfolder, f"fold_{fold}")

    summary_file_path = os.path.join(fold_output_dir, "output_summary.txt")
    indices_dir = os.path.join(fold_output_dir, "Indices")
    train_idx = np.load(os.path.join(indices_dir, "train_idx.npy"))
    test_idx = np.load(os.path.join(indices_dir, "test_idx.npy"))

    # Prepare data for train and test sets
    train_labels = labels[train_idx]
    train_paths = paths[train_idx]
    test_labels = labels[test_idx]
  
    if deconfound:
        con_confs = pd.read_pickle(os.path.join(outputfolder, "con_confounders.pkl"))
        cat_confs = pd.read_pickle(os.path.join(outputfolder, "cat_confounders.pkl"))
        
        train_con_confounders = con_confs.iloc[train_idx]
        test_con_confounders = con_confs.iloc[test_idx]
        train_cat_confounders = cat_confs.iloc[train_idx]
        test_cat_confounders = cat_confs.iloc[test_idx]
    else:
        train_con_confounders = None
        test_con_confounders = None
        train_cat_confounders = None
        test_cat_confounders = None

    # Save summary of data split
    if 'group_ID' in locals():
        train_groups = set(np.unique(group_ID[train_idx]))
        test_groups  = set(np.unique(group_ID[test_idx]))
        intersection = train_groups & test_groups
    else:
        intersection = set()

    save_text_results(f"Fold {fold + 1}:", summary_file_path)
    save_text_results(f"  Train size: {len(train_idx)}", summary_file_path)
    save_text_results(f"  Test size: {len(test_idx)}", summary_file_path)
    save_text_results(f"  Train labels distribution: {np.bincount(labels[train_idx].astype(int))}", summary_file_path)
    save_text_results(f"  Test labels distribution: {np.bincount(labels[test_idx].astype(int))}", summary_file_path)
    save_text_results(f"  Intersection of groups: {len(intersection)} (Groups: {intersection})", summary_file_path)
    
    if paired:
        paired_train = np.array_equal(
            sub_ID[train_idx][train_labels == a_label],
            sub_ID[train_idx][train_labels == b_label]
        )
        paired_test = np.array_equal(
            sub_ID[test_idx][test_labels == a_label],
            sub_ID[test_idx][test_labels == b_label]
        )
        save_text_results(f"  Paired Across Train: {paired_train}", summary_file_path)
        save_text_results(f"  Paired Across Test: {paired_test}", summary_file_path)

    filters_dir = os.path.join(fold_output_dir, "Filters")
    voxel_filters_dir = os.path.join(filters_dir, "Voxel")
    parcellated_filters_dir = os.path.join(filters_dir, "Parcellated")
    for d in (filters_dir, voxel_filters_dir, parcellated_filters_dir):
            os.makedirs(d, exist_ok=True)

    mA = 2 * time_points[train_idx][train_labels == a_label].max()
    mB = 2 * time_points[train_idx][train_labels == b_label].max()

    # Voxel FKT
    voxel_filters_path = os.path.join(voxel_filters_dir, "filters.npy")
    job_id = None
    voxel_loaded = False
    if os.path.exists(voxel_filters_path):
        print(f"[cache] Loading voxel filters from {voxel_filters_path}")
        voxel_filters = np.load(voxel_filters_path)
        voxel_loaded = True
    else:
        print("[run] voxelwise FKT")
        # TODO check batch size
        # Function save voxel level filters instead of returning
        if use_gpu:
            print("Submitting Voxel FKT to GPU")
            vfkt_script = "/project/3022057.01/IFA/run_IFA/run_voxel_fkt.sh"
            vfkt_command = [
                "sbatch",
                "--output", os.path.join(voxel_filters_dir, "vfkt-%j.out"),
                "--error", os.path.join(voxel_filters_dir, "vfkt-%j.err"),
                vfkt_script,
                outputfolder, fold_output_dir, voxel_filters_dir
            ]
            vfkt_process = subprocess.run(vfkt_command, capture_output=True, text=True)
            if vfkt_process.returncode != 0:
                print(f"Error submitting VFKT job: {vfkt_process.stderr}")
                return
            job_id = vfkt_process.stdout.strip().split()[-1]
            print(f"VFKT job submitted successfully with job ID: {job_id}")
        else:
            highdim_fkt(outputfolder, voxel_filters_dir, train_paths, train_labels, a_label, b_label, mA,mB, batch_size=1,cifti=cifti)
            voxel_filters = np.load(voxel_filters_path)
            voxel_loaded = True


    # Parcel FKT
    # Need to partial the data before parcellating; partial then parcellate each subject
    ts_path  = os.path.join(filters_dir, "parcellated_data.pkl")
    cov_path = os.path.join(filters_dir, "parcellated_covs.npy")
    if os.path.exists(ts_path) and os.path.exists(cov_path):
        print("[cache] parcellated data/covs")
        with open(ts_path, "rb") as f: 
            parcel_data = pickle.load(f)
        parcel_covs = np.load(cov_path)
    else:
        print("[run] parcellate_subjects")
        parcel_data, parcel_covs = parcellate_subjects(paths, filters_dir, n_workers=20, cifti=cifti)

    parcel_train_data = [parcel_data[i] for i in train_idx] # Can not slice due to ragged array (subjects have different length scans)
    parcel_test_data  = [parcel_data[i] for i in test_idx]
    parcel_train_covs = parcel_covs[train_idx]
    parcel_test_covs  = parcel_covs[test_idx]

    # Remove predictions before saving since that will be too long of a print
    def _strip_preds(d): 
        return {k: {kk: vv for kk, vv in v.items() if kk != "predictions"} for k, v in d.items()}

    tangent_metrics_path = os.path.join(filters_dir, "tangent_class_metrics.pkl")
    if not os.path.exists(tangent_metrics_path):
        # Run tangent classification for measuring separability in parcellated space
        # TODO decide on z scoring here, gp opt for hyperparameter selection, and decide which classifiers
        tangent_class_metrics = tangent_classification(parcel_train_covs, train_labels, parcel_test_covs, test_labels, 
                            clf_str='svc_l2_sq', z_score=0, metric=metric, deconf=deconfound, 
                            con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                            con_confounder_test=test_con_confounders, cat_confounder_test=test_cat_confounders,
                        random_state=0, n_inner_splits=5,n_cpus=20, n_batches=10)
        # Save those tangent classification results to overall fold results directory
        with open(os.path.join(filters_dir, "tangent_class_metrics.pkl"), "wb") as f:
            pickle.dump(_strip_preds(tangent_class_metrics), f)

        save_text_results("Parcellated Tangent Classification " + str(_strip_preds(tangent_class_metrics)), summary_file_path)

    filtersA_path = os.path.join(parcellated_filters_dir, "filtersA.npy")
    filtersB_path = os.path.join(parcellated_filters_dir, "filtersB.npy")
    filters_parcel_path = os.path.join(parcellated_filters_dir, "filters_parcellated.npy")

    if not all(os.path.exists(p) for p in [filtersA_path, filtersB_path, filters_parcel_path]):
        # TODO GPOT for hyperparameter selection, decide on z_scoring here
        if tangent_class:
            # Tune tangent-space model (on TRAIN only) and pick winner
            sel_path = os.path.join(parcellated_filters_dir, "tssf_selection.json")
            if os.path.exists(sel_path):
                print("[cache] loading TSSF selection")
                with open(sel_path, "r") as f:
                    sel = json.load(f)
            else:
                print("[run] TSSF hyperparameter selection")
                sel = TSSF_select(parcel_train_covs, train_labels, parcel_train_data,
                                a_label=a_label, b_label=b_label, n=n_filters_per_group, metric=metric, feature_kind="log-cov",          # or "log-var"
                                deconf=deconfound, con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders,
                                tan_model_keys=["svc_l2_sq"], final_svm_key="svc",
                                z_score_tan=0, haufe=False, 
                                n_inner_splits=5, n_cpus=20, n_batches=10, random_state=random_state,)
                with open(sel_path, "w") as f:
                    json.dump(sel, f, indent=2)
            # Fit filters with the selected model + BO-tuned params (still TRAIN only)
            _, filters_all, _, _ = TSSF(parcel_train_covs, train_labels, clf_str=sel["winner_model"], clf_params=sel["winner_theta"],
                                        metric=metric, deconf=deconfound, con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders,
                                        z_score=0, haufe=False, visualize=True, output_dir=parcellated_filters_dir,)
        else:
            _, filters_all = FKT(parcel_train_covs, train_labels, a_label, b_label,
                                    metric=metric, deconf=deconfound, 
                                    con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                                    visualize=True, output_dir=parcellated_filters_dir)
        
        # if TSSF was used then the lower label is the negative class and corresponds to eigenvalues < 1
        if a_label < b_label and tangent_class:
            filtersB = filters_all[:, -n_filters_per_group:]
            filtersA = filters_all[:, :n_filters_per_group]
        else: 
            filtersA = filters_all[:, -n_filters_per_group:]
            filtersB = filters_all[:, :n_filters_per_group]

        filters_parcellated = np.concatenate((filtersB, filtersA), axis=1)

        np.save(filtersA_path, filtersA)
        np.save(filtersB_path, filtersB)
        np.save(filters_parcel_path, filters_parcellated)


        # TODO should i z score teh derived features before classification (will also need to change in TSSF_select)
        logvar_stats, logcov_stats = evaluate_filters(parcel_train_data, train_labels, parcel_test_data, test_labels, 
                                                        filters_parcellated, metric=metric, deconf=deconfound, 
                                                        con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                                                        con_confounder_test=test_con_confounders, cat_confounder_test=test_cat_confounders,output_dir=parcellated_filters_dir)

        with open(os.path.join(filters_dir, "logvar_stats.pkl"), "wb") as f:
            pickle.dump(_strip_preds(logvar_stats), f)
        with open(os.path.join(filters_dir, "logcov_stats.pkl"), "wb") as f:
            pickle.dump(_strip_preds(logcov_stats), f)

        save_text_results("Log Var Filter Feature Classification " + str(_strip_preds(logvar_stats)), summary_file_path)
        save_text_results("Log Cov Filter Feature Classification " + str(_strip_preds(logcov_stats)), summary_file_path)
    else:
        filtersA = np.load(filtersA_path)
        filtersB = np.load(filtersB_path)
        filters_parcellated = np.load(filters_parcel_path)

    A_haufe_path = os.path.join(parcellated_filters_dir, "A_filters_haufe.npy")
    B_haufe_path = os.path.join(parcellated_filters_dir, "B_filters_haufe.npy")
    parcelvoxel_filters_path = os.path.join(parcellated_filters_dir, "filters.npy")
    if not all(os.path.exists(p) for p in [A_haufe_path, B_haufe_path, parcelvoxel_filters_path]):
        print("[run] Haufe + parcel→voxel")
        # Get indices where label == a_label
        idx_a_label = np.where(train_labels == a_label)[0]
        idx_b_label = np.where(train_labels == b_label)[0]

        # Use list comprehension to select subjects
        parcel_train_data_a = [parcel_train_data[i] for i in idx_a_label]
        parcel_train_data_b = [parcel_train_data[i] for i in idx_b_label]

        # Haufe transform and project parcellated filters to full dimension
        filtersA_transform = partial_filter_dual_regression(filtersA, parcel_train_data_a, train_paths[idx_a_label], None, workers=20)
        filtersB_transform = partial_filter_dual_regression(filtersB, parcel_train_data_b, train_paths[idx_b_label], None, workers=20)

        np.save(A_haufe_path, filtersA_transform)
        np.save(B_haufe_path, filtersB_transform)
        parcelvoxel_filters = orthonormalize_filters(filtersA_transform, filtersB_transform)
        np.save(parcelvoxel_filters_path, parcelvoxel_filters)
        if cifti:
            for i in range(parcelvoxel_filters.shape[1]):
                save_brain(parcelvoxel_filters[:,i], f"parcelvoxel_filters{i}", parcellated_filters_dir)
    else:
        print("[cache] Haufe + parcel→voxel")
        filtersA_transform = np.load(A_haufe_path)
        filtersB_transform = np.load(B_haufe_path)
        parcelvoxel_filters = np.load(parcelvoxel_filters_path)

    if use_gpu and not voxel_loaded:
        # Wait for voxel FKT job completion so can read in voxel level filters
        if not check_job_completion(job_id):
            print(f"voxel FKT  job {job_id} did not complete successfully.")
            return
        print(f"voxel FKT  job {job_id} completed successfully.")
        voxel_filters = np.load(voxel_filters_path)

    # Run MIGP
    # Need 3 MIGPs: one for GICA (full data), one for parcel IFA (residualized on parcel discrim), one for voxel IFA (residualized on voxel discrim)
    m = np.max((mA,mB))
    migp_dir = os.path.join(fold_output_dir, "MIGP")
    os.makedirs(migp_dir, exist_ok=True)

    reduced_subs_path_GICA = os.path.join(migp_dir, "reducedsubs_GICA.npy")
    reducedsubs_GICA = MIGP_wrapper(train_paths, reduced_subs_path_GICA,m=m, n_jobs=20, batch_size=1, vt=None)
    
    # TODO Should this be transposed
    reduced_subs_path_PIFA = os.path.join(migp_dir, "reducedsubs_PIFA.npy")
    reducedsubs_PIFA = MIGP_wrapper(train_paths, reduced_subs_path_PIFA, m=m, n_jobs=20, batch_size=1, vt=np.asarray(parcelvoxel_filters.T, dtype=np.float64, order='C'))
    
    # TODO Should this be transposed
    reduced_subs_path_VIFA = os.path.join(migp_dir, "reducedsubs_VIFA.npy")
    reducedsubs_VIFA = MIGP_wrapper(train_paths, reduced_subs_path_VIFA, m=m, n_jobs=20, batch_size=1, vt=np.asarray(voxel_filters.T, dtype=np.float64, order='C'))

    for nPCA in nPCA_levels:
        nPCA = int(nPCA)
        nPCA_dir = os.path.join(fold_output_dir, f"nPCA_{nPCA}")
        nPCA_results = os.path.join(nPCA_dir, "Results")
        
        ICA_dir = os.path.join(nPCA_dir, "ICA")
        GICA_dir = os.path.join(ICA_dir, "GICA")
        parcel_IFA_dir = os.path.join(ICA_dir, "parcel_IFA")
        voxel_IFA_dir = os.path.join(ICA_dir, "voxel_IFA")

        for d in (nPCA_dir, nPCA_results, ICA_dir, GICA_dir, voxel_IFA_dir, parcel_IFA_dir):
            os.makedirs(d, exist_ok=True)

        if os.path.exists(os.path.join(GICA_dir, "spatial_maps.npy")):
            ICA_zmaps = np.load(os.path.join(GICA_dir, "spatial_maps.npy"))
        else:
            ICA_zmaps = PPCA_ICA(reducedsubs_GICA,basis=None, n_components=int(nPCA+2*n_filters_per_group), random_state=random_state, output_folder=GICA_dir)


        vt_path_PIFA = os.path.join(parcel_IFA_dir, "vt.npy")
        parcel_IFA_maps_path = os.path.join(parcel_IFA_dir, "spatial_maps.npy")
        if os.path.exists(vt_path_PIFA) and os.path.exists(parcel_IFA_maps_path):
            parcelvoxel_IFA_zmaps = np.load(parcel_IFA_maps_path)
        else:
            _, vt_PIFA = PPCA(reducedsubs_PIFA.copy(), threshold=0.0, niters=1, n=nPCA)
            np.save(vt_path_PIFA, vt_PIFA)
            # Calculate the overlap between retained major eigenspace and discriminant subspace
            major_recon_discrim(parcelvoxel_filters, vt_PIFA, parcel_IFA_dir)
            parcelvoxel_IFA_zmaps = PPCA_ICA(reducedsubs_PIFA,basis=np.vstack((vt_PIFA, parcelvoxel_filters.T)), n_components=None,random_state=random_state, output_folder=parcel_IFA_dir)
        
        
        vt_path_VIFA = os.path.join(voxel_IFA_dir, "vt.npy")
        voxel_IFA_zmaps_path = os.path.join(voxel_IFA_dir, "spatial_maps.npy")
        if os.path.exists(vt_path_VIFA) and os.path.exists(voxel_IFA_zmaps_path):
            voxel_IFA_zmaps = np.load(voxel_IFA_zmaps_path)
        else:
            _, vt_VIFA = PPCA(reducedsubs_VIFA.copy(), threshold=0.0, niters=1, n=nPCA)
            np.save(vt_path_VIFA, vt_VIFA)
            # Calculate the overlap between retained major eigenspace and discriminant subspace
            major_recon_discrim(voxel_filters, vt_VIFA, voxel_IFA_dir)
            voxel_IFA_zmaps = PPCA_ICA(reducedsubs_VIFA,basis=np.vstack((vt_VIFA, voxel_filters.T)), n_components=None,random_state=random_state, output_folder=voxel_IFA_dir)
        
        
        spatial_maps = [ICA_zmaps, parcelvoxel_IFA_zmaps, voxel_IFA_zmaps]
        outputfolders = [GICA_dir, parcel_IFA_dir, voxel_IFA_dir]

        # Expected outputs for each spatial map
        expected_dualreg_files = [os.path.join(outdir, fname) 
                                  for outdir in [GICA_dir, parcel_IFA_dir, voxel_IFA_dir] 
                                  for fname in ("A.pkl", "spatial_map.npy", "reconstruction_error.npy")
        ]

        if not all(os.path.exists(f) for f in expected_dualreg_files):
            print("[run] Dual Regression")
            sample = np.min((200,train_idx.shape[0]))
            dual_regressor = DualRegress(
                subs=paths,
                spatial_maps=spatial_maps,
                train_index=train_idx,
                train_labels=train_labels,
                outputfolders=outputfolders,
                workers=20,
                sample=sample,
                method="bayesian",
                parallel_points=1,
                parallel_subs=20,
                n_calls=20,
                random_state=random_state
            )

            dual_regressor.dual_regress()
            del dual_regressor
            gc.collect()
        else:
            print("[cache] Dual Regression")
  
        map_names = ["GICA","parcel_IFA","voxel_IFA"]

        results = []
        
        for i, map_i in enumerate(map_names):        
            nPCA_results_maps = os.path.join(nPCA_results, map_i)
            if not os.path.exists(nPCA_results_maps):
                os.makedirs(nPCA_results_maps)

            with open(os.path.join(outputfolders[i], "A.pkl"), "rb") as f:
                tmp_a = pickle.load(f)
            tmp_spatial_map = np.load(os.path.join(outputfolders[i], "spatial_map.npy"))
            tmp_recon = np.load(os.path.join(outputfolders[i], "reconstruction_error.npy"))
            # TODO change 2d lda to account for class imbalance, t-test cluster/tfce logic changes, decide on z scoring projection, 
            #           have plots check if cifti, change spatial_map classification to be hyperparam learned
            result_i = evaluate((tmp_a, tmp_spatial_map, tmp_recon), 
                                        labels, train_idx, test_idx, a_label, b_label,
                                        metric=metric, alpha=0.05, paired=paired, 
                                        permutations=10000, deconf=deconfound, 
                                        con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                                        con_confounder_test=test_con_confounders, cat_confounder_test=test_cat_confounders,
                                        output_dir=nPCA_results_maps, random_seed=random_state, basis=f"{map_i}", n_workers=20)           

            results.append(result_i)
            # Remove the temporary variables and force garbage collection.
            del tmp_a, tmp_spatial_map, tmp_recon
            gc.collect()

        # # Define the pairwise comparisons (same for both normalized and unnormalized)
        # pairs = [
        #     (0, 1, "GICA", "parcel_IFA"),
        #     (0, 2, "GICA", "voxel_IFA"),
        #     (1, 2, "parcel_IFA", "voxel_IFA")
        # ]
 
        # # Run for normalized results
        # compare_dir_norm = os.path.join(nPCA_results, "Compare", "Normalized")
        # run_comparisons(results, compare_dir_norm, pairs, alpha=0.05)

# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fold analysis for a given output folder and fold number.")
    parser.add_argument("outputfolder", type=str, help="Path to the output folder")
    parser.add_argument("fold", type=int, help="Fold number")

    args = parser.parse_args()
    run_fold(args.outputfolder, args.fold)