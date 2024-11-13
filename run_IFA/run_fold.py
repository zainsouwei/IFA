import sys
import os
import json
import numpy as np
import argparse
import subprocess
import time
import pickle


# Add the path to custom modules
sys.path.append('/project/3022057.01/IFA/utils')

# Import necessary modules
from analysis import evaluate_IFA_results
from PCA import PPCA
from filters import whiten, orthonormalize_filters
from ICA import ICA, threshold_and_visualize
from DualRegression import dual_regress
from filters import TSSF, FKT, evaluate_filters
from tangent import tangent_classification
from haufe import filter_dual_regression

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

def run_fold(outputfolder, fold):
    # Load variables
    labels = np.load(os.path.join(outputfolder,"labels.npy"))
    data = np.load(os.path.join(outputfolder,"data.npy"))
    covs = np.load(os.path.join(outputfolder,"covs.npy"))
    with open(os.path.join(outputfolder, "paths.pkl"), "rb") as f:
        paths = pickle.load(f)    
    con_confounders = np.load(os.path.join(outputfolder,"con_confounders.npy"))
    with open(os.path.join(outputfolder, "cat_confounders.pkl"), "rb") as f:
        cat_confounders = pickle.load(f)    
    with open(os.path.join(outputfolder, "family_ID.pkl"), "rb") as f:
        family_ID = pickle.load(f)
    
    # Load settings
    settings_filepath = os.path.join(outputfolder, "settings.json")
    # Load the settings from the JSON file
    with open(settings_filepath, "r") as f:
        settings = json.load(f)
    # Access each setting from the dictionary
    random_state = settings["random_state"]
    n_filters_per_group = settings["n_filters_per_group"]
    nPCA = settings["nPCA"]
    Tangent_Class = settings["Tangent_Class"]
    metric = settings["metric"]
    a_label = int(settings["a_label"])
    b_label = int(settings["b_label"])
    self_whiten = settings["self_whiten"]
    deconfound = settings["deconfound"]


    # Load Fold Specific Vairables
    fold_output_dir = os.path.join(outputfolder, f"fold_{fold}")
    summary_file_path = os.path.join(fold_output_dir, "output_summary.txt")
    indices_dir = os.path.join(fold_output_dir, "Indices")
    train_idx = np.load(os.path.join(indices_dir, "train_idx.npy"))
    test_idx = np.load(os.path.join(indices_dir, "test_idx.npy"))
    
    # Prepare data for train and test sets
    train_labels = labels[train_idx]
    train_data = data[train_idx]
    train_covs = covs[train_idx]
    train_paths = paths[train_idx]
    train_con_confounders = con_confounders[train_idx]
    train_cat_confounders = cat_confounders[train_idx]

    test_labels = labels[test_idx]
    test_data = data[test_idx]
    test_covs = covs[test_idx]
    test_con_confounders = con_confounders[test_idx]
    test_cat_confounders = cat_confounders[test_idx]

    
    # Fold Analysis Results Folder
    fold_results = os.path.join(fold_output_dir, "Results")
    if not os.path.exists(fold_results):
        os.makedirs(fold_results)

    # Train/test group validation
    train_groups = set(np.unique(family_ID[train_idx]))
    test_groups = set(np.unique(family_ID[test_idx]))
    intersection = train_groups & test_groups

    # Save summary of data split
    save_text_results(f"Fold {fold + 1}:", summary_file_path)
    save_text_results(f"  Train size: {len(train_idx)}", summary_file_path)
    save_text_results(f"  Test size: {len(test_idx)}", summary_file_path)
    save_text_results(f"  Train labels distribution: {np.bincount(labels[train_idx].astype(int))}", summary_file_path)
    save_text_results(f"  Test labels distribution: {np.bincount(labels[test_idx].astype(int))}", summary_file_path)
    save_text_results(f"  Intersection of groups: {len(intersection)} (Groups: {intersection})", summary_file_path)


    # Run tangent classification
    tangent_class_metrics = tangent_classification(
        train_covs, train_labels, test_covs, test_labels, clf_str='all', z_score=0, metric=metric, deconf=False
    )
    with open(os.path.join(fold_results, "tangent_class_metrics.pkl"), "wb") as f:
        pickle.dump(tangent_class_metrics, f)    
    save_text_results("Parcellated Tangent Classification " + str(tangent_class_metrics), summary_file_path)

    # Deconfounded tangent classification
    tangent_class_metrics_deconf = tangent_classification(
        train_covs, train_labels, test_covs, test_labels, clf_str='all', z_score=0, metric=metric, deconf=True,
        con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders,
        con_confounder_test=test_con_confounders, cat_confounder_test=test_cat_confounders
    )
    with open(os.path.join(fold_results, "tangent_class_metrics_deconf.pkl"), "wb") as f:
            pickle.dump(tangent_class_metrics_deconf, f)    
    save_text_results("Parcellated Tangent Classification Deconfounded " + str(tangent_class_metrics_deconf), summary_file_path)


    if Tangent_Class:
        eigs, filters_all = TSSF(train_covs, train_labels, clf_str='SVM (C=0.1)', metric=metric, deconf=deconfound, con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, z_score=0, haufe=False, visualize=True, output_dir=fold_results)
    else:
        eigs, filters_all = FKT(train_covs, train_labels, metric=metric, deconf=deconfound, con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, visualize=True, output_dir=fold_results)

    filtersA = filters_all[:, -n_filters_per_group:]
    filtersB = filters_all[:, :n_filters_per_group]
    filters_parcellated = np.concatenate((filtersB, filtersA), axis=1)

    filters_dir = os.path.join(fold_output_dir, "Filters")
    if not os.path.exists(filters_dir):
        os.makedirs(filters_dir)
    np.save(os.path.join(filters_dir, "filtersA.npy"), filtersA)
    np.save(os.path.join(filters_dir, "filtersB.npy"), filtersB)
    np.save(os.path.join(filters_dir, "filters_parcellated.npy"), filters_parcellated)

    logvar_stats, logcov_stats = evaluate_filters(train_data, train_labels, test_data, test_labels, filters_parcellated, metric=metric, deconf=False, output_dir=fold_results)
    with open(os.path.join(fold_results, "logvar_stats.pkl"), "wb") as f:
            pickle.dump(logvar_stats, f)     
    with open(os.path.join(fold_results, "logcov_stats.pkl"), "wb") as f:
            pickle.dump(logcov_stats, f)      
    save_text_results("Log Var Filter Feature Classification " + str(logvar_stats), summary_file_path)
    save_text_results("Log Cov Filter Feature Classification " + str(logcov_stats), summary_file_path)

    logvar_stats_deconf, logcov_stats_deconf = evaluate_filters(train_data, train_labels, test_data, test_labels, filters_parcellated, metric=metric, deconf=True, con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, con_confounder_test=test_con_confounders, cat_confounder_test=test_cat_confounders)
    with open(os.path.join(fold_results, "logvar_stats_deconf.pkl"), "wb") as f:
            pickle.dump(logvar_stats_deconf, f)     
    with open(os.path.join(fold_results, "logcov_stats_deconf.pkl"), "wb") as f:
            pickle.dump(logcov_stats_deconf, f) 
    save_text_results("Log Var Filter Feature Classification Deconfounded " + str(logvar_stats_deconf), summary_file_path)
    save_text_results("Log Cov Filter Feature Classification Deconfounded" + str(logcov_stats_deconf), summary_file_path)
    
    # Submit Separate Job for PCA Related Stuff which uses GPU
    # Submit the PCA job using sbatch
    pca_script = "/project/3022057.01/IFA/run_IFA/run_pca.sh"
    pca_command = [
        "sbatch",
        "--output", os.path.join(fold_output_dir, "pca-%j.out"),
        "--error", os.path.join(fold_output_dir, "pca-%j.err"),
        pca_script,
        outputfolder, fold_output_dir, str(nPCA)
    ]

    # Run the PCA job and wait for it to complete
    pca_process = subprocess.run(pca_command, capture_output=True, text=True)
    if pca_process.returncode != 0:
        print(f"Error submitting PCA job: {pca_process.stderr}")
        return
    job_id = pca_process.stdout.strip().split()[-1]
    print(f"PCA job submitted successfully with job ID: {job_id}")

    filtersA_transform = filter_dual_regression(filtersA, train_data[train_labels == a_label], train_paths[train_labels == a_label])
    filtersB_transform = filter_dual_regression(filtersB, train_data[train_labels == b_label], train_paths[train_labels == b_label])
    np.save(os.path.join(filters_dir, "filtersA_transform.npy"), filtersA_transform)
    np.save(os.path.join(filters_dir, "filtersB_transform.npy"), filtersB_transform)
    haufe_filters_ortho = orthonormalize_filters(filtersA_transform, filtersB_transform)
    np.save(os.path.join(filters_dir, "haufe_filters_ortho.npy"), haufe_filters_ortho)
    
    # Wait for PCA job completion
    if not check_job_completion(job_id):
        print(f"PCA job {job_id} did not complete successfully.")
        return
    print(f"PCA job {job_id} completed successfully.")
    ## Load relevant results from PCA job 
    migp_dir = os.path.join(fold_output_dir, "MIGP")
    reducedsubs = np.load(os.path.join(migp_dir, "reducedsubs.npy"))
    vt = np.load(os.path.join(migp_dir, "vt.npy"))


    ica_dir = os.path.join(fold_output_dir, "ICA")
    if not os.path.exists(ica_dir):
        os.makedirs(ica_dir)
    #  Group IFA
    ifa_dir = os.path.join(ica_dir, "IFA")
    if not os.path.exists(ifa_dir):
        os.makedirs(ifa_dir)
    IFA_basis = np.vstack((vt, haufe_filters_ortho.T))
    if self_whiten:
        ## Whiten Basis, will need to whiten because this is a combined basis; method chosen wil rotate it to change ICA unmixing starting position (ICA unmixing is nondeterministic)
        whitened_IFA_basis, _ = whiten(IFA_basis, n_components=IFA_basis.shape[0], method="InvCov")
    else:
        # Not actually whitened but having Fastica default whiten whiten the basis
        whitened_IFA_basis = IFA_basis
    ## Variance Normalize Subjects Based on Combined Basis      
    subs_data_IFA_VN, _ = PPCA(reducedsubs.copy(), filters=whitened_IFA_basis.T, threshold=0.0, niters=1)
    IFA_spatial_maps, IFA_A, IFA_W = ICA(subs_data_IFA_VN, whitened_IFA_basis,whiten=(not self_whiten), output_dir=ifa_dir,random_state=random_state)
    np.save(os.path.join(ifa_dir, "IFA_basis.npy"), IFA_basis)
    np.save(os.path.join(ifa_dir, "whitened_IFA_basis.npy"), whitened_IFA_basis)
    np.save(os.path.join(ifa_dir, "subs_data_IFA_VN.npy"), subs_data_IFA_VN)
    np.save(os.path.join(ifa_dir, "IFA_spatial_maps.npy"), IFA_spatial_maps)
    np.save(os.path.join(ifa_dir, "IFA_A.npy"), IFA_A)
    np.save(os.path.join(ifa_dir, "IFA_W.npy"), IFA_W)

    IFA_zmaps, IFA_zmaps_thresh = threshold_and_visualize(subs_data_IFA_VN, IFA_W, IFA_spatial_maps.T, visualize=True,output_dir=ifa_dir)
    np.save(os.path.join(ifa_dir, "IFA_zmaps.npy"), IFA_zmaps)
    np.save(os.path.join(ifa_dir, "IFA_zmaps_thresh.npy"), IFA_zmaps_thresh)

    #  Group ICA
    gica_dir = os.path.join(ica_dir, "GICA")
    if not os.path.exists(gica_dir):
        os.makedirs(gica_dir)
    ## Match Dimensionality of IFA and Variance Normalize based on this basis
    subs_data_ICA_VN, ICA_basis = PPCA(reducedsubs.copy(), threshold=0.0, niters=1, n=IFA_basis.shape[0])
    if self_whiten:
        ## Although basis is orthogonal, this rewhitening accounts for number of samples for whitening
        whitened_ICA_basis, _ = whiten(ICA_basis, n_components=ICA_basis.shape[0], method="SVD")
    else:
        # Not actually whitened but having Fastica default whiten whiten the basis
        whitened_ICA_basis = ICA_basis

    ICA_spatial_maps, ICA_A, ICA_W = ICA(subs_data_ICA_VN, whitened_ICA_basis, whiten=(not self_whiten), output_dir=gica_dir,random_state=random_state)
    np.save(os.path.join(gica_dir, "ICA_basis.npy"), ICA_basis)
    np.save(os.path.join(gica_dir, "whitened_ICA_basis.npy"), whitened_ICA_basis)
    np.save(os.path.join(gica_dir, "subs_data_ICA_VN.npy"), subs_data_ICA_VN)
    np.save(os.path.join(gica_dir, "ICA_spatial_maps.npy"), ICA_spatial_maps)
    np.save(os.path.join(gica_dir, "ICA_A.npy"), ICA_A)
    np.save(os.path.join(gica_dir, "ICA_W.npy"), ICA_W)
        
    ICA_zmaps, ICA_zmaps_thresh = threshold_and_visualize(subs_data_ICA_VN, ICA_W, ICA_spatial_maps.T, visualize=True,output_dir=gica_dir)
    np.save(os.path.join(gica_dir, "ICA_zmaps.npy"), ICA_zmaps)
    np.save(os.path.join(gica_dir, "ICA_zmaps_thresh.npy"), ICA_zmaps_thresh)
    
    # Dual Regression for both basis so only need to load subject into memory once
    dual_dir = os.path.join(fold_output_dir, "Dual_Regression")
    if not os.path.exists(dual_dir):
        os.makedirs(dual_dir)
    # Unpack results from dual_regress
    ((IFA_An_subs, IFA_netmats, IFA_spatial_maps_subs, IFA_reconstruction_errors),
    (IFA_Adm_subs, IFA_netmatdm, IFA_spatial_mapdm_subs)), \
    ((ICA_An_subs, ICA_netmats, ICA_spatial_maps_subs, ICA_reconstruction_errors),
    (ICA_Adm_subs, ICA_netmatdm, ICA_spatial_mapdm_subs)) = dual_regress(paths, IFA_zmaps, ICA_zmaps)

    # Save normalized results for IFA
    np.save(os.path.join(dual_dir, "IFA_An.npy"), IFA_An_subs)
    np.save(os.path.join(dual_dir, "IFA_netmats.npy"), IFA_netmats)
    np.save(os.path.join(dual_dir, "IFA_spatial_maps.npy"), IFA_spatial_maps_subs)
    np.save(os.path.join(dual_dir, "IFA_reconstruction_errors.npy"), IFA_reconstruction_errors)

    # Save demeaned results for IFA
    np.save(os.path.join(dual_dir, "IFA_Adm.npy"), IFA_Adm_subs)
    np.save(os.path.join(dual_dir, "IFA_netmatdm.npy"), IFA_netmatdm)
    np.save(os.path.join(dual_dir, "IFA_spatial_mapdm.npy"), IFA_spatial_mapdm_subs)

    # Save normalized results for ICA
    np.save(os.path.join(dual_dir, "ICA_An.npy"), ICA_An_subs)
    np.save(os.path.join(dual_dir, "ICA_netmats.npy"), ICA_netmats)
    np.save(os.path.join(dual_dir, "ICA_spatial_maps.npy"), ICA_spatial_maps_subs)
    np.save(os.path.join(dual_dir, "ICA_reconstruction_errors.npy"), ICA_reconstruction_errors)

    # Save demeaned results for ICA
    np.save(os.path.join(dual_dir, "ICA_Adm.npy"), ICA_Adm_subs)
    np.save(os.path.join(dual_dir, "ICA_netmatdm.npy"), ICA_netmatdm)
    np.save(os.path.join(dual_dir, "ICA_spatial_mapdm.npy"), ICA_spatial_mapdm_subs)

    # Prepare data for evaluation using normalized data
    IFA_eval_data_normalized = (
        IFA_An_subs[train_idx], IFA_netmats[train_idx], IFA_reconstruction_errors[train_idx],
        IFA_An_subs[test_idx], IFA_netmats[test_idx], IFA_reconstruction_errors[test_idx]
    )

    ICA_eval_data_normalized = (
        ICA_An_subs[train_idx], ICA_netmats[train_idx], ICA_reconstruction_errors[train_idx],
        ICA_An_subs[test_idx], ICA_netmats[test_idx], ICA_reconstruction_errors[test_idx]
    )

    # Prepare data for evaluation using demeaned data (reuse reconstruction errors from normalized data)
    IFA_eval_data_demeaned = (
        IFA_Adm_subs[train_idx], IFA_netmatdm[train_idx], IFA_reconstruction_errors[train_idx],
        IFA_Adm_subs[test_idx], IFA_netmatdm[test_idx], IFA_reconstruction_errors[test_idx]
    )

    ICA_eval_data_demeaned = (
        ICA_Adm_subs[train_idx], ICA_netmatdm[train_idx], ICA_reconstruction_errors[train_idx],
        ICA_Adm_subs[test_idx], ICA_netmatdm[test_idx], ICA_reconstruction_errors[test_idx]
    )

    fold_results_normalized = os.path.join(fold_results, f"Normalized")
    if not os.path.exists(fold_results_normalized):
        os.makedirs(fold_results_normalized)
    
    # Evaluate results using normalized data
    IFA_results_normalized, ICA_results_normalized = evaluate_IFA_results(
        IFA_eval_data_normalized, ICA_eval_data_normalized,
        train_labels, test_labels, alpha=0.05,
        permutations=False, correction='fdr_bh', metric=metric,
        deconf=deconfound, con_confounder_train=train_con_confounders,
        cat_confounder_train=train_cat_confounders, con_confounder_test=test_con_confounders,
        cat_confounder_test=test_cat_confounders, output_dir=fold_results_normalized
    )

    # Save evaluation results
    with open(os.path.join(fold_results_normalized, "IFA_results_normalized.pkl"), "wb") as f:
        pickle.dump(IFA_results_normalized, f)

    with open(os.path.join(fold_results_normalized, "ICA_results_normalized.pkl"), "wb") as f:
        pickle.dump(ICA_results_normalized, f)

    fold_results_demeaned = os.path.join(fold_results, f"Demeaned")
    if not os.path.exists(fold_results_demeaned):
        os.makedirs(fold_results_demeaned)

    # Evaluate results using demeaned data
    IFA_results_demeaned, ICA_results_demeaned = evaluate_IFA_results(
        IFA_eval_data_demeaned, ICA_eval_data_demeaned,
        train_labels, test_labels, alpha=0.05,
        permutations=False, correction='fdr_bh', metric=metric,
        deconf=deconfound, con_confounder_train=train_con_confounders,
        cat_confounder_train=train_cat_confounders, con_confounder_test=test_con_confounders,
        cat_confounder_test=test_cat_confounders, output_dir=fold_results_demeaned
    )

    with open(os.path.join(fold_results_demeaned, "IFA_results_demeaned.pkl"), "wb") as f:
        pickle.dump(IFA_results_demeaned, f)

    with open(os.path.join(fold_results_demeaned, "ICA_results_demeaned.pkl"), "wb") as f:
        pickle.dump(ICA_results_demeaned, f)


# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fold analysis for a given output folder and fold number.")
    parser.add_argument("outputfolder", type=str, help="Path to the output folder")
    parser.add_argument("fold", type=int, help="Fold number")

    args = parser.parse_args()
    run_fold(args.outputfolder, args.fold)