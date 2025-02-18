import sys
import os
import json
import numpy as np
import argparse
import subprocess
import time
import pickle
import hcp_utils as hcp

# Add the path to custom modules
sys.path.append('/project/3022057.01/IFA/utils')

# Import necessary modules
from analysis import evaluate, compare
from PCA import PPCA
from filters import whiten, orthonormalize_filters, save_brain
from ICA import ICA, threshold_and_visualize
from DualRegression import DualRegress
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

def major_recon_discrim(discrim_basis, major_space,output_folder):
    try:
        # Compute reconstruction
        reconstructed = discrim_basis.T @ np.linalg.pinv(major_space) @ major_space
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

def PPCA_ICA(reducedsubs,basis=None, n_components=None, IFA=True, self_whiten=False,random_state=42,whiten_method="InvCov", output_folder=None):
    if IFA:
        if self_whiten:
            ## Whiten Basis, will need to whiten because this is a combined basis; method chosen wil rotate it to change ICA unmixing starting position (ICA unmixing is nondeterministic)
            basis, _ = whiten(basis, n_components=basis.shape[0], method=whiten_method)
        
        # Variance Normalize Data (PPCA is only being used for variance normalizing data since we already have the basis)
        data_vn, _ = PPCA(reducedsubs.copy(), filters=basis.T, threshold=0.0, niters=1)
    else:
        # For group ICA need to use PPCA to get the major space to match the dimensionality from IFA
        data_vn, basis = PPCA(reducedsubs.copy(), threshold=0.0, niters=1, n=n_components)

        if self_whiten:
            ## Although basis is orthogonal, this rewhitening accounts for number of samples for whitening
            basis, _ = whiten(basis, n_components=basis.shape[0], method=whiten_method)

    spatial_maps, A, W = ICA(data_vn, basis, whiten=(not self_whiten), output_dir=output_folder,random_state=random_state)
    zmaps, zmaps_thresh = threshold_and_visualize(data_vn, W, spatial_maps.T, visualize=True,output_dir=output_folder)

    for i in range(zmaps.shape[1]):
        save_brain(zmaps[:,i], f"z_map_{i}", output_folder)

    np.save(os.path.join(output_folder, "basis.npy"), basis)
    np.save(os.path.join(output_folder, "data_vn.npy"), data_vn)
    np.save(os.path.join(output_folder, "spatial_maps.npy"), spatial_maps)
    np.save(os.path.join(output_folder, "A.npy"), A)
    np.save(os.path.join(output_folder, "W.npy"), W)
    np.save(os.path.join(output_folder, "ICA_zmaps.npy"), zmaps)
    np.save(os.path.join(output_folder, "ICA_zmaps_thresh.npy"), zmaps_thresh)

    return zmaps

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

def run_fold(outputfolder, fold):
    # Read the settings from the JSON file
    with open(os.path.join(outputfolder, "settings.json"), "r") as f:
        settings = json.load(f)
    random_state = settings["random_state"]
    n_filters_per_group = settings["n_filters_per_group"]
    nPCA_levels = settings["nPCA_levels"]
    tangent_class = settings["tangent_class"]
    tan_class_model = settings["tan_class_model"]
    metric = settings["metric"]
    a_label = settings["a_label"]
    b_label = settings["b_label"]
    self_whiten = settings["self_whiten"]
    deconfound = settings["deconfound"]
    paired = settings["paired"]

    # Load pickle files
    with open(os.path.join(outputfolder, "paths.pkl"), "rb") as f:
        paths = pickle.load(f)

    with open(os.path.join(outputfolder, "cat_confounders.pkl"), "rb") as f:
        cat_confounders = pickle.load(f)

    with open(os.path.join(outputfolder, "family_ID.pkl"), "rb") as f:
        family_ID = pickle.load(f)

    # Load numpy files
    sub_ID = np.load(os.path.join(outputfolder, "Sub_ID.npy"))
    labels = np.load(os.path.join(outputfolder, "labels.npy"))
    data = np.load(os.path.join(outputfolder, "data.npy"))
    covs = np.load(os.path.join(outputfolder, "covs.npy"))
    con_confounders = np.load(os.path.join(outputfolder, "con_confounders.npy"))

    # Load Fold Specific Vairables
    fold_output_dir = os.path.join(outputfolder, f"fold_{fold}")
    summary_file_path = os.path.join(fold_output_dir, "output_summary.txt")
    indices_dir = os.path.join(fold_output_dir, "Indices")
    train_idx = np.load(os.path.join(indices_dir, "train_idx.npy"))
    test_idx = np.load(os.path.join(indices_dir, "test_idx.npy"))
    fold_results = os.path.join(fold_output_dir, "Results")
    if not os.path.exists(fold_results):
        os.makedirs(fold_results)

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
    
    # Save summary of data split
    train_groups = set(np.unique(family_ID[train_idx]))
    test_groups = set(np.unique(family_ID[test_idx]))
    intersection = train_groups & test_groups
    save_text_results(f"Fold {fold + 1}:", summary_file_path)
    save_text_results(f"  Train size: {len(train_idx)}", summary_file_path)
    save_text_results(f"  Test size: {len(test_idx)}", summary_file_path)
    save_text_results(f"  Train labels distribution: {np.bincount(labels[train_idx].astype(int))}", summary_file_path)
    save_text_results(f"  Test labels distribution: {np.bincount(labels[test_idx].astype(int))}", summary_file_path)
    save_text_results(f"  Intersection of groups: {len(intersection)} (Groups: {intersection})", summary_file_path)

    # Run tangent classification for measuring separability in parcellated space
    tangent_class_metrics = tangent_classification(train_covs, train_labels, test_covs, test_labels, 
                           clf_str='all', z_score=0, metric=metric, deconf=deconfound, 
                           con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                           con_confounder_test=test_con_confounders, cat_confounder_test=test_cat_confounders)
    
    # Save those tangent classification results to overall fold results directory
    with open(os.path.join(fold_results, "tangent_class_metrics.pkl"), "wb") as f:
        pickle.dump(tangent_class_metrics, f)   
    save_text_results("Parcellated Tangent Classification " + str(tangent_class_metrics), summary_file_path)

    # Get Parcellated Filters
    filters_dir = os.path.join(fold_output_dir, "Filters")
    if not os.path.exists(filters_dir):
        os.makedirs(filters_dir)
    
    # Directory for all things related to the parecllated filters
    parcellated_filters_dir = os.path.join(filters_dir, "Parcellated")
    if not os.path.exists(parcellated_filters_dir):
        os.makedirs(parcellated_filters_dir)

    if tangent_class:
        eigs, filters_all, W, C = TSSF(train_covs, train_labels, 
                                       clf_str=tan_class_model, metric=metric, deconf=deconfound, 
                                       con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                                       z_score=0, haufe=False, visualize=True, output_dir=parcellated_filters_dir)
    else:
        eigs, filters_all = FKT(train_covs, train_labels, 
                                metric=metric, deconf=deconfound, 
                                con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                                visualize=True, output_dir=parcellated_filters_dir)
    
    filtersA = filters_all[:, -n_filters_per_group:]
    filtersB = filters_all[:, :n_filters_per_group]
    filters_parcellated = np.concatenate((filtersB, filtersA), axis=1)

    np.save(os.path.join(parcellated_filters_dir, "filtersA.npy"), filtersA)
    np.save(os.path.join(parcellated_filters_dir, "filtersB.npy"), filtersB)
    np.save(os.path.join(parcellated_filters_dir, "filters_parcellated.npy"), filters_parcellated)
    for i in range(filters_parcellated.shape[1]):
        save_brain(hcp.unparcellate(filters_parcellated[:,i],hcp.mmp), f"parcellated_filter_{i}", parcellated_filters_dir)

    # Evaluate filters and save those results to overall fold results directory
    logvar_stats, logcov_stats = evaluate_filters(train_data, train_labels, test_data, test_labels, 
                                                    filters_parcellated, metric=metric, deconf=deconfound, 
                                                    con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                                                    con_confounder_test=test_con_confounders, cat_confounder_test=test_cat_confounders,output_dir=parcellated_filters_dir)

    with open(os.path.join(fold_results, "logvar_stats.pkl"), "wb") as f:
            pickle.dump(logvar_stats, f)     
    with open(os.path.join(fold_results, "logcov_stats.pkl"), "wb") as f:
            pickle.dump(logcov_stats, f)      
    save_text_results("Log Var Filter Feature Classification " + str(logvar_stats), summary_file_path)
    save_text_results("Log Cov Filter Feature Classification " + str(logcov_stats), summary_file_path)
  

    # Run the PCA job, calculates MIGP to reduce subjects*time and uses MIGP to calculate voxel level filters
    # Directory for all things related to the parecllated filters
    voxel_filters_dir = os.path.join(filters_dir, "Voxel")
    if not os.path.exists(voxel_filters_dir):
        os.makedirs(voxel_filters_dir)

    pca_script = "/project/3022057.01/IFA/run_IFA/run_pca.sh"
    pca_command = [
        "sbatch",
        "--output", os.path.join(fold_output_dir, "pca-%j.out"),
        "--error", os.path.join(fold_output_dir, "pca-%j.err"),
        pca_script,
        outputfolder, fold_output_dir, voxel_filters_dir
    ]

    pca_process = subprocess.run(pca_command, capture_output=True, text=True)
    if pca_process.returncode != 0:
        print(f"Error submitting PCA job: {pca_process.stderr}")
        return
    job_id = pca_process.stdout.strip().split()[-1]
    print(f"PCA job submitted successfully with job ID: {job_id}")
    
    # While PCA Job Runs, run dual regression on parcellated filters
    filtersA_transform = filter_dual_regression(filtersA, train_data[train_labels == a_label], train_paths[train_labels == a_label],workers=20)
    filtersB_transform = filter_dual_regression(filtersB, train_data[train_labels == b_label], train_paths[train_labels == b_label],workers=20)
    np.save(os.path.join(parcellated_filters_dir, "A_filters_haufe.npy"), filtersA_transform)
    np.save(os.path.join(parcellated_filters_dir, "B_filters_haufe.npy"), filtersB_transform)
    parcelvoxel_filters = orthonormalize_filters(filtersA_transform, filtersB_transform)
    np.save(os.path.join(parcellated_filters_dir, "filters.npy"), parcelvoxel_filters)
    for i in range(parcelvoxel_filters.shape[1]):
        save_brain(parcelvoxel_filters[:,i], f"parcelvoxel_filters{i}", parcellated_filters_dir)


    # Wait for PCA job completion
    if not check_job_completion(job_id):
        print(f"PCA job {job_id} did not complete successfully.")
        return
    print(f"PCA job {job_id} completed successfully.")

    
    # after the job completes, load the relevant data
    voxel_filters = np.load(os.path.join(voxel_filters_dir, "filters.npy"))
    migp_dir = os.path.join(fold_output_dir, "MIGP")
    reducedsubs  = np.load(os.path.join(migp_dir, "reducedsubs.npy"))

    for nPCA in nPCA_levels:
        # Directory for resulst for basis spanned by nPCA components + fixed number of filters
        nPCA_dir = os.path.join(fold_output_dir, f"nPCA_{nPCA}")
        if not os.path.exists(nPCA_dir):
            os.makedirs(nPCA_dir)
        
        # Calculate the major eigenspace for the ICA basis
        _, vt = PPCA(reducedsubs.copy(), threshold=0.0, niters=1, n=nPCA)
        np.save(os.path.join(filters_dir, f"vt_{nPCA}.npy"), vt)
        
        # Calculate the overlap between retained major eigenspace and discriminant subspace
        major_recon_discrim(parcelvoxel_filters, vt, parcellated_filters_dir)
        major_recon_discrim(voxel_filters, vt, voxel_filters_dir)
        
        #Create Folders to store ICA outputs for this subspace dimension and run ICA for each basis
        ICA_dir = os.path.join(nPCA_dir, "ICA")
        if not os.path.exists(ICA_dir):
            os.makedirs(ICA_dir)
        
        parcel_IFA_dir = os.path.join(ICA_dir, "parcel_IFA")
        if not os.path.exists(parcel_IFA_dir):
            os.makedirs(parcel_IFA_dir)
        parcelvoxel_IFA_zmaps = PPCA_ICA(reducedsubs,basis=np.vstack((vt, parcelvoxel_filters.T)), n_components=None, IFA=True, self_whiten=self_whiten,random_state=random_state,whiten_method="InvCov", output_folder=parcel_IFA_dir)

        voxel_IFA_dir = os.path.join(ICA_dir, "voxel_IFA")
        if not os.path.exists(voxel_IFA_dir):
            os.makedirs(voxel_IFA_dir)
        voxel_IFA_zmaps = PPCA_ICA(reducedsubs,basis=np.vstack((vt, voxel_filters.T)), n_components=None, IFA=True, self_whiten=self_whiten,random_state=random_state,whiten_method="InvCov", output_folder=voxel_IFA_dir)

        GICA_dir = os.path.join(ICA_dir, "GICA")
        if not os.path.exists(GICA_dir):
            os.makedirs(GICA_dir)
        ICA_zmaps = PPCA_ICA(reducedsubs,basis=None, n_components=voxel_IFA_zmaps.shape[1], IFA=False, self_whiten=self_whiten,random_state=random_state,whiten_method="InvCov", output_folder=GICA_dir)


        #Perform dual regression for all ICA maps
        dual_dir = os.path.join(fold_output_dir, f"Dual_Regression_{nPCA}")
        if not os.path.exists(dual_dir):
            os.makedirs(dual_dir)

        spatial_maps = [ICA_zmaps, parcelvoxel_IFA_zmaps, voxel_IFA_zmaps]
        outputfolders = [GICA_dir, parcel_IFA_dir, voxel_IFA_dir]

        sample = np.min((20,train_idx.shape[0]))
        dual_regressor = DualRegress(
            subs=paths,
            spatial_maps=spatial_maps,
            train_index=train_idx,
            outputfolders=outputfolders,
            workers=20,
            sample=sample,
            method="bayesian",
            parallel_points=10,
            parallel_subs=2,
            n_calls=15,
            random_state=random_state
        )

        dual_regressor.dual_regress()

        # Analyze each set of spatial maps
        nPCA_results = os.path.join(nPCA_dir, "Results")
        if not os.path.exists(nPCA_results):
            os.makedirs(nPCA_results)

        map_names = ["GICA","parcel_IFA","voxel_IFA"]
        normalized_result = []
        unnormalized_result = []

        for i, map_i in enumerate(map_names):
            result = dual_regressor.dual_regression_results[i]
        
            nPCA_results_maps = os.path.join(nPCA_results, map_i)
            if not os.path.exists(nPCA_results_maps):
                os.makedirs(nPCA_results_maps)
            
            # For Normalized results
            nPCA_results_maps_norm = os.path.join(nPCA_results_maps, "Normalized")
            if not os.path.exists(nPCA_results_maps_norm):
                os.makedirs(nPCA_results_maps_norm)

            normalized_result_i = evaluate((result['normalized']['An'], result['normalized']['spatial_map'], result['reconstruction_error']), 
                                labels, train_idx, test_idx, 
                                metric=metric, alpha=0.05, paired=paired, 
                                permutations=10000, deconf=deconfound, 
                                con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                                con_confounder_test=test_con_confounders, cat_confounder_test=test_cat_confounders,
                                output_dir=nPCA_results_maps_norm, random_seed=random_state, basis=f"{map_i}_Normalized")           
            
            normalized_result.append(normalized_result_i)
            
            # For Unnormalized (demeaned) results
            nPCA_results_maps_unnorm = os.path.join(nPCA_results_maps, "Unnormalized")
            if not os.path.exists(nPCA_results_maps_unnorm):
                os.makedirs(nPCA_results_maps_unnorm)

            unnormalized_result_i = evaluate((result['demean']['Adm'], result['demean']['spatial_mapdm'], result['reconstruction_error']), 
                    labels, train_idx, test_idx, 
                    metric=metric, alpha=0.05, paired=paired, 
                    permutations=10000, deconf=deconfound, 
                    con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                    con_confounder_test=test_con_confounders, cat_confounder_test=test_cat_confounders,
                    output_dir=nPCA_results_maps_unnorm, random_seed=random_state, basis=f"{map_i}_Unnormalized")
            
            unnormalized_result.append(unnormalized_result_i)

        # Define the pairwise comparisons (same for both normalized and unnormalized)
        pairs = [
            (0, 1, "GICA", "parcel_IFA"),
            (0, 2, "GICA", "voxel_IFA"),
            (1, 2, "parcel_IFA", "voxel_IFA")
        ]

        # Run for normalized results
        compare_dir_norm = os.path.join(nPCA_results, "Compare", "Normalized")
        run_comparisons(normalized_result, compare_dir_norm, pairs, alpha=0.05)

        # Run for unnormalized results
        compare_dir_unnorm = os.path.join(nPCA_results, "Compare", "Unnormalized")
        run_comparisons(unnormalized_result, compare_dir_unnorm, pairs, alpha=0.05)
        

# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fold analysis for a given output folder and fold number.")
    parser.add_argument("outputfolder", type=str, help="Path to the output folder")
    parser.add_argument("fold", type=int, help="Fold number")

    args = parser.parse_args()
    run_fold(args.outputfolder, args.fold)