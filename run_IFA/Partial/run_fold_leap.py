import sys
import os
import json
import numpy as np
import argparse
import pickle
from pyriemann.estimation import Covariances
import traceback
from concurrent.futures import ProcessPoolExecutor
import functools
import gc
import pandas as pd
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_basc_multiscale_2015
basc = fetch_atlas_basc_multiscale_2015(resolution=444)
basc_labels = basc.maps

# Add the path to custom modules
sys.path.append('/project/3022057.01/IFA/utils')

# Import necessary modules
from analysis import evaluate, compare
from PCA import PPCA, migp
from filters import orthonormalize_filters, save_brain
from ICA import ICA
from DualRegression import DualRegress
from filters import TSSF, FKT, evaluate_filters
from tangent import tangent_classification
from haufe import partial_filter_dual_regression
from preprocessing import load_subject


# End of save block
from filters import voxelwise_FKT


def run_pca(outputfolder, voxel_filters_dir, train_paths, train_labels, a_label, b_label, mA, mB, batch_size=5):
    try:
        with open(os.path.join(outputfolder, "settings.json"), "r") as f:
            settings = json.load(f)
        

        n_filters_per_group = settings["n_filters_per_group"]
        cov_log = settings["cov_log"]
        shrink = settings["shrinkage"]
        filters_dir = os.path.dirname(voxel_filters_dir)
        vt = np.load(os.path.join(filters_dir, "vt.npy"))

        A_partial = migp(train_paths[train_labels == a_label], m=mA, n_jobs=15,batch_size=batch_size,vt=vt)
        A_partial = A_partial.astype(np.float32) 
        np.save(os.path.join(voxel_filters_dir, "A_partial.npy"), A_partial)

        B_partial = migp(train_paths[train_labels == b_label], m=mB, n_jobs=15,batch_size=batch_size,vt=vt)
        B_partial = B_partial.astype(np.float32)
        np.save(os.path.join(voxel_filters_dir, "B_partial.npy"), B_partial)

        voxelwise_FKT(groupA=A_partial, groupB=B_partial, 
                        n_filters_per_group=n_filters_per_group, 
                        groupA_paths=None, groupB_paths=None, 
                        paths=False,log=cov_log,shrinkage=shrink,
                        cov_method='svd',outputfolder=voxel_filters_dir, save=False, save_img=False)

    except Exception as e:
        print(f"Error in run_pca: {e}", flush=True)
        import traceback
        traceback.print_exc()
        # Exit with non-zero code so SLURM knows the job failed
        sys.exit(1)


def save_text_results(text, filepath):
    """Save text results to a file."""
    with open(filepath, "a") as f:  # Using 'a' to append results to the file
        f.write(text + "\n")


def partiallate_subject(sub, vt):
    try:
        # Load subject data using our load_subject function.
        data = load_subject(sub)
        # Remove the projection onto vt.
        partialled_subject = data - (data @ np.linalg.pinv(vt)) @ vt
        del data  # free memory


        _, gm_mask_file = sub
        masker = NiftiMasker(mask_img=gm_mask_file, dtype=np.float32)
        masker.fit()

        partial_full = masker.inverse_transform(partialled_subject)
        del partialled_subject

        labels_masker = NiftiLabelsMasker(labels_img=basc_labels, mask_img=gm_mask_file, standardize='zscore_sample',  dtype=np.float32)
        Xp = labels_masker.fit_transform(partial_full)
        del partial_full

        return Xp
    
    except Exception as e:
        print(f"Error processing subject {sub}: {e}")
        traceback.print_exc()
        raise

def partiallate_subjects(paths, vt, output_dir, n_workers=20):
    try:
        # Create a function that always uses the same vt.
        func = functools.partial(partiallate_subject, vt=vt)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Map over the list of subject paths. Results will be in the same order.
            partiallated_data_list = list(executor.map(func, paths))
        
        # Convert the list to a NumPy array.
        data_save_path = os.path.join(output_dir, "partiallated_data.pkl")
        with open(data_save_path, "wb") as f:
            pickle.dump(partiallated_data_list, f)
        print(f"Partiallated data saved to {data_save_path}")
        
        # Compute covariances one subject at a time
        cov_est = Covariances(estimator='oas')
        partial_covs = []
        for subject_data in partiallated_data_list:
            # subject_data shape: (T, Parcels)
            cov = cov_est.transform(subject_data.T[np.newaxis, :, :])  # (1, Parcels, Parcels)
            partial_covs.append(cov[0])  # cov_est.transform returns a batch (n_matrices, n_channels, n_channels)
        
        partial_covs = np.array(partial_covs)
        covs_save_path = os.path.join(output_dir, "partiallated_covs.npy")
        np.save(covs_save_path, partial_covs)
        print(f"Covariances saved to {covs_save_path}")
        
        return partiallated_data_list, partial_covs
    except Exception as e:
        print(f"Error in parcellation process: {e}")
        traceback.print_exc()
        raise  # Re-raise the error so the process crashes.

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

def PPCA_ICA(reducedsubs,basis=None, n_components=None, IFA=True,random_state=42, output_folder=None):
    if not IFA:
        _, basis = PPCA(reducedsubs.copy(), threshold=0.0, niters=1, n=n_components)

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

def run_fold(outputfolder, fold):
    # Read the settings from the JSON file
    with open(os.path.join(outputfolder, "settings.json"), "r") as f:
        settings = json.load(f)

    random_state           = settings["random_state"]
    n_filters_per_group    = settings["n_filters_per_group"]
    nPCA_levels            = settings["nPCA_levels"]
    tangent_class          = settings["tangent_class"]
    tan_class_model        = settings["tan_class_model"]
    metric                 = settings["metric"]
    a_label                = settings["a_label"]
    b_label                = settings["b_label"]
    deconfound             = settings["deconfound"]
    paired                 = settings["paired"]

    # Load numpy files
    sub_ID = np.load(os.path.join(outputfolder, "Sub_ID.npy"))
    labels = np.load(os.path.join(outputfolder, "labels.npy"))
    time_points = np.load(os.path.join(outputfolder, "time_points.npy"))

    # Load Fold Specific Vairables
    fold_output_dir = os.path.join(outputfolder, f"fold_{fold}")

    with open(os.path.join(fold_output_dir, "paths.pkl"), "rb") as f:
        paths = pickle.load(f)

    summary_file_path = os.path.join(fold_output_dir, "output_summary.txt")
    indices_dir = os.path.join(fold_output_dir, "Indices")
    train_idx = np.load(os.path.join(indices_dir, "train_idx.npy"))
    test_idx = np.load(os.path.join(indices_dir, "test_idx.npy"))
    fold_results = os.path.join(fold_output_dir, "Results")
    if not os.path.exists(fold_results):
        os.makedirs(fold_results)

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
    save_text_results(f"Fold {fold + 1}:", summary_file_path)
    save_text_results(f"  Train size: {len(train_idx)}", summary_file_path)
    save_text_results(f"  Test size: {len(test_idx)}", summary_file_path)
    save_text_results(f"  Train labels distribution: {np.bincount(labels[train_idx].astype(int))}", summary_file_path)
    save_text_results(f"  Test labels distribution: {np.bincount(labels[test_idx].astype(int))}", summary_file_path)
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
        
    # Run MIGP
    migp_dir = os.path.join(fold_output_dir, "MIGP")
    if not os.path.exists(migp_dir):
        os.makedirs(migp_dir)
    
    reduced_subs_path = os.path.join(migp_dir, "reducedsubs.npy")
    mA = 2 * time_points[train_idx][train_labels == a_label].max()
    mB = 2 * time_points[train_idx][train_labels == b_label].max()
    if os.path.exists(reduced_subs_path):
        print(f"Loading existing reducedsubs from {reduced_subs_path}")
        reducedsubs = np.load(reduced_subs_path)
    else:
        print("Reducedsubs not found — running MIGP")
        reducedsubsA = migp(train_paths[train_labels == a_label], m=mA, n_jobs=15,batch_size=1)
        np.save(os.path.join(migp_dir, "reducedsubsA.npy"), reducedsubsA)

        reducedsubsB = migp(train_paths[train_labels == b_label], m=mB, n_jobs=15,batch_size=1)
        np.save(os.path.join(migp_dir, "reducedsubsB.npy"), reducedsubsB)

        reducedsubs = np.concatenate((reducedsubsA, reducedsubsB), axis=0)
        np.save(reduced_subs_path, reducedsubs)
        del reducedsubsA, reducedsubsB
        gc.collect()

    for nPCA in nPCA_levels:
        # Directory for resulst for basis spanned by nPCA components + fixed number of filters
        nPCA_dir = os.path.join(fold_output_dir, f"nPCA_{nPCA}")
        if not os.path.exists(nPCA_dir):
            os.makedirs(nPCA_dir)
        
        # Get Parcellated Filters
        filters_dir = os.path.join(nPCA_dir, "Filters")
        if not os.path.exists(filters_dir):
            os.makedirs(filters_dir)

        vt_path = os.path.join(filters_dir, "vt.npy")
        if os.path.exists(vt_path):
            print("Loading existing vt.npy...")
            vt = np.load(vt_path).astype(np.float32)
        else:
            print("vt.npy not found, running PPCA...")
            _, vt = PPCA(reducedsubs.copy(), threshold=0.0, niters=1, n=nPCA)
            np.save(vt_path, vt)
        
        # Run the PCA job, now just to get the voxel level filters using GPU
        voxel_filters_dir = os.path.join(filters_dir, "Voxel")
        if not os.path.exists(voxel_filters_dir):
            os.makedirs(voxel_filters_dir)

        # Now call run_pca directly
        run_pca(outputfolder, voxel_filters_dir, train_paths, train_labels, a_label, b_label, mA, mB, batch_size=1)

        # Need to partial the data before parcellating; partial then parcellate each subject
        partial_data, partial_covs = partiallate_subjects(paths, vt, output_dir=filters_dir, n_workers=15)

        # Then split into train and test using your indices:
        partial_train_data = [partial_data[i] for i in train_idx]
        partial_test_data  = [partial_data[i] for i in test_idx]


        partial_train_covs = partial_covs[train_idx]
        partial_test_covs  = partial_covs[test_idx]


        # Run tangent classification for measuring separability in parcellated space
        tangent_class_metrics = tangent_classification(partial_train_covs, train_labels, partial_test_covs, test_labels, 
                            clf_str='all', z_score=0, metric=metric, deconf=deconfound, 
                            con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                            con_confounder_test=test_con_confounders, cat_confounder_test=test_cat_confounders)
        
        # Save those tangent classification results to overall fold results directory
        with open(os.path.join(filters_dir, "tangent_class_metrics.pkl"), "wb") as f:
            pickle.dump(tangent_class_metrics, f)   
        save_text_results("Parcellated Tangent Classification " + str(tangent_class_metrics), summary_file_path)

        
        # Directory for all things related to the parecllated filters
        parcellated_filters_dir = os.path.join(filters_dir, "Parcellated")
        if not os.path.exists(parcellated_filters_dir):
            os.makedirs(parcellated_filters_dir)

        if tangent_class:
            eigs, filters_all, W, C = TSSF(partial_train_covs, train_labels, 
                                        clf_str=tan_class_model, metric=metric, deconf=deconfound, 
                                        con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                                        z_score=0, haufe=False, visualize=True, output_dir=parcellated_filters_dir)
        else:
            eigs, filters_all = FKT(partial_train_covs, train_labels, a_label, b_label,
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

        np.save(os.path.join(parcellated_filters_dir, "filtersA.npy"), filtersA)
        np.save(os.path.join(parcellated_filters_dir, "filtersB.npy"), filtersB)
        np.save(os.path.join(parcellated_filters_dir, "filters_parcellated.npy"), filters_parcellated)
        # for i in range(filters_parcellated.shape[1]):
        #     save_brain(hcp.unparcellate(filters_parcellated[:,i],hcp.mmp), f"parcellated_filter_{i}", parcellated_filters_dir)

        # Evaluate filters and save those results to overall fold results directory
        logvar_stats, logcov_stats = evaluate_filters(partial_train_data, train_labels, partial_test_data, test_labels, 
                                                        filters_parcellated, metric=metric, deconf=deconfound, 
                                                        con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                                                        con_confounder_test=test_con_confounders, cat_confounder_test=test_cat_confounders,output_dir=parcellated_filters_dir)

        with open(os.path.join(filters_dir, "logvar_stats.pkl"), "wb") as f:
                pickle.dump(logvar_stats, f)     
        with open(os.path.join(filters_dir, "logcov_stats.pkl"), "wb") as f:
                pickle.dump(logcov_stats, f)      
        save_text_results("Log Var Filter Feature Classification " + str(logvar_stats), summary_file_path)
        save_text_results("Log Cov Filter Feature Classification " + str(logcov_stats), summary_file_path)
        
        # Get indices where label == a_label
        idx_a_label = np.where(train_labels == a_label)[0]
        idx_b_label = np.where(train_labels == b_label)[0]

        # Use list comprehension to select subjects
        partial_train_data_a = [partial_train_data[i] for i in idx_a_label]
        partial_train_data_b = [partial_train_data[i] for i in idx_b_label]

        # Then call
        filtersA_transform = partial_filter_dual_regression(filtersA, partial_train_data_a, train_paths[idx_a_label], vt, workers=15)
        filtersB_transform = partial_filter_dual_regression(filtersB, partial_train_data_b, train_paths[idx_b_label], vt, workers=15)


        np.save(os.path.join(parcellated_filters_dir, "A_filters_haufe.npy"), filtersA_transform)
        np.save(os.path.join(parcellated_filters_dir, "B_filters_haufe.npy"), filtersB_transform)
        parcelvoxel_filters = orthonormalize_filters(filtersA_transform, filtersB_transform)
        np.save(os.path.join(parcellated_filters_dir, "filters.npy"), parcelvoxel_filters)
        # for i in range(parcelvoxel_filters.shape[1]):
        #     save_brain(parcelvoxel_filters[:,i], f"parcelvoxel_filters{i}", parcellated_filters_dir)

        
        # after the job completes, load the relevant data
        voxel_filters = np.load(os.path.join(voxel_filters_dir, "filters.npy"))

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
        parcelvoxel_IFA_zmaps = PPCA_ICA(reducedsubs,basis=np.vstack((vt, parcelvoxel_filters.T)), n_components=None, IFA=True,random_state=random_state, output_folder=parcel_IFA_dir)

        voxel_IFA_dir = os.path.join(ICA_dir, "voxel_IFA")
        if not os.path.exists(voxel_IFA_dir):
            os.makedirs(voxel_IFA_dir)
        voxel_IFA_zmaps = PPCA_ICA(reducedsubs,basis=np.vstack((vt, voxel_filters.T)), n_components=None, IFA=True,random_state=random_state, output_folder=voxel_IFA_dir)

        GICA_dir = os.path.join(ICA_dir, "GICA")
        if not os.path.exists(GICA_dir):
            os.makedirs(GICA_dir)
        ICA_zmaps = PPCA_ICA(reducedsubs,basis=None, n_components=int(nPCA+2*n_filters_per_group), IFA=False,random_state=random_state, output_folder=GICA_dir)


        spatial_maps = [ICA_zmaps, parcelvoxel_IFA_zmaps, voxel_IFA_zmaps]

        outputfolders = [GICA_dir, parcel_IFA_dir, voxel_IFA_dir]


        sample = np.min((200,train_idx.shape[0]))
        dual_regressor = DualRegress(
            subs=paths,
            spatial_maps=spatial_maps,
            train_index=train_idx,
            train_labels=train_labels,
            outputfolders=outputfolders,
            workers=15,
            sample=sample,
            method="bayesian",
            parallel_points=15,
            parallel_subs=15,
            n_calls=15,
            random_state=random_state
        )

        dual_regressor.dual_regress()
        del dual_regressor
        gc.collect()

        # Analyze each set of spatial maps
        nPCA_results = os.path.join(nPCA_dir, "Results")
        if not os.path.exists(nPCA_results):
            os.makedirs(nPCA_results)

        map_names = ["GICA","parcel_IFA","voxel_IFA"]

        normalized_result = []
        unnormalized_result = []

        for i, map_i in enumerate(map_names):        
            nPCA_results_maps = os.path.join(nPCA_results, map_i)
            if not os.path.exists(nPCA_results_maps):
                os.makedirs(nPCA_results_maps)
            
            # For Normalized results
            nPCA_results_maps_norm = os.path.join(nPCA_results_maps, "Normalized")
            if not os.path.exists(nPCA_results_maps_norm):
                os.makedirs(nPCA_results_maps_norm)
            ##### BELOW HERE ############
            # Load files into temporary variables.
            tmp_an = np.load(os.path.join(outputfolders[i], "An.npy"))
            tmp_spatial_map = np.load(os.path.join(outputfolders[i], "spatial_map.npy"))
            tmp_recon_norm = np.load(os.path.join(outputfolders[i], "reconstruction_error_norm.npy"))
            
            normalized_result_i = evaluate((tmp_an, tmp_spatial_map, tmp_recon_norm), 
                                        labels, train_idx, test_idx, a_label, b_label,
                                        metric=metric, alpha=0.05, paired=paired, 
                                        permutations=10000, deconf=deconfound, 
                                        con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                                        con_confounder_test=test_con_confounders, cat_confounder_test=test_cat_confounders,
                                        output_dir=nPCA_results_maps_norm, random_seed=random_state, basis=f"{map_i}_Normalized", n_workers=10)           

            normalized_result.append(normalized_result_i)
            
            # Remove the temporary variables and force garbage collection.
            del tmp_an, tmp_spatial_map, tmp_recon_norm
            gc.collect()
            
            # For Unnormalized (demeaned) results:
            nPCA_results_maps_unnorm = os.path.join(nPCA_results_maps, "Unnormalized")
            if not os.path.exists(nPCA_results_maps_unnorm):
                os.makedirs(nPCA_results_maps_unnorm)
            
            tmp_adm = np.load(os.path.join(outputfolders[i], "Adm.npy"))
            tmp_spatial_mapdm = np.load(os.path.join(outputfolders[i], "spatial_mapdm.npy"))
            tmp_recon_dm = np.load(os.path.join(outputfolders[i], "reconstruction_error_dm.npy"))
            
            unnormalized_result_i = evaluate((tmp_adm, tmp_spatial_mapdm, tmp_recon_dm), 
                                            labels, train_idx, test_idx, a_label, b_label,
                                            metric=metric, alpha=0.05, paired=paired, 
                                            permutations=10000, deconf=deconfound, 
                                            con_confounder_train=train_con_confounders, cat_confounder_train=train_cat_confounders, 
                                            con_confounder_test=test_con_confounders, cat_confounder_test=test_cat_confounders,
                                            output_dir=nPCA_results_maps_unnorm, random_seed=random_state, basis=f"{map_i}_Unnormalized", n_workers=10)
            
            unnormalized_result.append(unnormalized_result_i)
            
            # Clean up the temporary variables.
            del tmp_adm, tmp_spatial_mapdm, tmp_recon_dm
            gc.collect()

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