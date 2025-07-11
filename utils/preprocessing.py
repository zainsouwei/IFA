import os
import numpy as np
import pandas as pd
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import hcp_utils as hcp # https://rmldj.github.io/hcp-utils/
import nibabel as nib
import resource
import psutil
import torch
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import sys
import json
import re
from pyriemann.estimation import Covariances
from scipy.stats import norm
import scipy.io
from sklearn.preprocessing import StandardScaler

sys.path.append('/utils')

from regression import deconfound, confounders, continuous_confounders, categorical_confounders, phen_confounders, phen_continuous_confounders, phen_categorical_confounders

def gpu_mem():
    # Memory usage information
    print(f"Total memory available: {(torch.cuda.get_device_properties('cuda').total_memory / 1024**3):.2f} GB")
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def cpu_mem():
    # Get the soft and hard limits of virtual memory (address space)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    print(f"Soft limit: {soft / (1024 ** 3):.2f} GB")
    print(f"Hard limit: {hard / (1024 ** 3):.2f} GB")

    # Get the soft and hard limits of the data segment (physical memory usage)
    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    print(f"Soft limit: {soft / (1024 ** 3):.2f} GB")
    print(f"Hard limit: {hard / (1024 ** 3):.2f} GB")
   # Display memory information
    print(f"Total Memory: { psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Available Memory: { psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"Used Memory: { psutil.virtual_memory().used / (1024**3):.2f} GB")
    print(f"Memory Usage: { psutil.virtual_memory().percent}%")



def normalize_data(data):
    return hcp.normalize(data - data.mean(axis=1, keepdims=True))

def update_df_for_condition(df, condition_keyword):
    df_group = df.copy() 
    new_parcellated = []
    new_paths = []
    
    pattern = re.compile(r'(?:LR|RL)_(.+)$', re.IGNORECASE)

    for _, row in df_group.iterrows():
        cond_indices = row["Condition_Indices"]
        # Find keys that contain the condition keyword (case-insensitive)
        matching_keys = [key for key in cond_indices if condition_keyword in pattern.search(key).group(1)]
        
        slices = []
        # Initialize used_indices for LR and RL as empty lists.
        used_indices = {"LR": [], "RL": []}
        # Iterate over matching keys, grouping indices into LR and RL.
        for key in matching_keys:
            indices_list = cond_indices[key]
            if "lr" in key.lower():
                used_indices["LR"].extend(indices_list)
            elif "rl" in key.lower():
                used_indices["RL"].extend(indices_list)
            # Extract the time slices for each (start, end) pair.
            for (start, end) in indices_list:
                slices.append(row["parcellated_data"][start:end, :])
        
        extracted_data = np.vstack(slices)
            
        new_parcellated.append(extracted_data)
        # Override the paths column with the tuple format.
        new_paths.append([row["paths"], used_indices])
    
    # Update the DataFrame with new columns.
    df_group["parcellated_data"] = new_parcellated
    df_group["paths"] = new_paths
    
    return df_group

def get_conditions(settings):
    full_data = pd.read_pickle(settings["data_path"])
    a_condition = settings["phenotype"][0]
    b_condition = settings["phenotype"][1]
    a = update_df_for_condition(full_data,a_condition)
    b = update_df_for_condition(full_data,b_condition)
    return a,b

def check_conf(settings,phenotype,outputfolder):
    # Check for missing data
    data = pd.read_pickle(settings["data_path"])
    data = data.dropna(subset=phen_confounders)
    data["motion"] = data["motion"].apply(lambda x: np.mean(np.array(x)))
    for conf in phen_confounders:
        sns.set(style="whitegrid")
        if len(set(data[conf])) < 20:
            plt.figure(figsize=(15, 8))
            sns.violinplot(x=conf, y=phenotype, data=data, inner="quartile", palette="Set2")
            plt.title(f"Distribution of {phenotype} by {conf}")
            plt.xlabel(f"{conf}")
            plt.ylabel(f"{phenotype}")
            plt.tight_layout()
            plt.savefig(os.path.join(outputfolder,f"{phenotype}_by_{conf}_violin.svg"), format="svg")
    plt.close('all')

def extract_phenotype(file_path_restricted='/project/3022057.01/HCP/RESTRICTED_zainsou_8_6_2024_2_11_21.csv', file_path_unrestricted='/project/3022057.01/HCP/unrestricted_zainsou_8_2_2024_6_13_22.csv'):
    try:
        # Load data from CSV files
        data_r = pd.read_csv(file_path_restricted)
        data_ur = pd.read_csv(file_path_unrestricted)
    except FileNotFoundError:
        print(f"File not found: {file_path_restricted} or {file_path_unrestricted}")
        raise

    # Combine restricted and unrestricted data on Subject ID
    data = pd.merge(data_r, data_ur, on='Subject', how='outer')

    # Filter out rows with NaNs in the specified phenotypes and where '3T_RS-fMRI_Count' is not equal to 4
    # data_filtered = data[data['3T_RS-fMRI_Count'] == 4]    
    return data
    
def load_subject(subject_info):
    """
    Load and normalize subject data with different behavior based on input structure.
    
    Cases:
    1. If subject_info is a list of file paths:
         Process: load, normalize each task, concatenate, and final normalization.
    
    2. If subject_info is [paths, truncate_to]:
         Process: as in Case 1, but then truncate the concatenated data to truncate_to rows.
    
    3. If subject_info is [paths, condition_indices, truncate_to]:
         Process: load, normalize, concatenate; then extract slices using condition_indices 
         (assumed to be a dict with keys like 'LR' and 'RL' whose values are lists of (start, end) pairs),
         combine the extracted slices, truncate the result to truncate_to rows, and then normalize.
    
    Parameters:
      subject_info: either a list of paths (case 1), 
                    or [paths, truncate_to] (case 2),
                    or [paths, condition_indices, truncate_to] (case 3).
    
    Returns:
      A normalized NumPy array of subject data or None if an error occurs.
    """
    try:
        # --- Case: Single file path (simulated data) ---
        if isinstance(subject_info, str) and subject_info.endswith('.npy'):
            # Load the .npy file
            return np.load(subject_info)
        
        # --- Determine the input structure ---
        # Case 1: subject_info is a list of file paths (all elements are strings)
        if isinstance(subject_info, list) and all(isinstance(task, str) for task in subject_info):
            paths = subject_info
            condition_indices = None
            truncate_to = None

        # Case 2: subject_info is [paths, truncate_to]
        elif (isinstance(subject_info, np.ndarray) and len(subject_info) == 2 and
              isinstance(subject_info[0], list) and isinstance(subject_info[1], int)):
            paths = subject_info[0]
            truncate_to = subject_info[1]
            condition_indices = None

        # Case 3: subject_info is [paths, condition_indices, truncate_to]
        elif (isinstance(subject_info, np.ndarray) and len(subject_info) == 3 and
              isinstance(subject_info[0], list) and isinstance(subject_info[1], dict) and isinstance(subject_info[2], int)):
            paths = subject_info[0]
            condition_indices = subject_info[1]
            truncate_to = subject_info[2]

        else:
            raise ValueError("Unsupported subject_info format.")

        # --- Load and normalize each task ---
        concatenated_data = []
        for task in paths:
            X = nib.load(task).get_fdata(dtype=np.float32)
            Xn = hcp.normalize(X - X.mean(axis=1, keepdims=True))
            concatenated_data.append(Xn)
            del X, Xn

        subject_data = np.concatenate(concatenated_data, axis=0)
        del concatenated_data

        # --- Case 3: Apply condition-based slicing if provided ---
        if condition_indices is not None:
            extracted_data = []
            # Expect condition_indices to be a dict with keys (e.g., 'LR' and 'RL')
            for key in condition_indices:
                for start, end in condition_indices[key]:
                    extracted_data.append(subject_data[start:end])
            subject_data = np.vstack(extracted_data)
            del extracted_data

        # --- Apply truncation if requested (Cases 2 and 3) ---
        if truncate_to is not None and subject_data.shape[0] > truncate_to:
            subject_data = subject_data[:truncate_to, :]

        # --- Final normalization ---
        subject_normalized = hcp.normalize(subject_data - subject_data.mean(axis=1, keepdims=True))
        del subject_data

        return subject_normalized

    except Exception as e:
        print(f"Error processing subject: {e}")
        traceback.print_exc()
        return None



def process_subject(sub):
    try:
        normalized_subject = load_subject(sub)
        Xp = hcp.parcellate(normalized_subject, hcp.mmp)
        del normalized_subject  # Explicitly delete the subject array
        Xp = hcp.normalize(Xp - Xp.mean(axis=1,keepdims=True))

        return Xp

    except Exception as e:
        print(f"Error processing subject: {e}")
        traceback.print_exc()  # Print the full traceback
        return None

def create_cca_phenotype(num_weights, headers_path, cca_mat_path, loaded_data_path, apply_deconfounding=True):
    """
    Creates a composite phenotype column based on the top num_weights from the CCA subject-measure weight vector.
    It converts any T-scores to percentiles (for columns with '_Pct'), extracts the top variables,
    optionally deconfounds those phenotype columns, and then computes a weighted sum.
    
    Parameters:
      num_weights (int): Number of top weights (by absolute value) to use.
      headers_path (str): Path to the column_headers.txt file.
      cca_mat_path (str): Path to the MAT file with the CCA weights.
      loaded_data_path (str): Path to the pickle file containing the DataFrame.
      output_col_name (str): (Optional) Name for the new phenotype column.
                             Defaults to "CCA_score_top<num_weights>".
      apply_deconfounding (bool): If True, deconfounds each top phenotype column using phen_continuous_confounders
                                  and phen_categorical_confounders.
    
    Returns:
      pd.DataFrame: A new DataFrame containing only subjects with complete data for the top variables,
                    along with a new composite phenotype column.
    """
    # Load headers (list of 478 variable names)
    with open(headers_path, 'r') as f:
        headers = [line.strip() for line in f if line.strip()]
    
    # Load the MAT file with CCA weights
    cca_data = scipy.io.loadmat(cca_mat_path)
    
    # Load the DataFrame from the pickle file
    loaded_data = pd.read_pickle(loaded_data_path)
    
    # For each header that contains '_Pct', convert the corresponding T score column to percentiles.
    # This will update loaded_data by creating (or replacing) the _Pct columns.
    for col in headers:
        if '_Pct' in col:
            base = col.replace('_Pct', '')
            t_col = base + '_T'
            if t_col in loaded_data.columns:
                loaded_data[col] = norm.cdf((loaded_data[t_col] - 50) / 10) * 100
                print(f"Created/updated column '{col}' from '{t_col}'")
            else:
                print(f"Warning: T column '{t_col}' not found for header '{col}'")
    
    # Extract the weight vector from the MAT file.
    # We assume the subject-measure weights are in cca_data['hcp_cca_public'][0][0][1]
    weights = cca_data['hcp_cca_public'][0][0][1].flatten()  # expected shape (478,)
    
    # Create a mask for non-NaN weights and for headers present in loaded_data
    valid_mask = ~np.isnan(weights)
    loaded_columns = set(loaded_data.columns)
    header_in_loaded = np.array([header in loaded_columns for header in headers])
    combined_mask = valid_mask & header_in_loaded
    
    valid_indices = np.where(combined_mask)[0]
    valid_weights = weights[combined_mask]
    
    # Sort valid weights by descending absolute value
    sorted_valid_indices = np.argsort(np.abs(valid_weights))[::-1]
    
    # Select the top num_weights indices (relative to the full array)
    top_indices = valid_indices[sorted_valid_indices[:num_weights]]
    top_weights = weights[top_indices]
    top_headers = [headers[i] for i in top_indices]
    
    print(f"Top {num_weights} subject-measure weights:")
    for header, weight in zip(top_headers, top_weights):
        print(f"{header}: {weight}")
    
    if apply_deconfounding:
        unique_confounders = list(dict.fromkeys(confounders + phen_confounders))
        # Drop rows with NaN values in the specified columns
        loaded_data["motion"] = loaded_data["motion"].apply(lambda x: np.mean(np.array(x)))
        columns = ["Subject", "parcellated_data", "paths", "Family_ID"] + top_headers + unique_confounders
    else:
        columns = ["Subject", "parcellated_data", "paths", "Family_ID"] + top_headers
    
    X_top = loaded_data[columns]

    # Drop rows (subjects) with any NaNs in these top columns
    X_top_complete = X_top.dropna(axis=0)
    print("Shape of X_top before dropping NaNs:", X_top.shape)
    print("Shape of X_top after dropping rows with NaNs:", X_top_complete.shape)
    
    # If deconfounding is enabled, deconfound each phenotype column individually.
    if apply_deconfounding:
        # For each top variable, deconfound it using the specified phenotype confounders.
        for phenotype in top_headers:
            if X_top_complete[phenotype].dtype != bool:
                # deconfound returns a Series with residuals
                deconf_series = deconfound(
                    X_train=X_top_complete[phenotype],
                    con_confounder_train=X_top_complete[phen_continuous_confounders],
                    cat_confounder_train=X_top_complete[phen_categorical_confounders],
                    phenotype_labels=None,
                    output_path=None 
                )
                # Replace the column with its deconfounded values for the subjects where deconfounding succeeded
                X_top_complete.loc[deconf_series.index, phenotype] = deconf_series

    # After you finish deconfounding your top columns...
    for phenotype in top_headers:
        # Only standardize if it's not boolean
        if X_top_complete[phenotype].dtype != bool:
            # scikit-learn requires a 2D array, so reshape
            values_2d = X_top_complete[phenotype].values.reshape(-1, 1)
            scaler = StandardScaler()
            standardized = scaler.fit_transform(values_2d).flatten()
            X_top_complete.loc[:, phenotype] = standardized

    # Compute the composite phenotype: weighted sum (dot product) for each subject in complete cases.
    phenotype_scores = X_top_complete[top_headers].dot(top_weights)
    
    # Define output column name if not provided.
    output_col_name = "CCA_top"
    
    # Create a new DataFrame (using the deconfounded complete data) and add the composite phenotype column.
    new_data = X_top_complete.copy()
    new_data[output_col_name] = phenotype_scores
    
    return new_data, columns
           
def get_groups(phenotypes, quantile, data_path, apply_deconfounding=True, regression=False, visualize=True, output_dir="plots",bins=30,min_frequency_factor=0.05):
    # Load the phenotype data
    # unique_confounders = list(dict.fromkeys(confounders + phen_confounders))
    # columns = ["Subject", "parcellated_data", "paths", "Family_ID"] + phenotypes + unique_confounders

    # # Load additional data from the provided data path
    # loaded_data = pd.read_pickle(data_path)

    # # Perform the merge using all common columns
    # # Drop rows with NaN values in the specified columns
    # loaded_data["motion"] = loaded_data["motion"].apply(lambda x: np.mean(np.array(x)))
    # # loaded_data = loaded_data[(loaded_data["motion"] < 0.15 ) & (loaded_data["MMSE_Score"] > 26 )]
    # phenotype_data = loaded_data[columns].dropna()
    
    # # Iterate over each categorical confounder
    # unique_cat_confounders = list(dict.fromkeys(categorical_confounders + phen_categorical_confounders))
    # for cat_conf in unique_cat_confounders:
    #     if cat_conf in phenotype_data.columns:  # Ensure the column exists
    #         # Calculate category proportions
    #         category_proportions = phenotype_data[cat_conf].value_counts(normalize=True)

    #         # Identify frequent categories based on the proportion threshold
    #         frequent_categories = category_proportions[category_proportions >= min_frequency_factor].index

    #         # Filter rows with infrequent categories
    #         phenotype_data = phenotype_data[phenotype_data[cat_conf].isin(frequent_categories)]


    # if regression:
    #     # Ensure all phenotypes are continuous for regression
    #     if any(phenotype_data[phenotype].nunique() <= 2 for phenotype in phenotypes):
    #         raise ValueError("All phenotypes must be continuous for regression.")

    #     # Compute summed phenotype values
    #     phenotype_data["SummedValues"] = phenotype_data[phenotypes].sum(axis=1)
        
    #     if visualize:
    #         # Plot histogram for the summed values
    #         plt.figure(figsize=(10, 6))
    #         plt.hist(phenotype_data["SummedValues"], bins=bins, alpha=0.7, color='green')
    #         plt.xlabel('Summed Phenotype Values')
    #         plt.ylabel('Frequency')
    #         plt.title('Histogram of Summed Phenotype Values for Regression Case')
    #         plt.savefig(os.path.join(output_dir, "phenotype_values_histogram.svg"))
    #     return phenotype_data

    loaded_data = pd.read_pickle(data_path)
    if isinstance(phenotypes[0], int) and len(phenotypes) == 1:
        phenotype_data, columns = create_cca_phenotype(phenotypes[0], '/project/3022057.01/HCP/column_headers.txt', 
                                '/project/3022057.01/HCP/hcp_cca_public.mat',
                                data_path,
                                apply_deconfounding=apply_deconfounding)
        # Replace phenotype with newly created CCA column
        phenotypes = ["CCA_top"]
        # Set deconfounding to False so we do not deconfound again since, deconfounding is handled before the CCA projection
        apply_deconfounding = False
    else:
        if apply_deconfounding:
            unique_confounders = list(dict.fromkeys(confounders + phen_confounders))
            # Drop rows with NaN values in the specified columns
            loaded_data["motion"] = loaded_data["motion"].apply(lambda x: np.mean(np.array(x)))
            columns = ["Subject", "parcellated_data", "paths", "Family_ID"] + phenotypes + unique_confounders
            # loaded_data = loaded_data[(loaded_data["motion"] < 0.15 ) & (loaded_data["MMSE_Score"] > 26 )]
        else:
            columns = ["Subject", "parcellated_data", "paths", "Family_ID"] + phenotypes
            
        phenotype_data = loaded_data[columns].dropna()

    group_a_subjects = set(phenotype_data["Subject"])
    group_b_subjects = set(phenotype_data["Subject"])

    labels = ["Upper Quantile", "Lower Quantile"]
    for phenotype in phenotypes:
        unique_values = phenotype_data[phenotype].nunique()

        if unique_values > 2:  # Continuous phenotype
            if apply_deconfounding:
                phenotype_var = deconfound(
                    X_train=phenotype_data[phenotype],
                    con_confounder_train=phenotype_data[phen_continuous_confounders],
                    cat_confounder_train=phenotype_data[phen_categorical_confounders],
                    phenotype_labels=phenotype,output_path=output_dir)
            else:
                phenotype_var = phenotype_data[phenotype]

            # Calculate quantiles on the target
            lower_quantile = np.quantile(phenotype_var, quantile)
            upper_quantile = np.quantile(phenotype_var, 1 - quantile)

            # Identify subjects in the top and bottom quantiles
            top_quantile_subjects = phenotype_data[phenotype_var >= upper_quantile]["Subject"]
            bottom_quantile_subjects = phenotype_data[phenotype_var <= lower_quantile]["Subject"]

            if visualize and apply_deconfounding:
                plt.figure(figsize=(10, 6))
                plt.hist(phenotype_var, bins=bins, alpha=0.5, label='Phenotype Deconfounded', color='green')
                plt.hist(phenotype_var[phenotype_var >= upper_quantile], bins=bins, alpha=0.5, label='Group A (Top Quantiles)', color='blue')
                plt.hist(phenotype_var[phenotype_var <= lower_quantile], bins=bins, alpha=0.5, label='Group B (Bottom Quantiles)', color='red')
                plt.xlabel('Phenotype Values')
                plt.ylabel('Frequency')
                plt.title('Histogram of Phenotype Deconfounded Values for Group A and Group B')
                plt.legend()
                plt.savefig(os.path.join(output_dir, "group_a_b_phenotype_deconf_values_histogram.svg"))

        elif unique_values == 2:  # Binary discrete phenotype
            classes = np.sort(phenotype_data[phenotype].unique())
            top_class, bottom_class = classes[1], classes[0]
            top_quantile_subjects = phenotype_data[phenotype_data[phenotype] == top_class]["Subject"]
            bottom_quantile_subjects = phenotype_data[phenotype_data[phenotype] == bottom_class]["Subject"]
        
        # Update sets with the intersection of subjects
        group_a_subjects &= set(top_quantile_subjects)
        group_b_subjects &= set(bottom_quantile_subjects)

    # Filter phenotype data for the subject groups and make explicit copies
    group_a = phenotype_data[phenotype_data["Subject"].isin(group_a_subjects)].copy()
    group_b = phenotype_data[phenotype_data["Subject"].isin(group_b_subjects)].copy()

    # Add labels to the group DataFrames
    group_a["Label"] = labels[0]
    group_b["Label"] = labels[1]

    # Check for overlapping subjects and raise a warning if found
    if np.sum(group_a["Subject"].isin(group_b["Subject"])) > 0:
        raise ValueError("Overlap between subjects in each class")


    if visualize:
        # Plot histograms for Group A and Group B
        group_a_values = group_a[phenotypes].to_numpy().sum(axis=1)
        group_b_values = group_b[phenotypes].to_numpy().sum(axis=1)

        plt.figure(figsize=(10, 6))
        plt.hist(phenotype_data[phenotypes].to_numpy().sum(axis=1), bins=bins, alpha=0.5, label='Phenotype', color='green')
        plt.hist(group_a_values, bins=bins, alpha=0.5, label='Group A (Top Quantiles)', color='blue')
        plt.hist(group_b_values, bins=bins, alpha=0.5, label='Group B (Bottom Quantiles)', color='red')
        plt.xlabel('Phenotype Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Phenotype Values for Group A and Group B')
        plt.legend()
        plt.savefig(os.path.join(output_dir, "group_a_b_phenotype_values_histogram.svg"))

    plt.close('all')
    return group_a, group_b


def filter_and_truncate_data(all_data, threshold_percentile=0.10,paired=False,outputfile="path"):
    # Compute the length of each subject's time series
    all_data = all_data.copy()  
    all_data['length'] = all_data['parcellated_data'].apply(lambda x: x.shape[0])
    
    # Determine the threshold for dropping subjects
    threshold = all_data['length'].quantile(threshold_percentile)
    
    
    kept_data = all_data[all_data['length'] >= threshold].copy()
    dropped_data = all_data[all_data['length'] < threshold].copy()
    
    # For the paired case have to drop teh subjects everywhere
    if paired:
        dropped_subjects = set(dropped_data["Subject"])
        kept_data = kept_data[~kept_data["Subject"].isin(dropped_subjects)]
        # Update dropped_data to include all subjects not in kept_data.
        dropped_data = all_data[~all_data["Subject"].isin(kept_data["Subject"])]

    # Determine the new truncation length from the kept subjects
    truncated_length = int(kept_data['length'].min())
    
    # Truncate the 'parcellated_data' for each kept subject
    kept_data['parcellated_data'] = kept_data['parcellated_data'].apply(lambda sub: sub[:truncated_length, :])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(kept_data.index, kept_data['length'], color='green', label='Kept Subjects')
    ax.scatter(dropped_data.index, dropped_data['length'], color='red', label='Dropped Subjects')
    ax.axhline(y=truncated_length, color='blue', linestyle='--', label=f'Truncation Point: {truncated_length}')
    num_dropped = dropped_data.shape[0]
    num_kept = kept_data.shape[0]
    ax.set_title(f'Time Series Lengths: {num_kept} Kept, {num_dropped} Dropped (Truncated at {truncated_length})')
    ax.set_xlabel('Subject Index')
    ax.set_ylabel('Time Series Length')
    ax.legend()
    ax.grid(True)
    plt.savefig(outputfile, format="svg")
    plt.close(fig)

    return kept_data, truncated_length

def prepare_data(settings,threshold_percentile=0.10):
    """ Unified function to prepare data for either between-subjects (`get_groups`) or within-subject (`get_conditions`). """
    outputfolder = settings["outputfolder"]
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    # Save the settings dictionary as a JSON file in the output folder
    settings_filepath = os.path.join(outputfolder, 'settings.json')
    with open(settings_filepath, 'w') as f:
        json.dump(settings, f, indent=4)
    
    outputfolder_visualization = os.path.join(settings["outputfolder"],"preprocess_figures")
    if not os.path.exists(outputfolder_visualization):
        os.makedirs(outputfolder_visualization)
    if settings["paired"]:  # Within-subject case
       a,b = get_conditions(settings)
    else:  # Between-subjects case
        if settings["deconfound"]:
            for phenotype in settings["phenotype"]:
                check_conf(settings,phenotype,outputfolder_visualization)
        a, b = get_groups(
            settings["phenotype"],
            quantile=settings["percentile"],
            data_path=settings["data_path"],
            apply_deconfounding=settings["deconfound"],
            regression=False,
            visualize=True,
            output_dir=outputfolder_visualization,
            bins=20
        )
        # Downsample both groups to have an equal number of samples
        min_size = min(len(a), len(b))
        a = a.sample(n=min_size, random_state=settings["random_state"])
        b = b.sample(n=min_size, random_state=settings["random_state"])
        
    a['label'] = settings["a_label"]
    b['label'] = settings["b_label"]
    all_data = pd.concat([a, b], ignore_index=True)
    
    fig_path = os.path.join(outputfolder_visualization, "time_length.svg")
    filtered_data, truncate = filter_and_truncate_data(all_data, threshold_percentile=threshold_percentile, paired=settings["paired"], outputfile=fig_path)
 
    if settings["deconfound"]:
        con_confounders = filtered_data[continuous_confounders].to_numpy()
        cat_confounders = filtered_data[categorical_confounders].to_numpy()
        
        with open(os.path.join(outputfolder, "cat_confounders.pkl"), "wb") as f:
            pickle.dump(cat_confounders, f)
        np.save(os.path.join(outputfolder,"con_confounders.npy"),con_confounders)

    sub_ID = filtered_data["Subject"].to_numpy()
    family_ID = filtered_data["Family_ID"].to_numpy()
    labels = filtered_data["label"].to_numpy()
    
    data = np.array([normalize_data(sub) for sub in filtered_data["parcellated_data"]])
    cov_est = Covariances(estimator='oas')
    covs = cov_est.transform(data.transpose(0, 2, 1))

    paths = list(filtered_data["paths"])
    if isinstance(paths, list) and all(isinstance(task, str) for task in paths[0]):
        paths = [[path] + [truncate] for path in paths]
    else:
        paths = [path + [truncate] for path in paths]
    paths = np.array(paths,dtype=object)

    with open(os.path.join(outputfolder, "paths.pkl"), "wb") as f:
        pickle.dump(paths, f)
    with open(os.path.join(outputfolder, "family_ID.pkl"), "wb") as f:
        pickle.dump(family_ID, f)
    
    np.save(os.path.join(outputfolder,"Sub_ID.npy"),sub_ID)
    np.save(os.path.join(outputfolder,"labels.npy"),labels)
    np.save(os.path.join(outputfolder,"data.npy"),data)
    np.save(os.path.join(outputfolder,"covs.npy"),covs)

    print(f"Data saved in {outputfolder}")


def get_meta_data(condition,base_directory='/project_cephfs/3022017.01/S1200'):
    # Metadata keys and variables
    subid = "Subject"

    # Assuming extract_phenotype() is a function that returns a DataFrame with a 'Subject' column
    data_dict_data_df = extract_phenotype()

    # Set up directories and file paths for fMRI data
    subdirectory = "MNINonLinear/Results"
    if condition == "REST":
        folders = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]
    elif condition == "WM":
        folders = ["tfMRI_WM_LR", "tfMRI_WM_RL"]
    elif condition == "GAMBLING":
        folders = ["tfMRI_GAMBLING_LR","tfMRI_GAMBLING_RL"]
    elif condition == "RELATIONAL":
        folders = ["tfMRI_RELATIONAL_LR", "tfMRI_RELATIONAL_RL"]
    else:
        raise ValueError("Unrecognized condition")

    file_suffix = "_Atlas_MSMAll.dtseries.nii"
    motion_file = "Movement_RelativeRMS_mean.txt"

    # Initialize lists to store subject IDs, scan paths, and motion values
    valid_subids = []
    all_scan_paths = []
    all_motion_values = []

    # Get the list of subject IDs
    subids = data_dict_data_df[subid].tolist()
    for sub in subids:
        subject_scans = []
        subject_motion = []

        # Collect scan paths and motion values for each session
        for folder in folders:
            scan_path = os.path.join(base_directory, str(sub), subdirectory, folder, folder + file_suffix)
            motion_path = os.path.join(base_directory, str(sub), subdirectory, folder, motion_file)

            if os.path.exists(scan_path) and os.path.exists(motion_path):
                subject_scans.append(scan_path)
                # Read only the first line of motion values (single value)
                with open(motion_path, 'r') as f:
                    motion_value = float(f.readline().strip())
                subject_motion.append(motion_value)
            else:
                # If either the scan or motion file doesn't exist, skip to the next subject
                break

        # Only add subjects with all required scan and motion files
        if len(subject_scans) == len(folders):
            valid_subids.append(sub)
            all_scan_paths.append(subject_scans)
            all_motion_values.append(subject_motion)

    # Create a new DataFrame from the collected scan paths and motion values
    motion_data_df = pd.DataFrame({
        'Subject': valid_subids,
        'paths': all_scan_paths,
        'motion': all_motion_values
    })

    # Perform an intersection merge to keep only subjects present in both DataFrames
    combined_df = data_dict_data_df.merge(motion_data_df, on='Subject', how='inner')

    return combined_df

def parcellate(condition, output_dir,base_directory = "/project_cephfs/3022017.01/S1200",n_workers=20):
    meta_data_df = get_meta_data(condition, base_directory=base_directory)
    subids = meta_data_df["Subject"].tolist()
    paths = meta_data_df["paths"].tolist()

    try:
        with ProcessPoolExecutor(max_workers=(n_workers)) as executor:
            # Use map to process subjects in parallel
            group_parcellated = list(executor.map(process_subject, paths))
        
        # Filter out any None results and ensure correct shape
        valid_index = [index for index, result in enumerate(group_parcellated) if result is not None]

        # Filter subids and parcellated data using list comprehensions
        valid_subids = [subids[index] for index in valid_index]
        valid_parcellated = [group_parcellated[index] for index in valid_index]

        # Create a new DataFrame with valid subids and parcellated data
        parcellated_data_df = pd.DataFrame({
            'Subject': valid_subids,
            'parcellated_data': valid_parcellated
        })

        # Perform an inner join with the metadata DataFrame
        full_df = meta_data_df.merge(parcellated_data_df, on='Subject', how='inner')

        # Save the combined DataFrame as a pickle file
        pickle_path = os.path.join(output_dir, "combined_data.pkl")
        full_df.to_pickle(pickle_path)
        print(f"Data successfully saved to {pickle_path}")

        return full_df

    except Exception as e:
        print(f"Error in parcellation process: {e}")
        traceback.print_exc()
        return []

def times(full_data,output_dir,task):
    # Resources
    # Task setups https://www.humanconnectome.org/hcp-protocols-ya-task-fmri#_ENREF_7
    # Task Protocals https://www.humanconnectome.org/hcp-protocols-ya-3t-imaging <--- TR from

    # Constants
    TR = 0.72  # Repetition Time in seconds
    if task == "WM":
        FRAMES_PER_RUN = 405  # Frames per run (for RL shift)
        conditions = ["0bk_faces", "0bk_places", "0bk_tools", "0bk_body",
                            "2bk_faces", "2bk_places", "2bk_tools", "2bk_body"]
        task_dir = ["tfMRI_WM_LR", "tfMRI_WM_RL"]
    elif task == "GAMBLING":
        FRAMES_PER_RUN = 253  # Frames per run (for RL shift)
        conditions = ["loss", "win"]
        task_dir = ["tfMRI_GAMBLING_LR", "tfMRI_GAMBLING_RL"]
    elif task == "RELATIONAL":
        FRAMES_PER_RUN = 232  # Frames per run (for RL shift)
        conditions = ["match", "relation"]
        task_dir = ["tfMRI_RELATIONAL_LR", "tfMRI_RELATIONAL_RL"]
    else:
        raise ValueError("Unrecognized condition")
        
    # Function to map EV times to sample indices
    def ev_to_indices(ev_file, TR, shift=0):
        events = pd.read_csv(ev_file, sep='\t', header=None, names=['Onset', 'Duration', 'Amplitude'])
        indices = [
            (int((row['Onset'] / TR) + shift), int(((row['Onset'] + row['Duration']) / TR) + shift))
            for _, row in events.iterrows()
        ]
        return indices

    # Add index ranges for all conditions for each subject
    def process_subject_condition(row):
        subject_id = row['Subject']
        print(subject_id)
        # Initialize a dictionary to hold condition indices
        condition_indices = {}

        # Iterate over task folders (LR and RL)
        for task_folder, shift in zip(task_dir, [0, FRAMES_PER_RUN]):
            # Define the base EV path
            base_path = f"/project_cephfs/3022017.01/S1200/{subject_id}/MNINonLinear/Results/{task_folder}/EVs"

            # Extract indices for each condition
            for condition in conditions:
                ev_file = f"{base_path}/{condition}.txt"
                try:
                    condition_indices[f"{task_folder}_{condition}"] = ev_to_indices(ev_file, TR, shift)
                except FileNotFoundError:
                    print(f"EV file not found: {ev_file}")
                    condition_indices[f"{task_folder}_{condition}"] = []

        return condition_indices

    # Process all subjects
    all_indices = []
    for idx, row in full_data.iterrows():
        print(idx)
        subject_indices = process_subject_condition(row)
        all_indices.append(subject_indices)

    # Add the indices as a new column in the DataFrame
    full_data["Condition_Indices"] = all_indices

    # Save the combined DataFrame as a pickle file
    pickle_path = os.path.join(output_dir, "combined_data_times.pkl")
    full_data.to_pickle(pickle_path)
    print(f"Data successfully saved to {pickle_path}")
