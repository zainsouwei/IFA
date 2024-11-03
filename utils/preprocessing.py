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


def extract_phenotype(phenotypes, file_path_restricted='/project/3022057.01/HCP/RESTRICTED_zainsou_8_6_2024_2_11_21.csv', file_path_unrestricted='/project/3022057.01/HCP/unrestricted_zainsou_8_2_2024_6_13_22.csv'):
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
    data_filtered = data[data['3T_RS-fMRI_Count'] == 4]
    data_filtered = data_filtered[phenotypes].dropna()
    
    return data_filtered

def get_meta_data(base_directory='/project_cephfs/3022017.01/S1200'):
    # Metadata keys and variables
    subid = "Subject"
    familyid = "Family_ID"
    # Confounders 
    # confounds detailed in https://www.sciencedirect.com/science/article/pii/S1053811920300914 & https://www.humanconnectome.org/storage/app/media/documentation/s500/HCP500_MegaTrawl_April2015.pdf
    # In Data Table: Age (Age_in_Yrs), Sex (Gender), Ethnicity (Ethnicity), Weight (Weight), Brain Size (FS_BrainSeg_Vol), Intracranial Volume (FS_IntraCranial_Vol), Confounds Modelling Slow Drift (TestRetestInterval), reconstruction code version (fMRI_3T_ReconVrs) or Acquisition Quarter (Acquisition)
    # In pathfile: Head Motion (a summation over all timepoints of timepoint-to-timepoint relative head motion or average) Movement_RelativeRMS_mean.txt (Since LR RL and session scans are concateanted, take average of this average)
    # Mentioned in papers but not found: variables (x, y, z, table) related to bed position in scanner
    age = "Age_in_Yrs"
    sex = "Gender"
    ethnicity = "Ethnicity"
    weight = "Weight"
    brain_size = "FS_BrainSeg_Vol"
    intracranial_volume = "FS_IntraCranial_Vol"
    reconstruction_code = "fMRI_3T_ReconVrs"

    data_dict_data = [subid, familyid, age, sex, ethnicity, weight, brain_size, intracranial_volume, reconstruction_code]
    data_dict_data_df = extract_phenotype(data_dict_data)

    # Set up directories and file paths for fMRI data
    subdirectory = "MNINonLinear/Results"
    folders = ["rfMRI_REST1_LR", "rfMRI_REST1_RL", "rfMRI_REST2_LR", "rfMRI_REST2_RL"]
    file_suffix = "_Atlas_MSMAll_hp2000_clean.dtseries.nii"
    motion_file = "Movement_RelativeRMS_mean.txt"
    # Initialize lists to store subject IDs, scan paths, and single motion values
    valid_subids = []
    all_scan_paths = []
    all_motion_values = []

    # Vectorized iteration over the subject IDs
    subids = data_dict_data_df[subid].tolist()
    for sub in subids:
        subject_scans = []
        subject_motion = []

        # Collect scan paths and single motion values for each session
        for folder in folders:
            scan_path = os.path.join(base_directory, str(sub), subdirectory, folder, folder + file_suffix)
            motion_path = os.path.join(base_directory, str(sub), subdirectory, folder, motion_file)

            if os.path.exists(scan_path) and os.path.exists(motion_path):
                subject_scans.append(scan_path)
                # Read only the first line of motion values (single value)
                with open(motion_path, 'r') as f:
                    motion_value = float(f.readline().strip())
                subject_motion.append(motion_value)

        # Only add subjects with all required scan and motion files
        if len(subject_scans) == len(folders):
            valid_subids.append(sub)
            all_scan_paths.append(subject_scans)
            all_motion_values.append(subject_motion)  # List of single values

    # Create a new DataFrame from the collected scan paths and motion values
    motion_data_df = pd.DataFrame({
        'Subject': valid_subids,
        'paths': all_scan_paths,
        'motion': all_motion_values
    })

    # Perform an intersection merge to keep only subjects present in both DataFrames
    combined_df = data_dict_data_df.merge(motion_data_df, on='Subject', how='inner')

    return combined_df
    

def process_subject(sub):
    try:
        concatenated_data = []
        for task in sub:
            X = nib.load(task).get_fdata(dtype=np.float32)
            Xn = hcp.normalize(X-X.mean(axis=1, keepdims=True))
            concatenated_data.append(Xn)
            del X, Xn

        # Concatenate data along the first axis
        subject = np.concatenate(concatenated_data, axis=0)
        del concatenated_data  # Explicitly delete the concatenated data list

        Xp = hcp.parcellate(hcp.normalize(subject - subject.mean(axis=1,keepdims=True)), hcp.mmp)
        Xp = hcp.normalize(Xp - Xp.mean(axis=1,keepdims=True))
        del subject  # Explicitly delete the subject array

        return Xp

    except Exception as e:
        print(f"Error processing subject: {e}")
        traceback.print_exc()  # Print the full traceback
        return None

           
def parcellate(output_dir,base_directory = "/project_cephfs/3022017.01/S1200", target_shape=(4800, 379),n_workers=.4):
    meta_data_df = get_meta_data(base_directory=base_directory)
    subids = meta_data_df["Subject"].tolist()
    paths = meta_data_df["paths"].tolist()

    try:
        with ProcessPoolExecutor(max_workers=(int(os.cpu_count()*n_workers))) as executor:
            # Use map to process subjects in parallel
            group_parcellated = list(executor.map(process_subject, paths))
        
        # Filter out any None results and ensure correct shape
        valid_index = [index for index, result in enumerate(group_parcellated) if result is not None and result.shape == target_shape]

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
    
def get_groups(phenotypes, quantile=0.33, data_path='/project/3022057.01/HCP/combined_data.pkl', regression=False, visualize=True):
    # Load the phenotype data
    columns = ["Subject"] + phenotypes
    phenotype_data = extract_phenotype(columns)

    # Load additional data from the provided data path
    loaded_data = pd.read_pickle(data_path)
    
    # Identify all common columns between the two DataFrames, including 'Subject'
    common_columns = phenotype_data.columns.intersection(loaded_data.columns).tolist()

    # Perform the merge using all common columns
    phenotype_data = phenotype_data.merge(loaded_data, on=common_columns, how="inner")

    if regression:
        # Ensure all phenotypes are continuous for regression
        if any(phenotype_data[phenotype].nunique() <= 2 for phenotype in phenotypes):
            raise ValueError("All phenotypes must be continuous for regression.")

        # Compute summed phenotype values
        phenotype_data["SummedValues"] = phenotype_data[phenotypes].sum(axis=1)
        
        if visualize:
            # Plot histogram for the summed values
            plt.figure(figsize=(10, 6))
            plt.hist(phenotype_data["SummedValues"], bins=30, alpha=0.7, color='green')
            plt.xlabel('Summed Phenotype Values')
            plt.ylabel('Frequency')
            plt.title('Histogram of Summed Phenotype Values for Regression Case')
            plt.show()
        
        return phenotype_data

    # Initialize subject sets for intersection
    group_a_subjects = set(phenotype_data["Subject"])
    group_b_subjects = set(phenotype_data["Subject"])

    labels = ["Upper Quantile", "Lower Quantile"]
    for phenotype in phenotypes:
        unique_values = phenotype_data[phenotype].nunique()

        if unique_values > 2:  # Continuous phenotype
            lower_quantile = phenotype_data[phenotype].quantile(quantile)
            upper_quantile = phenotype_data[phenotype].quantile(1 - quantile)
            top_quantile_subjects = phenotype_data[phenotype_data[phenotype] >= upper_quantile]["Subject"]
            bottom_quantile_subjects = phenotype_data[phenotype_data[phenotype] <= lower_quantile]["Subject"]
        elif unique_values == 2:  # Binary discrete phenotype
            classes = phenotype_data[phenotype].unique()
            labels = classes
            top_class, bottom_class = classes[0], classes[1]
            top_quantile_subjects = phenotype_data[phenotype_data[phenotype] == top_class]["Subject"]
            bottom_quantile_subjects = phenotype_data[phenotype_data[phenotype] == bottom_class]["Subject"]
        
        # Update sets with the intersection of subjects
        group_a_subjects &= set(top_quantile_subjects)
        group_b_subjects &= set(bottom_quantile_subjects)

    # Filter phenotype data for the subject groups
    # Filter phenotype data for the subject groups and make explicit copies
    group_a = phenotype_data[phenotype_data["Subject"].isin(group_a_subjects)].copy()
    group_b = phenotype_data[phenotype_data["Subject"].isin(group_b_subjects)].copy()

    # Add labels to the group DataFrames
    group_a["Label"] = labels[0]
    group_b["Label"] = labels[1]

    # Check for overlapping subjects and raise a warning if found
    if np.sum(group_a["Subject"].isin(group_b["Subject"])) > 0:
        raise ValueError("Overlap bnetween subejcts in each class")


    if visualize:
        # Plot histograms for Group A and Group B
        group_a_values = group_a[phenotypes].to_numpy().sum(axis=1)
        group_b_values = group_b[phenotypes].to_numpy().sum(axis=1)

        plt.figure(figsize=(10, 6))
        plt.hist(group_a_values, bins=30, alpha=0.5, label='Group A (Top Quantiles)', color='blue')
        plt.hist(group_b_values, bins=30, alpha=0.5, label='Group B (Bottom Quantiles)', color='red')
        plt.xlabel('Phenotype Values')
        plt.ylabel('Frequency')
        plt.title('Histogram of Phenotype Values for Group A and Group B')
        plt.legend()
        plt.show()

    return group_a, group_b