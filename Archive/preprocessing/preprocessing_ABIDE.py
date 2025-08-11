import os
import numpy as np
import pandas as pd
import glob
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import hcp_utils as hcp # https://rmldj.github.io/hcp-utils/
import nibabel as nib
import resource
import psutil
import torch
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiLabelsMasker
from nilearn.maskers import NiftiMasker
import nibabel as nib
from sklearn.impute import SimpleImputer

confounders = ["AGE_AT_SCAN", "SEX", "SITE_ID", "func_mean_fd"]
continuous_confounders = ["AGE_AT_SCAN", "func_mean_fd"]
categorical_confounders = ["SEX", "SITE_ID"]

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


def get_meta_data():
    # Base path for preprocessed files
    # base_path = '/project_cephfs/3022017.06/ABIDE/data/Outputs/css/filt_noglobal/func_preproc/*_*_func_preproc.nii.gz'
    base_path = '/project/3022057.01/ABIDE/Outputs/ccs/filt_noglobal/func_preproc/*_*_func_preproc.nii.gz'
    # Use glob to find all matching files
    matching_files = glob.glob(base_path)

    # Extract subject IDs and paths
    data = []
    for file in matching_files:
        # Extract the filename from the full path
        filename = file.split('/')[-1]  # Get the file name (e.g., Pitt_0050003_func_preproc.nii.gz)
        
        # Extract the subject ID (FILE_ID) from the filename
        match = re.match(r'(.*?)_func_preproc\.nii\.gz', filename)
        if match:
            subid = match.group(1)
            data.append({'FILE_ID': subid, 'Path': file})

    # Convert to DataFrame
    file_df = pd.DataFrame(data)

    # Load the phenotypic data
    phenotype = pd.read_csv("/project/3022057.01/ABIDE/Phenotypic_V1_0b_preprocessed1.csv")

    # Merge the file DataFrame with the phenotypic DataFrame on FILE_ID
    merged_df = file_df.merge(phenotype, on="FILE_ID", how="inner")

    return merged_df
    
def load_subject(subject):
    try:
        # Load the functional image
        func_img = nib.load(subject)
        
        # Apply whole-brain masking
        masker = NiftiMasker(mask_strategy="whole-brain-template", standardize=False)
        X = masker.fit_transform(func_img)
        
        # Clean up memory
        del func_img
        
        # Normalize the data, subtracting the mean
        Xn = X - X.mean(axis=1, keepdims=True)
        del X
        # Replace NaNs with column means using SimpleImputer
        imputer = SimpleImputer(strategy="mean")
        Xn = imputer.fit_transform(Xn)
        
        # Further normalization using HCP's normalization
        Xn = hcp.normalize(Xn)
        return Xn

    except Exception as e:
        print(f"Error processing subject: {e}")
        traceback.print_exc()  # Print the full traceback
        return None


def process_subject(sub):
    try:
        atlas_path = "/project/3022057.01/resources/rois/ICPAtlas_v4_fine_208parcels.nii.gz"
        atlas_img = nib.load(atlas_path)
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=False)
        func_img = nib.load(sub)
        Xp = masker.fit_transform(func_img)
        del func_img  # Explicitly delete the subject array
        Xp = hcp.normalize(Xp - Xp.mean(axis=1,keepdims=True))
        return Xp

    except Exception as e:
        print(f"Error processing subject: {e}")
        traceback.print_exc()  # Print the full traceback
        return None

           
def parcellate(output_dir, target_shape=(195, 208),n_workers=.4):
    meta_data_df = get_meta_data()
    subids = meta_data_df["FILE_ID"].tolist()
    paths = meta_data_df["Path"].tolist()
    try:
        with ProcessPoolExecutor(max_workers=(int(os.cpu_count()*n_workers))) as executor:
            # Use map to process subjects in parallel
            group_parcellated = list(executor.map(process_subject, paths))
        
        # Filter out any None results and ensure correct shape
        # valid_index = [index for index, result in enumerate(group_parcellated) if result is not None and result.shape == target_shape] #TODO uncomment
        valid_index = [index for index, result in enumerate(group_parcellated) if result is not None] #TODO delete

        # Filter subids and parcellated data using list comprehensions
        valid_subids = [subids[index] for index in valid_index]
        valid_parcellated = [group_parcellated[index] for index in valid_index]

        # Create a new DataFrame with valid subids and parcellated data
        parcellated_data_df = pd.DataFrame({
            'FILE_ID': valid_subids,
            'parcellated_data': valid_parcellated
        })

        # Perform an inner join with the metadata DataFrame
        full_df = meta_data_df.merge(parcellated_data_df, on='FILE_ID', how='inner')

        # Save the combined DataFrame as a pickle file
        pickle_path = os.path.join(output_dir, "phenotype_parcellated_data.pkl")
        full_df.to_pickle(pickle_path)
        print(f"Data successfully saved to {pickle_path}")

        return full_df

    except Exception as e:
        print(f"Error in parcellation process: {e}")
        traceback.print_exc()
        return []