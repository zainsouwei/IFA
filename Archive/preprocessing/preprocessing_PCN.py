import os
import numpy as np
import pandas as pd
import glob
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

# Confounders eTIV = estimated Total Intracranial Volume
confounders = ["age_at_cnb", "Sex", "Race", "Motion","eTIV"] 
continuous_confounders = ["age_at_cnb","Motion","eTIV"]
categorical_confounders = ["Sex", "Race"]

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
    # Define the base path with a wildcard for the subject ID
    base_path = '/project_cephfs/3022017.06/PNC/subjects/sub-*/func/rest.feat/ICA_AROMA/denoised_func_data_nonaggr.feat/filtered_func_data_standard.nii.gz'

    # Use glob to find all matching files
    matching_files = glob.glob(base_path)

    # Extract the subject ID (subid) and create a DataFrame
    data = []
    for file in matching_files:
        # Extract subject ID from the file path
        subid = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(file)))))).replace('sub-', '')
        motion_file = f'/project_cephfs/3022017.06/PNC/subjects/sub-{subid}/func/rest.feat/report_prestats.html'
        relative_motion = float([line for line in open(motion_file) if 'Mean displacements' in line][0].split('relative=')[1].split('mm')[0])
        eTIV_file = f'/project_cephfs/3022017.06/PNC/freesurfer/sub-{subid}/stats/aseg.stats'
        eTIV = float([line for line in open(eTIV_file) if 'eTIV' in line][0].split(',')[-2])
        data.append({'SUBJID': int(subid), 'Path': file, 'Motion': relative_motion, 'eTIV': eTIV})

    # Convert the list of dictionaries into a pandas DataFrame
    phenotype = pd.read_csv("/project_cephfs/3022017.06/PNC/phenotypes/phs000607.v3.pht003445.v3.p2.c1.Neurodevelopmental_Genomics_Subject_Phenotypes.GRU-NPU-1.txt", delimiter='\t')
    phenotype = phenotype.loc[phenotype['INT_NUM'] == 1]
    df = pd.DataFrame(data).merge(phenotype, on="SUBJID", how="inner")
    return df
    
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

           
def parcellate(output_dir, target_shape=(124, 208),n_workers=.4):
    meta_data_df = get_meta_data()
    subids = meta_data_df["SUBJID"].tolist()
    paths = meta_data_df["Path"].tolist()
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
            'SUBJID': valid_subids,
            'parcellated_data': valid_parcellated
        })

        # Perform an inner join with the metadata DataFrame
        full_df = meta_data_df.merge(parcellated_data_df, on='SUBJID', how='inner')

        # Save the combined DataFrame as a pickle file
        pickle_path = os.path.join(output_dir, "phenotype_parcellated_data.pkl")
        full_df.to_pickle(pickle_path)
        print(f"Data successfully saved to {pickle_path}")

        return full_df

    except Exception as e:
        print(f"Error in parcellation process: {e}")
        traceback.print_exc()
        return []