import hcp_utils as hcp # https://rmldj.github.io/hcp-utils/
import nibabel as nib
import torch
import numpy as np
import matlab.engine
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append('/utils')
from preprocessing import load_subject #TODO chaneg back to non PCN
from preprocessing import gpu_mem, cpu_mem

def migp(subs,batch_size=1, m=4800):  #TODO chaneg back to non PCN m=4800

    if batch_size > len(subs):
        print(f"Warning: batch_size ({batch_size}) is greater than number of subjects ({len(subs)}). Setting batch_size to {len(subs)}.")
        batch_size = len(subs)

    W_gpu = None

    for batch_start in range(0, len(subs), batch_size):
        # Select the current batch of subjects
        batch_subs = subs[batch_start:batch_start + batch_size]
        try:
            concatenated_data = []  
            for path in batch_subs:
                concatenated_data.append(load_subject(path))

            # Concatenate data along the first axis  
            batch = np.concatenate(concatenated_data, axis=0)
            del concatenated_data

            with torch.no_grad():

                # Convert to torch tensor and move to GPU
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                batch_gpu = torch.tensor(batch, dtype=torch.float32, device=device)
                del batch

                if torch.isnan(batch_gpu).any():
                    print("NaNs detected in the batch data. Aborting SVD operation.")
                    del batch_gpu
                    torch.cuda.empty_cache()
                    return None
                if W_gpu is None:
                    combined_data_gpu = batch_gpu
                else:
                    combined_data_gpu = torch.cat([W_gpu, batch_gpu], dim=0)
                del batch_gpu
                torch.cuda.empty_cache()


                # # Calculate size in GB
                # size_in_gb = combined_data_gpu.element_size() * combined_data_gpu.nelement() / (1024**3)
                # print(f"Size of the array: {size_in_gb:.2f} GB")
                # cpu_mem()
                # gpu_mem()
                # Perform SVD on the GPU
                # Check for NaNs in the data

                # _, S_gpu, Vh_gpu = torch.linalg.svd(combined_data_gpu, full_matrices=False)
                _, Q = torch.linalg.eigh(combined_data_gpu@combined_data_gpu.T)
                # cpu_mem()
                # gpu_mem()
                # Compute the updated W on the GPU
                # W_gpu = torch.diag(S_gpu[:m]) @ Vh_gpu[:m, :]
                # Returned in Ascending order
   
                W_gpu = Q[:, -m:].T@combined_data_gpu
                del Q, combined_data_gpu  # Free up GPU memory
                torch.cuda.empty_cache()
                # print(batch_start, "done",flush=True)

        except Exception as e:
            print(f"Failed during GPU processing: {e}")
            if "combined_data_gpu" in locals():
                del combined_data_gpu
            if "Q" in locals():
                del Q
            if "W_gpu" in locals():
                del W_gpu
            torch.cuda.empty_cache()
            return None

    # Transfer W back to CPU only at the end
    W = W_gpu.cpu().numpy()
    del W_gpu  # Free up GPU memory

    return W

def call_pca_dim(Data=None,eigs=None,N=None):
   # Start MATLAB engine
    eng = matlab.engine.start_matlab()
    
    # Add the path to the MATLAB function
    eng.addpath("/project/3022057.01/IFA/melodic", nargout=0)
    
    if Data is not None:
      # Call the MATLAB function
      prob = eng.pca_dim(matlab.double(Data))
      eig_vectors = np.array(prob['E'])
    else:
      prob = eng.pca_dim_eigs(matlab.double(eigs.tolist()), matlab.double([N]))

    # Extract and convert each variable
    lap = np.array(prob['lap']).flatten().reshape(-1, 1)
    bic = np.array(prob['bic']).flatten().reshape(-1, 1)
    rrn = np.array(prob['rrn']).flatten().reshape(-1, 1)
    AIC = np.array(prob['AIC']).flatten().reshape(-1, 1)
    MDL = np.array(prob['MDL']).flatten().reshape(-1, 1)
    eig = np.array(prob['eig']).flatten()
    orig_eig = np.array(prob['orig_eig']).flatten()
    leig = np.array(prob['leig']).flatten()

    # Stop MATLAB engine
    eng.eval('clearvars', nargout=0)
    eng.quit()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(eig)),eig,label="Adjusted Eigenspectrum")
    plt.scatter(np.arange(len(orig_eig)),orig_eig,label="Eigenspectrum")
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.title('Scree Plot')
    plt.show()


    # Use SimpleImputer to handle any missing values
    imputer = SimpleImputer(strategy='mean')
    lap = imputer.fit_transform(lap)
    bic = imputer.fit_transform(bic)
    rrn = imputer.fit_transform(rrn)
    AIC = imputer.fit_transform(AIC)
    MDL = imputer.fit_transform(MDL)
    
    # Use StandardScaler to standardize the data
    scaler = StandardScaler()
    lap_std = scaler.fit_transform(lap)
    bic_std = scaler.fit_transform(bic)
    rrn_std = scaler.fit_transform(rrn)
    AIC_std = scaler.fit_transform(AIC)
    MDL_std = scaler.fit_transform(MDL)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(len(lap_std)), lap_std, label='Laplacian')
    plt.scatter(np.arange(len(bic_std)), bic_std, label='BIC')
    plt.scatter(np.arange(len(rrn_std)), rrn_std, label='RRN')
    plt.scatter(np.arange(len(AIC_std)), AIC_std, label='AIC')
    plt.scatter(np.arange(len(MDL_std)), MDL_std, label='MDL')
    
    plt.xlabel('Index')
    plt.ylabel('Standardized Value')
    plt.legend()
    plt.title('Scatter Plot of Standardized Eigenvalues and Model Order Selection Values')
    plt.show()
   
    return np.argmax(rrn_std)+1

def get_n_and_some(data):
    # Check the shape of the data and determine the axis for mean subtraction

    # Move data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_gpu = data.to(device, dtype=torch.float32)
    groupN = data_gpu.shape[1] - 1

    if torch.isnan(data_gpu).any():
        raise ValueError("NaNs detected in the data after imputation.")

    # Subtract the mean along the specified axis
    data_centered = data_gpu - torch.mean(data_gpu, dim=1, keepdim=True)
    del data_gpu  # Free up GPU memory
    torch.cuda.empty_cache()

    # Check for NaNs after mean subtraction
    if torch.isnan(data_centered).any():
        raise ValueError("NaNs detected in the data after mean subtraction.")
    
    # Perform SVD decomposition
    _, d, Vh = torch.linalg.svd(data_centered, full_matrices=False)
    del data_centered  # Free up GPU memory
    torch.cuda.empty_cache()
    
    # Convert singular values to eigenvalues
    e = (d ** 2) / groupN

    # Move eigenvalues to CPU and convert to NumPy array
    e_np = e.cpu().numpy()
    del e, d  # Free up GPU memory
    torch.cuda.empty_cache()

    # Determine the number of components
    n_components = torch.tensor(call_pca_dim(eigs=e_np, N=groupN),device=device,dtype=torch.int32)

    return n_components, Vh

def PPCA(data, filters=None, threshold=1.6, niters=10, n=-1):
    n_components = -1
    n_prev = -2
    i = 0

    # Move data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_gpu = torch.tensor(data,device=device, dtype=torch.float32)

    while n_components != n_prev and i < niters:
        n_prev = n_components
        if filters is not None:
            basis_gpu =  torch.tensor(filters.T,device=device, dtype=torch.float32)
        else:
            n_components, vt = get_n_and_some(data_gpu)
            if n <= 0:
                basis_gpu = vt[:n_components, :]
            else:
                print(n)
                basis_gpu = vt[:n, :]
            del vt
            torch.cuda.empty_cache()
        
        print(n_prev, n_components)

        # Estimate noise and residual standard deviation
        est_noise = data_gpu - (data_gpu @ torch.linalg.pinv(basis_gpu)) @ basis_gpu
        est_residual_std = torch.std(est_noise,dim=0,correction=torch.linalg.matrix_rank(basis_gpu))
        del est_noise
        torch.cuda.empty_cache()

        # Normalize the data
        data_gpu = (data_gpu / est_residual_std)
        i += 1

    data = data_gpu.cpu().numpy()
    basis = basis_gpu.cpu().numpy()
    # del data_gpu, basis_gpu, est_residual_std
    del data_gpu, basis_gpu
    torch.cuda.empty_cache()
    return data, basis