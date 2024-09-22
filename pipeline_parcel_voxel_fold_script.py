
# import cupy as cp
import argparse
import torch
import hcp_utils as hcp # https://rmldj.github.io/hcp-utils/
from statsmodels.stats.multitest import multipletests
import matlab.engine
import os
import json
import psutil
import random
import numpy as np
import pandas as pd
import re
import psutil
import seaborn as sns
from scipy.linalg import eigh, svd
from scipy.stats import norm
from sklearn.decomposition import FastICA, PCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from nilearn import image as nimg
from nilearn import plotting
import nibabel as nib
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space, untangent_space, log_map_riemann, unupper
from pyriemann.utils.distance import distance_riemann, distance
from pyriemann.utils.base import logm, expm
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import resource
import traceback
from functools import partial
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import euclidean_distances


def process_subject_haufe(sub,pinv_TF):
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

        Xp = hcp.normalize(subject - subject.mean(axis=1, keepdims=True))
        del subject
        Xpf = pinv_TF@Xp
        del Xp
        return Xpf
    
    except Exception as e:
        print(f"Error processing subject: {e}")
        traceback.print_exc()  # Print the full traceback
        return None


def haufe_transform(F, parcellated,paths):
    # Ensure the tensors are on the correct device
    pinv_TF = np.linalg.pinv(parcellated.reshape(-1,parcellated.shape[-1]) @ np.linalg.pinv(F.T))

    pinv_TF_list = pinv_TF.reshape(len(paths),pinv_TF.shape[0],-1)
    with ProcessPoolExecutor(max_workers=(int(os.cpu_count()* 0.5))) as executor:
        # Use map to process subjects in parallel
        blocks = np.array(list(executor.map(process_subject_haufe, paths,pinv_TF_list)))
        print(blocks.shape)
        return (blocks.sum(axis=0))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process a specific fold for the analysis.")
    parser.add_argument(
        "--fold",
        type=int,
        required=True,
        help="The fold number to process (e.g., 0, 1, 2, ...)."
    )
    parser.add_argument(
        "--outputfolder",
        type=str,
        default="Gender_TangentSVMC1_logeuclid",
        help="The main output folder where results will be saved."
    )
    return parser.parse_args()

def orthonormalize_filters(W1, W2):
    # Stack the two filters into a single matrix
    W = np.concatenate((W1, W2)).T  # shape: (features x 2)
    print(W.shape)

    # Perform QR decomposition to orthonormalize the filters
    Q, _ = np.linalg.qr(W)

    print(Q.shape)

    # Verify that the inner product between the two orthonormalized vectors is 0 (orthogonality)
    print(f'Inner product between Q[:, 0] and Q[:, 1]: {np.dot(Q[:, 0].T, Q[:, 1])} (should be 0)')

    # Verify that the inner product within each vector is 1 (normalization)
    print(f'Norm of Q[:, 0]: {np.dot(Q[:, 0].T, Q[:, 0])} (should be 1)')
    print(f'Norm of Q[:, 1]: {np.dot(Q[:, 1].T, Q[:, 1])} (should be 1)')

    return Q

def FKT(groupA_cov_matrices, groupB_cov_matrices, mean="riemann", average=True, visualize=True, n=0):
    # Eigenvalues in ascending order from scipy eigh https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html
    if average:
        groupA_cov = mean_covariance(groupA_cov_matrices, metric=mean)
        groupB_cov = mean_covariance(groupB_cov_matrices, metric=mean)    
        # eigs, filters  = eigh(groupA_cov, groupA_cov + groupB_cov + gamma*np.identity(groupB_cov.shape[0]),eigvals_only=False)
    else:
        groupA_cov = groupA_cov_matrices
        groupB_cov = groupB_cov_matrices
    if n > 0:
        eigsA, filtersA  = eigh(groupA_cov, groupA_cov + groupB_cov,eigvals_only=False,subset_by_index=[groupA_cov.shape[0]-n, groupA_cov.shape[0]-1])
        eigsB, filtersB = eigh(groupB_cov, groupA_cov + groupB_cov,eigvals_only=False,subset_by_index=[groupB_cov.shape[0]-n, groupB_cov.shape[0]-1])
    else:
        eigsA, filtersA  = eigh(groupA_cov, groupA_cov + groupB_cov,eigvals_only=False,subset_by_value=[0.5,np.inf])
        eigsB, filtersB = eigh(groupB_cov, groupA_cov + groupB_cov,eigvals_only=False,subset_by_value=[0.5,np.inf])

    eigs = np.concatenate((eigsB[::-1], eigsA))
    filters = np.concatenate((filtersB[:, ::-1], filtersA), axis=1)
    fkt_riem_eigs = np.abs(np.log(eigs/(1-eigs)))**2

    return fkt_riem_eigs, filters, filtersA, filtersB

def tangent_classifier(group1_covs=None, group2_covs=None, Frechet_Mean=None, tangent_projected_1=None, tangent_projected_2=None, TSVM=True, TLDA=False, tangent_calc=True,metric="riemann",visualize=False,n=0):
    if tangent_calc:
        all_covs = np.concatenate((group1_covs, group2_covs))
        Frechet_Mean = mean_covariance(all_covs, metric=metric)
        tangent_projected_1 = tangent_space(group1_covs, Frechet_Mean, metric=metric)
        tangent_projected_2 = tangent_space(group2_covs, Frechet_Mean, metric=metric)
    # Create labels for each group
    labels_1 = np.ones(len(tangent_projected_1))  # Labels for group 1
    labels_2 = np.zeros(len(tangent_projected_2))   # Labels for group 2

    data = np.concatenate((tangent_projected_1, tangent_projected_2))
    labels = np.concatenate((labels_1, labels_2))

    if TSVM:
        # Create SVM classifier (adjust kernel and parameters as needed)
        # C_values = [0.001, 0.01, 0.1, 1, 10]
        # clf = LinearSVC(penalty='l1',loss='squared_hinge',dual=False,C=.1,class_weight='balanced')
        # clf =  LogisticRegression(class_weight='balanced')

        # clf = LinearSVC(penalty='l2',loss='hinge',C=1,class_weight='balanced')  
        clf = SVC(kernel='linear', C=1, class_weight='balanced')  
        # clf = LogisticRegression()

        # Train the classifier
        clf.fit(data, labels)
        # normalized_coef = normalize(clf.coef_, axis=1)
        filters_SVM = untangent_space(clf.coef_, Frechet_Mean)[0,:,:]
        fkt_riem_eigs_tangent_SVM, fkt_filters_tangent_SVM  = eigh(filters_SVM, Frechet_Mean)

        # If test data is provided, project the test data to tangent space
        # tangent_projected_1_discrim_reconstruction = np.mean(tangent_projected_1@np.linalg.pinv(clf.coef_)@clf.coef_,axis=0)
        # print(tangent_projected_1_discrim_reconstruction.shape)
        # tangent_projected_2_discrim_reconstruction = np.mean(tangent_projected_2@np.linalg.pinv(clf.coef_)@clf.coef_,axis=0)
        # print(tangent_projected_2_discrim_reconstruction.shape)
        # group1_discrim_mean = untangent_space(tangent_projected_1_discrim_reconstruction, Frechet_Mean)
        # print(group1_discrim_mean.shape)
        # group2_discrim_mean = untangent_space(tangent_projected_2_discrim_reconstruction, Frechet_Mean)
        # print(group2_discrim_mean.shape)
        # fkt_riem_eigs_tangent_SVM, fkt_filters_tangent_SVM, filtersA, filtersB = FKT(group1_discrim_mean, group2_discrim_mean, mean=metric, average=False, visualize=visualize, n=n)


        # Return accuracy along with filters
        # fkt_riem_eigs_tangent_SVM, fkt_filters_tangent_SVM, filtersA, filtersB = FKT(filters_SVM[0, :, :], Frechet_Mean, mean=metric, average=False, visualize=visualize, n=n)
        # eigs1, filters1  = eigh(filters_SVM, mean_covariance(group2_covs, metric=metric) ,eigvals_only=False,subset_by_value=[0.5,np.inf])
        # eigs2, filters2 = eigh(filters_SVM, mean_covariance(group1_covs, metric=metric) ,eigvals_only=False,subset_by_value=[0.5,np.inf])
        # eigs = np.concatenate((eigs2[::-1], eigs1))
        # fkt_filters_tangent_SVM = np.concatenate((filters2[:, ::-1], filters1), axis=1)
        # fkt_riem_eigs_tangent_SVM = np.abs(np.log(eigs))**2
    

        return fkt_riem_eigs_tangent_SVM, fkt_filters_tangent_SVM, None, None

    if TLDA:
        # Create LDA classifier
        lda = LDA()
        # Train the classifier
        lda.fit(data, labels)
        # Get the coefficients from LDA
        normalized_coef = normalize(lda.coef_, axis=1)
        filters_LDA = untangent_space(normalized_coef, Frechet_Mean)
        fkt_filters_tangent_LDA, fkt_riem_eigs_tangent_LDA, filtersA, filtersB = FKT(filters_LDA[0,:,:], Frechet_Mean, mean=metric, average=False, visualize=visualize,n=n)
        return fkt_filters_tangent_LDA, fkt_riem_eigs_tangent_LDA, filtersA, filtersB
    
def migp(subs, batch_size=2, m=4800):
    W_gpu = None
    for batch_start in range(0, len(subs), batch_size):
        # Select the current batch of subjects
        batch_subs = subs[batch_start:batch_start + batch_size]
        batch_paths = [path for sublist in batch_subs for path in sublist]

        concatenated_data = []

        for task in batch_paths:
            X = nib.load(task).get_fdata()
            Xn = hcp.normalize(X-X.mean(axis=1, keepdims=True))
            # print(Xn.mean(axis=0).mean())
            # print(Xn.std(axis=0).mean())
            concatenated_data.append(Xn)
            del X, Xn
                
        try:
            # Concatenate data along the first axis using numpy
            batch = np.concatenate(concatenated_data, axis=0)
            batch = hcp.normalize(batch - batch.mean(axis=1,keepdims=True))
            del concatenated_data

            with torch.no_grad():
                # Convert to torch tensor and move to GPU
                batch_gpu = torch.tensor(batch, dtype=torch.float32, device="cuda")
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
                print(batch_start, "done")
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
    
    return np.argmax(rrn_std)+1

def get_n_and_some(data):
    # Check the shape of the data and determine the axis for mean subtraction

    # Move data to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_gpu = data.to(device, dtype=torch.float32)
    groupN = data_gpu.shape[1] - 1

    # Subtract the mean along the specified axis
    data_centered = data_gpu - torch.mean(data_gpu, dim=1, keepdim=True)
    del data_gpu  # Free up GPU memory
    torch.cuda.empty_cache()
    # Perform SVD decomposition
    _, d, v = torch.svd(data_centered)
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

    return n_components, v.T

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

def whiten(X,n_components, method="SVD", visualize=False):
    # -1 to account for demean
    n_samples = X.shape[-1]-1
    X_mean = X.mean(axis=-1)
    X -= X_mean[:, np.newaxis]

    if method == "SVD":
        u, d = svd(X, full_matrices=False, check_finite=False)[:2]
        # Give consistent eigenvectors for both svd solvers
        # u *= np.sign(u[0])
        K = (u / d).T[:n_components]  # see (6.33) p.140
        del u, d
        whitening_matrix = np.sqrt(n_samples)*K
    elif method == "Cholesky":
    # Does not Orthogonalize, just has unit covariance
        # Step 2: Perform Cholesky decomposition
        L = np.linalg.cholesky(np.cov(X,ddof=1))
        # Step 3:
        whitening_matrix = np.linalg.inv(L)
    elif method == "InvCov":
        # Calculate the covariance matrix of the centered data
        cov_matrix = np.cov(X)
        # Perform eigenvalue decomposition of the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        # Calculate the whitening matrix
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        whitening_matrix = eigvecs @ D_inv_sqrt @ eigvecs.T
    
    whitened_data = whitening_matrix@X

    return whitened_data, whitening_matrix

def noise_projection(W,data, visualize=True):

    Signals = np.linalg.pinv(W)@(W@data)
    Residuals = data - Signals
    residual_std = np.std(Residuals,axis=0,ddof=np.linalg.matrix_rank(W))
    # Trace of I-pinv(W)(W) is equal to the nullity (n-m gvien n > m) of the reconstructed matrix 
    # trace = data.shape[0] - np.linalg.matrix_rank(W)
    # residual_std2 = (np.einsum('ij,ij->j', Residuals, Residuals)/(trace))**.5

    return residual_std


def threshold_and_visualize(data, W, components,visualize=False):
    voxel_noise = noise_projection(W,data)[:, np.newaxis]
    z_scores_array = np.zeros_like(components)
    z_scores = np.zeros_like(components)

    # Process each filter individually
    for i in range(components.shape[1]):
        z_score = ((components[:, i:i+1]))/voxel_noise
        # P(Z < -z \text{ or } Z > z) = (1 - \text{CDF}(z)) + (1 - \text{CDF}(z)) = 2 \times (1 - \text{CDF}(z))
        p_values = 2 * (1 - norm.cdf(np.abs(z_score)))
        # Apply multiple comparisons correction for the current filter https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
        reject, pvals_corrected, _, _ = multipletests(p_values.flatten(), alpha=0.05, method='fdr_bh')
        masked_comp = z_score*(reject[:,np.newaxis])
        # print(masked_comp, reject[:,np.newaxis],z_score)
        z_scores_array[:, i:i+1] = masked_comp  
        z_scores[:,i:i+1] = z_score

    return z_scores, z_scores_array


def ICA(data,whitened_data):
    ica = FastICA(whiten=False)
    # Takes in array-like of shape (n_samples, n_features) and returns ndarray of shape (n_samples, n_components)
    IFA_components = ica.fit_transform(whitened_data.T).T
    A = data@np.linalg.pinv(IFA_components)
    W = np.linalg.pinv(A)

    return IFA_components, A, W

def calculate_netmat_and_spatial_map(Xn, z_maps):
    """
    Calculate the network matrix (netmat) and spatial map for a given subject and z_maps.

    Parameters:
    Xn (array): Time x Grayordinates normalized data matrix (Time x V)
    z_maps (array): Grayordinates x Components map (V x C)

    Returns:
    netmat (array): Components x Components network matrix (C x C)
    spatial_map (array): Components x Grayordinates matrix (C x V)
    """
    # Time x Components
    # Demean the regressors (z_maps)
    z_maps_demeaned = z_maps - z_maps.mean(axis=0, keepdims=True)  # Demean the columns of z_maps (V x C)

    # Time x Components
    A = (Xn @ np.linalg.pinv(z_maps_demeaned.T))  # A is Time x Components (T x C)
    

    # Normalized Time x Components matrix
    An = hcp.normalize(A)  # An is Time x Components (T x C)
    del A

    # Components x Components network matrix
    netmat = (An.T @ An) / (Xn.shape[0] - 1)  # Netmat is Components x Components (C x C)

    # Components x Grayordinates spatial map
    spatial_map = np.linalg.pinv(An) @ Xn  # Spatial map is Components x Grayordinates (C x V)

    return An, netmat, spatial_map

def dual_regress_sub(sub_path, z_maps_1, z_maps_2):
    try:
        concatenated_data = []
        for task in sub_path:
            # Load and preprocess each task
            X = nib.load(task).get_fdata(dtype=np.float32)  # Grayordinates x Time (V x T)
            Xn = hcp.normalize(X - X.mean(axis=1, keepdims=True))  # Normalizing (V x T)
            concatenated_data.append(Xn)
            del X, Xn
            
        # Concatenate data along the first axis (all tasks into one big matrix)
        subject = np.concatenate(concatenated_data, axis=0)  # Time x Grayordinates (T x V)
        del concatenated_data
            
        # Normalize the concatenated data
        Xn = hcp.normalize(subject - subject.mean(axis=1,keepdims=True))  # Time x Grayordinates normalized data (T x V)
        del subject
            
        # Calculate netmat and spatial map for the first set of z_maps
        An_1, netmat_1, spatial_map_1 = calculate_netmat_and_spatial_map(Xn, z_maps_1)

        # Calculate netmat and spatial map for the second set of z_maps
        An_2, netmat_2, spatial_map_2 = calculate_netmat_and_spatial_map(Xn, z_maps_2)

        return (An_1, netmat_1, spatial_map_1), (An_2, netmat_2, spatial_map_2)

    except Exception as e:
        print(f"Error processing subject: {e}")
        return None, None

def dual_regress(group_paths, z_maps_1, z_maps_2):
    # Use partial to avoid duplicating z_maps in memory
    with ProcessPoolExecutor(max_workers=int(os.cpu_count() * 0.5)) as executor:
        # Create a partial function that "binds" the z_maps_1 and z_maps_2 without duplicating them
        partial_func = partial(dual_regress_sub, z_maps_1=z_maps_1, z_maps_2=z_maps_2)

        # Pass the subject paths to the executor without copying z_maps
        results = list(executor.map(partial_func, group_paths))

        # Separate the results for the two bases, collecting An, netmat, and spatial_map
        An_1, netmats_1, spatial_maps_1 = zip(*[(res[0][0], res[0][1], res[0][2]) for res in results if res[0] is not None])
        An_2, netmats_2, spatial_maps_2 = zip(*[(res[1][0], res[1][1], res[1][2]) for res in results if res[1] is not None])

        return (np.array(An_1), np.array(netmats_1), np.array(spatial_maps_1)), (np.array(An_2), np.array(netmats_2), np.array(spatial_maps_2))


def main():
    # Parse command-line arguments
    args = parse_arguments()
    fold = args.fold
    print(f"Processing fold {fold}")
        
    settings = {
        "outputfolder": "Gender_TangentSVMC1_logeuclid",
        "n_folds": 5,
        "TanSVM_C": 1,
        "random_state": 42,
        "n_filters_per_group": 1,
        "Tangent_Class": True,
        "metric": "logeuclid"
    }

    # Ensure the output folder exists
    outputfolder = settings["outputfolder"]
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)

    # Define the path for the settings file
    settings_filepath = os.path.join(outputfolder, "settings.json")

    # Save the settings to a JSON file
    with open(settings_filepath, "w") as f:
        json.dump(settings, f, indent=4)

    print(f"Settings have been saved to {settings_filepath}")
    # Define the output folder
    n_folds = settings["n_folds"]
    TanSVM_C = settings["TanSVM_C"]
    random_state = settings["random_state"]
    n_filters_per_group = settings["n_filters_per_group"]
    Tangent_Class = settings["Tangent_Class"]
    # Pyriemannian Mean https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/utils/mean.py#L633 Metric for mean estimation, can be: "ale", "alm", "euclid", "harmonic", "identity", "kullback_sym", "logdet", "logeuclid", "riemann", "wasserstein", or a callable function.
    # https://link.springer.com/article/10.1007/s12021-020-09473-9 <---- best descriptions/plots
    # Geometric means in a novel vector space structure on symmetric positive-definite matrices <https://epubs.siam.org/doi/abs/10.1137/050637996?journalCode=sjmael>`_
    metric = settings["metric"]

    def load_array_from_outputfolder(filename):
        filepath = os.path.join(outputfolder, filename)
        return np.load(filepath)
    # Function to save an array to the output folder
    def save_array_to_outputfolder(filename, array):
        filepath = os.path.join(outputfolder, filename)
        np.save(filepath, array)

    groupA_parcellated_array = load_array_from_outputfolder("groupA_parcellated_array.npy")
    groupB_parcellated_array = load_array_from_outputfolder("groupB_parcellated_array.npy")
    groupA_paths_filtered = load_array_from_outputfolder("groupA_paths_filtered.npy")
    groupB_paths_filtered = load_array_from_outputfolder("groupB_paths_filtered.npy")

    data = np.concatenate((groupA_parcellated_array, groupB_parcellated_array))
    labels = np.concatenate((np.ones(len(groupA_parcellated_array)), np.zeros(len(groupB_parcellated_array))))
    paths = np.concatenate((groupA_paths_filtered, groupB_paths_filtered))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    splits = list(skf.split(data, labels))

    fold_outputfolder = f"fold_{fold}"
    if not os.path.exists(os.path.join(outputfolder, f"fold_{fold}")):
        os.makedirs(os.path.join(outputfolder, f"fold_{fold}"))

    train_parcellated, test_parcellated = data[splits[fold][0]], data[splits[fold][1]]
    train_labels, test_labels = labels[splits[fold][0]], labels[splits[fold][1]]
    train_paths, test_paths = paths[splits[fold][0]], paths[splits[fold][1]]

    groupA_train_parcellated = train_parcellated[train_labels == 1]
    groupA_test_parcellated = test_parcellated[test_labels == 1]
    groupA_train_paths = train_paths[train_labels == 1]
    groupA_test_paths = test_paths[test_labels == 1]

    groupB_train_parcellated = train_parcellated[train_labels == 0]
    groupB_test_parcellated = test_parcellated[test_labels == 0]
    groupB_train_paths = train_paths[train_labels == 0]
    groupB_test_paths = test_paths[test_labels == 0]

    # https://pyriemann.readthedocs.io/en/latest/auto_examples/signal/plot_covariance_estimation.html
    cov_est = Covariances(estimator='lwf')
    groupA_train_parcellated_covs = cov_est.transform(np.transpose(groupA_train_parcellated, (0, 2, 1)))
    groupB_train_parcellated_covs = cov_est.transform(np.transpose(groupB_train_parcellated, (0, 2, 1)))

    if Tangent_Class:
        fkt_riem_eigs, filters, _, _ =  tangent_classifier(groupA_train_parcellated_covs,  groupB_train_parcellated_covs, TSVM=True, TLDA=False, tangent_calc=True, metric=metric,visualize=True,n=0)
    else:
        fkt_riem_eigs, filters, filtersA, filtersB = FKT(groupA_train_parcellated_covs, groupB_train_parcellated_covs, mean=metric, average=True, visualize=True, n=0)

    filters = filters[:, [0,-1]]
    filtersA = filters[:, -1:]
    filtersB = filters[:, :1]

    # reducedsubsA = migp((groupA_train_paths))
    # save_array_to_outputfolder(os.path.join(fold_outputfolder,'reducedsubsA.npy'), reducedsubsA)

    # reducedsubsB = migp((groupB_train_paths))
    # save_array_to_outputfolder(os.path.join(fold_outputfolder,'reducedsubsB.npy'), reducedsubsB)

    reducedsubsA_loaded = load_array_from_outputfolder(os.path.join(fold_outputfolder,'reducedsubsA.npy'))
    reducedsubsB_loaded = load_array_from_outputfolder(os.path.join(fold_outputfolder,'reducedsubsB.npy'))

    reducedsubs_combined_gpu = torch.tensor(np.concatenate((reducedsubsA_loaded,reducedsubsB_loaded),axis=0),dtype=torch.float32,device="cuda")
    # Returned in Descending Order
    Urc,_,_ = torch.linalg.svd(reducedsubs_combined_gpu, full_matrices=False)
    reducedsubs_combined = (Urc[:,:4800].T@reducedsubs_combined_gpu).cpu().numpy()
    del Urc, reducedsubs_combined_gpu

    reducedsubs_gpu = torch.tensor(reducedsubs_combined, dtype=torch.float32, device="cuda")
    U,_,_ = torch.linalg.svd(reducedsubs_gpu, full_matrices=False)
    reducedsubs= (U[:,:1000].T@reducedsubs_gpu).cpu().numpy()
    del U, reducedsubs_gpu

    print("Starting A Filter Transform")
    filtersA_transform = haufe_transform(filtersA,groupA_train_parcellated,groupA_train_paths)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"filtersA_transform.npy"), filtersA_transform)
    print("Finished A Filter Transform")
    
    print("Starting B Filter Transform")
    filtersB_transform = haufe_transform(filtersB,groupB_train_parcellated,groupB_train_paths)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"filtersB_transform.npy"), filtersB_transform)
    print("Finished B Filter Transform")

    filters = orthonormalize_filters(filtersA_transform, filtersB_transform)

    subs_data_VN, vt = PPCA(reducedsubs.copy(), threshold=0.0, niters=1)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"subs_data_VN.npy"), subs_data_VN)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"vt.npy"), vt)

    # Combine Basis
    combined_spatial = np.vstack((vt,filters.T))

    # Whiten
    whitened_basis, whitening_matrix_pre = whiten(combined_spatial,n_components=combined_spatial.shape[0],method="InvCov",visualize=True)
    subs_data_com_VN, _ = PPCA(reducedsubs_combined.copy(), filters=whitened_basis.T, threshold=0.0, niters=1)

    raw_components_combined, A_combined, W_combined = ICA(subs_data_com_VN,whitened_basis)

    vtwhiten,_ = whiten(vt,n_components=vt.shape[0],method="SVD")
    subs_data_VN, _ = PPCA(reducedsubs_combined.copy(), filters=vtwhiten.T, threshold=0.0, niters=1)
    raw_components_major, A_major, W_major = ICA(subs_data_VN,vtwhiten)

    subs_data_VN_more, vtmore = PPCA(reducedsubs.copy(), threshold=0.0, niters=1,n=vt.shape[0]+filters.shape[1])
    vtmorewhiten,_ = whiten(vtmore,n_components=vtmore.shape[0],method="SVD")
    subs_data_VN_more, _ = PPCA(reducedsubs_combined.copy(), filters=vtmorewhiten.T, threshold=0.0, niters=1)

    raw_components_major_more, A_major_more, W_major_more = ICA(subs_data_VN_more,vtmorewhiten)

    save_array_to_outputfolder(os.path.join(fold_outputfolder,"raw_components_combined.npy"), raw_components_combined)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"A_combined.npy"), A_combined)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"W_combined.npy"), W_combined)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"raw_components_major.npy"), raw_components_major)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"A_major.npy"), A_major)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"W_major.npy"), W_major)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"raw_components_major_more.npy"), raw_components_major_more)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"A_major_more.npy"), A_major_more)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"W_major_more.npy"), W_major_more)

    z_scores_unthresh, z_scores_thresh = threshold_and_visualize(subs_data_com_VN, W_combined, raw_components_combined.T, visualize=False)

    z_scores_unthresh_major, z_scores_thresh_major = threshold_and_visualize(subs_data_VN, W_major, raw_components_major.T, visualize=False)

    z_scores_unthresh_major_more, z_scores_thresh_major_more = threshold_and_visualize(subs_data_VN_more, W_major_more, raw_components_major_more.T, visualize=False)

    save_array_to_outputfolder(os.path.join(fold_outputfolder,"z_scores_unthresh.npy"), z_scores_unthresh)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"z_scores_thresh.npy"), z_scores_thresh)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"z_scores_unthresh_major.npy"), z_scores_unthresh_major)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"z_scores_thresh_major.npy"), z_scores_thresh_major)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"z_scores_unthresh_major_more.npy"), z_scores_unthresh_major_more)
    save_array_to_outputfolder(os.path.join(fold_outputfolder,"z_scores_thresh_major_more.npy"), z_scores_thresh_major_more)

    # Save function for An, netmats, and spatial maps
    def save_numpy_arrays(output_prefix, An_1, netmats_1, spatial_maps_1, An_2, netmats_2, spatial_maps_2):
        """
        Saves the An arrays, netmats, and spatial maps to disk using np.save.
        
        Parameters:
        output_prefix (str): Prefix for the output files.
        An_1 (np.array): Time x Components matrix for z_maps_1.
        netmats_1 (np.array): Network matrices for z_maps_1.
        spatial_maps_1 (np.array): Spatial maps for z_maps_1.
        An_2 (np.array): Time x Components matrix for z_maps_2.
        netmats_2 (np.array): Network matrices for z_maps_2.
        spatial_maps_2 (np.array): Spatial maps for z_maps_2.
        """
        save_array_to_outputfolder(f"{output_prefix}_An_1.npy", An_1)
        save_array_to_outputfolder(f"{output_prefix}_netmats_1.npy", netmats_1)
        save_array_to_outputfolder(f"{output_prefix}_spatial_maps_1.npy", spatial_maps_1)
        save_array_to_outputfolder(f"{output_prefix}_An_2.npy", An_2)
        save_array_to_outputfolder(f"{output_prefix}_netmats_2.npy", netmats_2)
        save_array_to_outputfolder(f"{output_prefix}_spatial_maps_2.npy", spatial_maps_2)

    print("Starting Dual Regression")
    # For Group A - Training Set
    (groupA_An_1_train, groupA_netmats_1_train, groupA_spatial_maps_1_train), (groupA_An_2_train, groupA_netmats_2_train, groupA_spatial_maps_2_train) = dual_regress(groupA_train_paths, z_scores_unthresh, z_scores_unthresh_major_more)
    save_numpy_arrays(os.path.join(fold_outputfolder,"groupA_train"), groupA_An_1_train, groupA_netmats_1_train, groupA_spatial_maps_1_train, groupA_An_2_train, groupA_netmats_2_train, groupA_spatial_maps_2_train)
    # For Group A - Test Set
    (groupA_An_1_test, groupA_netmats_1_test, groupA_spatial_maps_1_test), (groupA_An_2_test, groupA_netmats_2_test, groupA_spatial_maps_2_test) = dual_regress(groupA_test_paths, z_scores_unthresh, z_scores_unthresh_major_more)
    save_numpy_arrays(os.path.join(fold_outputfolder,"groupA_test"), groupA_An_1_test, groupA_netmats_1_test, groupA_spatial_maps_1_test, groupA_An_2_test, groupA_netmats_2_test, groupA_spatial_maps_2_test)
    # For Group B - Training Set
    (groupB_An_1_train, groupB_netmats_1_train, groupB_spatial_maps_1_train), (groupB_An_2_train, groupB_netmats_2_train, groupB_spatial_maps_2_train) = dual_regress(groupB_train_paths, z_scores_unthresh, z_scores_unthresh_major_more)
    save_numpy_arrays(os.path.join(fold_outputfolder,"groupB_train"), groupB_An_1_train, groupB_netmats_1_train, groupB_spatial_maps_1_train, groupB_An_2_train, groupB_netmats_2_train, groupB_spatial_maps_2_train)
    # For Group B - Test Set
    (groupB_An_1_test, groupB_netmats_1_test, groupB_spatial_maps_1_test), (groupB_An_2_test, groupB_netmats_2_test, groupB_spatial_maps_2_test) = dual_regress(groupB_test_paths, z_scores_unthresh, z_scores_unthresh_major_more)
    save_numpy_arrays(os.path.join(fold_outputfolder,"groupB_test"), groupB_An_1_test, groupB_netmats_1_test, groupB_spatial_maps_1_test, groupB_An_2_test, groupB_netmats_2_test, groupB_spatial_maps_2_test)
    print("Finished Dual Regression")


if __name__ == "__main__":
    main()