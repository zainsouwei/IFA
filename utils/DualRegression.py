import numpy as np
import nibabel as nib
import hcp_utils as hcp  # Ensure you have installed hcp-utils, or replace it with your custom method
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2017.00115/full

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
    reconstructed = A@z_maps_demeaned.T
    reconstruction_error = (1 - np.linalg.norm(reconstructed - reconstructed.mean())/np.linalg.norm(Xn - Xn.mean()))*100
   
    # Normalized Time x Components matrix
    An = hcp.normalize(A)  # An is Time x Components (T x C)
    del A

    # Components x Components network matrix
    netmat = (An.T @ An) / (Xn.shape[0] - 1)  # Netmat is Components x Components (C x C)

    # Components x Grayordinates spatial map
    spatial_map = np.linalg.pinv(An) @ Xn  # Spatial map is Components x Grayordinates (C x V)

    return An, netmat, spatial_map, reconstruction_error

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
        An_1, netmat_1, spatial_map_1, reconstruction_error_1 = calculate_netmat_and_spatial_map(Xn, z_maps_1)

        # Calculate netmat and spatial map for the second set of z_maps
        An_2, netmat_2, spatial_map_2, reconstruction_error_2 = calculate_netmat_and_spatial_map(Xn, z_maps_2)

        return (An_1, netmat_1, spatial_map_1, reconstruction_error_1), (An_2, netmat_2, spatial_map_2, reconstruction_error_2)

    except Exception as e:
        print(f"Error processing subject: {e}")
        return None, None

def dual_regress(group_paths, z_maps_1, z_maps_2):
    # Use partial to avoid duplicating z_maps in memory
    with ProcessPoolExecutor(max_workers=int(os.cpu_count() * 0.4)) as executor:
        # Create a partial function that "binds" the z_maps_1 and z_maps_2 without duplicating them
        partial_func = partial(dual_regress_sub, z_maps_1=z_maps_1, z_maps_2=z_maps_2)

        # Pass the subject paths to the executor without copying z_maps
        results = list(executor.map(partial_func, group_paths))
        
        # Separate the results for the two bases, collecting An, netmat, and spatial_map
        An_1, netmats_1, spatial_maps_1, reconstruction_errors_1 = zip(*[(res[0][0], res[0][1], res[0][2], res[0][3]) for res in results if res[0] is not None])
        An_2, netmats_2, spatial_maps_2, reconstruction_errors_2 = zip(*[(res[1][0], res[1][1], res[1][2], res[1][3]) for res in results if res[1] is not None])

        return (np.array(An_1), np.array(netmats_1), np.array(spatial_maps_1), np.array(reconstruction_errors_1)), (np.array(An_2), np.array(netmats_2), np.array(spatial_maps_2), np.array(reconstruction_errors_2))
