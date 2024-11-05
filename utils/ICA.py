import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


def ICA(data,whitened_data):
    ica = FastICA(whiten=False)
    # Takes in array-like of shape (n_samples, n_features) and returns ndarray of shape (n_samples, n_components)
    IFA_components = ica.fit_transform(whitened_data.T).T
    A = data@np.linalg.pinv(IFA_components)
    W = np.linalg.pinv(A)
    print("The combined unmixing matrix correctly calculates the components: ", np.allclose(W@data, IFA_components))
    print("The combined mixing matrix correctly reconstructs the low rank data_demean: ", np.allclose(A@IFA_components, A@(W@data)))


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Heat map for the combined unmixing matrix
    sns.heatmap(W@data, cmap='viridis', ax=axes[0])
    axes[0].set_title('Combined Unmixing Matrix (W @ data)')
    axes[0].set_xlabel('Components')
    axes[0].set_ylabel('Samples')

    # Heat map for the IFA components
    sns.heatmap(IFA_components, cmap='viridis', ax=axes[1])
    axes[1].set_title('IFA Components')
    axes[1].set_xlabel('Components')
    axes[1].set_ylabel('Samples')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return IFA_components, A, W

def noise_projection(W,data, visualize=True):

    Signals = np.linalg.pinv(W)@(W@data)
    Residuals = data - Signals
    residual_std = np.std(Residuals,axis=0,ddof=np.linalg.matrix_rank(W))
    # Trace of I-pinv(W)(W) is equal to the nullity (n-m gvien n > m) of the reconstructed matrix 
    # trace = data.shape[0] - np.linalg.matrix_rank(W)
    # residual_std2 = (np.einsum('ij,ij->j', Residuals, Residuals)/(trace))**.5


    if visualize:
        n=1000
        plt.figure()
        plt.plot(Signals[:n,0:1])
        plt.plot(Residuals[:n,0:1])
        # plt.plot(data[:n,0:1])
        # plt.plot(data[:n,0:1] - (Signals[:n,0:1]+Residuals[:n,0:1]))
        plt.legend(['Signal','Noise', 'Data' ,'Reconstruction Error'])
        plt.title("Calculations based on pinv(W)W Projection Matrix")
        plt.show()

        plt.scatter(range(0,residual_std.shape[0]), residual_std)
        plt.title("Noise std Per Voxel based on pinv(W)W Projection Matrix")
        plt.show()
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

       # Skip the iteration if there are no significant values
        if not np.any(reject) and visualize:
            print(f'Component {i} did not contain any significant values')
            plt.figure()
            plt.hist(z_score, bins=30, color='blue', alpha=0.7)
            plt.title(f"Histogram for Filter {i} NO SIGNIFICANT VALUES")
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.show()
        else:
            if visualize:
                # Create a figure and axes for subplots (1 row of 2 plots per filter)
                fig, axes = plt.subplots(1, 2, figsize=(18, 10))

                ax_hist1 = axes[0]
                ax_img = axes[1]

                # Plot the histogram of the current filter
                ax_hist1.hist(z_score, bins=30, color='blue', alpha=0.7)
                ax_hist1.set_title(f"Histogram for Filter {i}")
                ax_hist1.set_xlabel('Value')
                ax_hist1.set_ylabel('Frequency')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Heat map for the combined unmixing matrix
    sns.heatmap(z_scores, cmap='viridis', ax=axes[0])
    axes[0].set_title('z_score')
    axes[0].set_xlabel('Components')
    axes[0].set_ylabel('Samples')

    # Heat map for the IFA components
    sns.heatmap(z_scores_array, cmap='viridis', ax=axes[1])
    axes[1].set_title('z_score thresh')
    axes[1].set_xlabel('Components')
    axes[1].set_ylabel('Samples')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return z_scores, z_scores_array