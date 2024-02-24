# Algorithm Summary

## 2023.11.1-2023.11.28

* When running SPADE, I used [abs() on the filters](https://github.com/zainsouwei/ICASPADE/blob/070a5d3ab2e6b09d2aee03aa106db90f8a5f71f2/simulate\_time.py#L83). Otherwise, the sign invariance of EVD caused structured symmetry within a group's cluster if SPADE was run on the subject level
* For the subject-specific filter case, I am just using the [filters rather than a dual regression or a projection](https://github.com/zainsouwei/ICASPADE/blob/21adaa891aab69852804d4ae05bb6f2460be63d4/simulate\_time.py#L280C1-L284C49).
  * Using dual regression or projection should yield the same results, this is just the most accurate and computationally efficient
* For the dual regression, [I am using the inner product rather than the full pseudoinverse](https://github.com/zainsouwei/ICASPADE/blob/21adaa891aab69852804d4ae05bb6f2460be63d4/simulate\_time.py#L100C1-L105C44). Technically these are equal if the spatial filters are orthogonal which by the spectral theorem holds true when the matrix multiplication of the inverse of one of the covariance matrices times the other covariance matrix is hermitian.
* The [second filter](https://github.com/zainsouwei/ICASPADE/blob/21adaa891aab69852804d4ae05bb6f2460be63d4/simulate\_time.py#L82C12-L82C83), which says how the control group discriminates from the non-control group or the subject group, is not too stable. This is because, during the dual regression, this regressor does not actually correlate with the subject so if there is little noise then these regressors will model anything that has some value. How well this filter is expressed in the dual regression is related to [the level of noise](https://github.com/zainsouwei/ICASPADE/blob/21adaa891aab69852804d4ae05bb6f2460be63d4/simulate\_time.py#L256). Additionally, this filter does not add much to the variance captured between non-control groups so it is always not included in the [scree plots](https://github.com/zainsouwei/ICASPADE/blob/21adaa891aab69852804d4ae05bb6f2460be63d4/simulate\_time.py#L290).
* The level of accuracy of the clustering is dependent on the frequency of the independent components in the subjects which is controlled by the number of groups, [number of shared components, number of filters, max number of places the filters occur, and the number of parcels/voxels](https://github.com/zainsouwei/ICASPADE/blob/21adaa891aab69852804d4ae05bb6f2460be63d4/simulate\_time.py#L250C3-L257C1)

## 2023.11.29

## 2023.12.05

* The reason we do dual in SPADE ICA pipeline is because we are working with VEV' covairance matrix initially where in classic ICA we have the data and can project onto SVD components of choosing form UEV'
*   All of these are valid/useful

    <figure><img src="../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>
*   However our main pipeline/generation of the independent filter analysis is

    <figure><img src="../.gitbook/assets/image (2) (1).png" alt=""><figcaption></figcaption></figure>

    <figure><img src="../.gitbook/assets/image (4).png" alt=""><figcaption></figcaption></figure>

## 2023.12.08

* When adding future extensions/variants, the main changes should be in the
  * Discrimination - Current FKT
    * Kernel Discriminatory Analysis
    * Kernel PCA
  * Explanation - Current ICA
    * PROFUMO
    * Dictionary Learning

## 2023.12.13

* When adding future extensions/variants, can also look at different FCs
  * [Life | Free Full-Text | Statistical Approaches to Identify Pairwise and High-Order Brain Functional Connectivity Signatures on a Single-Subject Basis (mdpi.com)](https://www.mdpi.com/2075-1729/13/10/2075)
  * Hybrid high-order functional connectivity networks using resting-state functional mri for mild cognitive impairment diagnosis
  * [EAG-RS: A Novel Explainability-guided ROI-Selection Framework for ASD Diagnosis via Inter-regional Relation Learning](https://arxiv.org/pdf/2310.03404.pdf)
    * Has a lot of references of different FC methods
* Can look at different spectral clustering methods outside of FKT (linear or nonlinear)
  * Read [Open Problems in Spectral Dimensionality Reduction](https://link-springer-com.proxy.lib.umich.edu/book/10.1007/978-3-319-03943-5)

## 2023.12.14

## 2023.12.15

* For unsupervised analysis/individual analysis/no reference group, constructing a similarity matrix via summation of the eigenvalues variance from .5 or summation of the eigenvalues variance and then running MDS effectively discriminates the simulated data in a low dimension. I think it can be viewed as a manifold learning technique since it is unsupervised, the fisher discrimination can be viewed as metric learning, and then the MDS represents the data in a lower dimension. I am not sure if that interpretation is 100% correct, but I think it might be worthwhile to look into this problem from a manifold learning approach as well. For more on the relation of FDA/FKT to manifolds and metrics look into work from [Benyamin Ghojogh](https://arxiv.org/pdf/1906.09436.pdf). [This ](https://www-jstage-jst-go-jp.proxy.lib.umich.edu/article/jsaisigtwo/2007/DMSM-A603/2007\_04/\_pdf/-char/ja)seems to take a similar approach but is supervised I think, but look into it.

<figure><img src="../.gitbook/assets/image (5).png" alt=""><figcaption><p>Similarity matrix constructed using np.sum(abs(eigh(C1, C1+C2, eigvals_only=True)-.5))</p></figcaption></figure>

* With not as many shared components normal MDS with Euclidean similarity performs similarly. However, when there were more shared components, it appears the MDS with the Fisher discrimination similarity started to perform a little better. Maybe this can be better tested with complex data that has different noise to components ratio and shared components to discriminative components ratio

<div align="center">

<figure><img src="../.gitbook/assets/image (7).png" alt=""><figcaption><p>MDS with Similairty Matrix Constructed from Pairwise Fisher Discrimination</p></figcaption></figure>

</div>

<figure><img src="../.gitbook/assets/image (8).png" alt=""><figcaption><p>MDS with Similarity Matrix Constructed Using Euclidean Distance</p></figcaption></figure>

## 2023.12.17

## 2024.01.15

* Whitening in ICA seems to increase performance and interpretability but still needs to be looked into further
* Factors that influence the performance of FKT+ICA on semi-simulated data
  * Simulated Signal-to-Noise Ratio
  * Number of Parcels Containing Discriminate Signal
    * FKT so far seems to be robust to the number of parcels whereas dual regression is not as robust
  * Number of subjects containing discriminate signal
    * This is related to signal to noise ratio due to averaging of covariance matrices
  * Simulated signal to non-simulated signal ratio
    * When adding the simulated signal to subjects, I have been multiplying the original signal in that voxel by a value \[0.0,1.0]
    * Given the parcel(s) I have added the signal to is not noise
      * When the value is 0, the non-semi-simulated filter from FKT is also significant. This is because I removed this signal from all the subjects to whom I added the simulated signal.
      * When the value > 0, the non-semi-simulated filter becomes less discriminate because the semi-simulated data is similar to some aspects of the non-semi-simulated subjects since the semi-simulated data was created by adding a signal on top of a preexisting feature seen in all subjects
      * If the simulated signal is added to multiple parcels and the parcels have different features or vastly different noise, this decreases the discriminate ability of the feature or splits the discriminate filter into multiple filters since this is a spatial decomposition more than it is a temporal decomposition

## 2024.01.17

* Quantifying Variance/Discrimination
  * Quantifying Discrimination of PCA on Group-Concatenated Data
    * running PCA for model order selection and then FKT, and hopefully showing that as the model order is increased, FKT can find filters with higher discrimination (given the discriminatory information lies in the low variance eigenspace)
  * Quantifying Variance of FKT
    * quantify the amount of variance that the eigenvectors from the FKT after dual regression express of the full data (ie if you did an F-ratio test between the total projected data on the FKT vectors relative to the full variance)
    * That way individual vectors do not matter and instead you are comparing the volume of the space collectively spanned by the FKT/’PCA vectors independent on how precisely they span that space

## 2024.01.22

## 2024.01.23

## 2024.01.25

* Covariance Edge Cases
  * Group A has the same time series in parcels x,y,z. Group B has different time series in parcels x,y,z.
    * This will cause inv(groupA+groupB)\*Group A covariance matrix to have high off diagonal elements and low on diagonal elements on x,y,z which will result in one discriminant component for group A. However, inv(groupA+groupB)\*group B will have low off diagonal elements for parcels x,y,z but high elements on the diagonal for x,y,z which will cause group B to have 3 components. Since the 3 components from Group B can be used to form the component from Group A, ICA might split the Group A component
  * Group A has the same time series in x,y,z, and Group B has the same time series in x,y,z. The time series between Group A and Group B are different but have the same total energy causing same covariance matrices
  * Group A has the same time series in x,y,z and Group B has random noise in X,Y,Z. Random noise will have a higher variance and a lower covariance than other parcels, making it a more dominant feature in Group B than it is in Group A.
