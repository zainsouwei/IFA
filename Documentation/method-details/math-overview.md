# Algorithm Overview

* **Parcellate Each Subject**
  * Let $$A$$ denote the collection of subject matrices from group A, and $$B$$ denote the collection of subject matrices from group B
    * $$A \in \mathbb{R}^{M \times T_A \times P}$$ $$M$$ is the number of subjects in Group A,  $$T_A$$ is the number of timepoints per subject, and $$P$$ is the number of parcels created from ICP per subject
      * $$A^i$$, where $$A^i \in \mathbb{R}^{ T_A \times P}$$, refers to the subject of group A
    * $$B \in \mathbb{R}^{N \times T_B  \times P}$$  $$N$$ is the number of subjects in Group A,  $$T_B$$ is the number of timepoints per subject, and P is the number of parcels created from ICP per subject
      * $$B^j$$, where $$B^{j} \in \mathbb{R}^{ T_B\times P}$$, refers to the $$jth$$ subject of group A
* **Standardize & Demean Subject Level Time Series**
  * For Each Subject standardize and demean the time series (columns of $$A_i$$and $$B_i$$)
    * $$\text{For } A^i: \\ \text{Let } \bar{A}^i = \left( \frac{1}{T_A} \sum_{t=1}^{T_A} A^{i}_{t,l} \right)_{l=1}^{P} \\ \text{Then, for each column } l \text{ of } A^i, \text{ standardize and demean it as:} \\ (A^i_{:,l})' = \frac{A^i_{:,l} - \bar{A}^i_{l}}{\sigma_{A^i_{l}}} \\ \text{where } A^i_{:,l} \text{ represents the } l\text{th column of } A^i, \\ \bar{A}^i_{l} \text{ is the average of the } l\text{th column of } A^i, \\ \text{and } \sigma_{A^i_{l}} \text{ is the standard deviation of the } l\text{th column of } A^i \text{ (sample estimate).} \\$$
    * $$\text{Similarly, for } B^j: \\ \text{Let } \bar{B}^j = \left( \frac{1}{T_B} \sum_{t=1}^{T_B} B^{j}_{t,l} \right)_{l=1}^{P} \\ \text{Then, for each column } l \text{ of } B^j, \text{ standardize and demean it as:} \\ (B^j_{:,l})' = \frac{B^j_{:,l} - \bar{B}^j_{l}}{\sigma_{B^j_{l}}} \\ \text{where } B^j_{:,l} \text{ represents the } l\text{th column of } B^j, \\ \bar{B}^j_{l} \text{ is the average of the } l\text{th column of } B^j, \\ \text{and } \sigma_{B^j_{l}} \text{ is the standard deviation of the } l\text{th column of } B^j \text{ (sample estimate).}$$
* **Create Subject NetMat: Covariance Matrix**
  * Full $$P \times P$$covariance matrix for each subject$$\text{For each subject } A^i \text{ in group A:} \\ \text{1. Let } A^i \text{ be the fMRI data matrix of shape } (T_A, P), \\ \text{ where } T_A \text{ is the number of time points and } P \text{ is the number of brain regions or voxels.} \\ \text{2. Compute the sample covariance matrix } S_{A^i} \text{ using the formula:} \\ \text{ } S_{A^i} = \frac{1}{T_A - 1} (A^i)^T \cdot A^i \\ \text{ where } (A^i)^T \text{ denotes the transpose of the data matrix } A^i. \\ \text{3. Shrink the sample covariance matrix } S_{A^i} \text{ towards a target matrix } T \text{ using the Ledoit-Wolf method.} \\ \text{ The target matrix } T \text{ is typically chosen to be well-structured, such as the identity matrix or a scaled identity matrix.} \\ \text{4. The Ledoit-Wolf method determines the optimal shrinkage intensity } \alpha \text{ by minimizing a certain loss function,} \\ \text{ which takes into account the trade-off between bias and variance.} \\ \text{5. Compute the regularized covariance matrix } \bar{C}{A^i} \text{ using the formula:} \\ \text{ } \bar{C}{A^i} = (1 - \alpha) \cdot S_{A^i} + \alpha \cdot T \\ \text{ where } \alpha \text{ is the shrinkage intensity obtained from the Ledoit-Wolf method.} \\ \text{6. The resulting } \bar{C}{A^i} \text{ is the regularized covariance matrix for subject } A^i, \\ \text{ and } \bar{C}{A^i} \in \mathbb{S}_+^P \text{ represents a positive semidefinite matrix of size } P \times P. \\$$
  *   $$\text{For } B^j: \\ \text{Let } C_{B^j} = \frac{1}{T_B} (B^j)^T \cdot B^j \\ \text{where } C_{B^j} \text{ is the covariance matrix for subject } B^j \\ \text{and } C_{B^j} \in \mathbb{S}_+^P \text{ represents a positive semidefinite matrix of size } P \times P.$$


* **Create Single NetMat for Each Group: Arithemetic Mean**
  *
  *
* **Discriminant Analysis: FKT GEVD (**
  *

      ```python
      groupA_eigs, groupA_filters = eigh(groupA_cov - l3*np.linalg.inv(groupB_cov), (1-gamma)*(groupA_cov+groupB_cov) + gamma*np.identity(groupB_cov.shape[0]) + l2*np.identity(groupB_cov.shape[0]),eigvals_only=False,subset_by_value=[threshold,1.0])
      groupB_eigs, groupB_filters = eigh(groupB_cov - l3*np.linalg.inv(groupA_cov), (1-gamma)*(groupA_cov+groupB_cov) + gamma*np.identity(groupB_cov.shape[0]) + l2*np.identity(groupB_cov.shape[0]),eigvals_only=False,subset_by_value=[threshold,1.0])

      ```
* **Calculate variance explained by FKT to convey where in eigenspectrum discriminant space lies**
  * project onto basis formed by filters and calculate residuals
* ICA
  * **Project Each subject onto their respective group filters and run fast ICA**
