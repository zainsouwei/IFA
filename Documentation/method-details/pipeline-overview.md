# Pipeline Overview

* **Parcellate Each Subject**
  * Current: 208 Parcels from ICP
* **Create NetMats**
  * Current: full covariance matrix
  * Alternative measures of similarity
    * Partial correlations
    * Other Kernels
    * [Hybrid High-order Functional Connectivity Networks Using Resting-state Functional MRI for Mild Cognitive Impairment Diagnosis](https://www.nature.com/articles/s41598-017-06509-0)
    * [Statistical Approaches to Identify Pairwise and High-Order Brain Functional Connectivity Signatures on a Single-Subject Basis](https://www.mdpi.com/2075-1729/13/10/2075)
    * [EAG-RS: A Novel Explainability-guided ROI-Selection Framework for ASD Diagnosis via Inter-regional Relation Learning](https://arxiv.org/pdf/2310.03404.pdf)
      * Has references of alternative FCs
* **Apply Shrinkage or Thresholding of NetMats at the Subject Level**
  * Current: None
* **Create Single NetMat for Each Group**
  * Current: Arithmetic Mean
  * Alternative Means/Metrics
    * Generalized Means (harmonic, geometric, p=?)
    * List of [metrics](https://marco-congedo.github.io/PosDefManifold.jl/dev/introToRiemannianGeometry/)
* **Apply Shrinkage or Thresholding of Covariance at the Group Level**
  * Current: None
  * Alternative
    * OptShrink
    * Singular Value Soft Thresholding + Stein's Unbiased Risk Estimate
* **Discriminant Analysis**
  * Currently FKT/Generalized Eigenvalue Decomposition with $$||w||_2^2$$ regularized
  * Alternatives
    * Maybe some FKT variant after applying Laplacian Eigen Maps as to respect geometry
    * Kernel Discriminatory Analysis
    * Manifold Discriminant Analysis
    * Fisher Discriminant Based Methods
      * Foley Sammon
      * HTC
    * Other regularization methods (would cause different optimization techniques)
    * Cost function of $$argmax_w (C_A-C_B)w$$ where difference is measured on the plane tangent to the manifold defined by the mean of subject FCs
* **Interpretable Components Decomposition**
  * Current: ICA
  * Alternatives
    * Nonlinear ICA or Normalizing Flows (nonvolume preserving)
    * [Free Component Analysis](https://link.springer.com/article/10.1007/s10208-022-09564-w)
    * Dictionary Learning
    * Gaussian Mixture Models
