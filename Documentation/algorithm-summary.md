# Algorithm Summary

* When running SPADE, I used [abs() on the filters](https://github.com/zainsouwei/ICASPADE/blob/070a5d3ab2e6b09d2aee03aa106db90f8a5f71f2/simulate\_time.py#L83). Otherwise, the sign invariance of EVD caused structured symmetry within a group's cluster if SPADE was run on the subject level
* For the subject-specific filter case, I am just using the [filters rather than a dual regression or a projection](https://github.com/zainsouwei/ICASPADE/blob/21adaa891aab69852804d4ae05bb6f2460be63d4/simulate\_time.py#L280C1-L284C49).
* Using dual regression or projection should yield the same results, this is just the most accurate and computationally efficient
* For the dual regression, [I am using the inner product rather than the full pseudoinverse](https://github.com/zainsouwei/ICASPADE/blob/21adaa891aab69852804d4ae05bb6f2460be63d4/simulate\_time.py#L100C1-L105C44). Technically these are equal if the spatial filters are orthogonal which by the spectral theorem holds true when the matrix multiplication of the inverse of one of the covariance matrices times the other covariance matrix is hermitian.
* The [second filter](https://github.com/zainsouwei/ICASPADE/blob/21adaa891aab69852804d4ae05bb6f2460be63d4/simulate\_time.py#L82C12-L82C83), which says how the control group discriminates from the non-control group or the subject group, is not too stable. This is because, during the dual regression, this regressor does not actually correlate with the subject so if there is little noise then these regressors will model anything that has some value. How well this filter is expressed in the dual regression is related to [the level of noise](https://github.com/zainsouwei/ICASPADE/blob/21adaa891aab69852804d4ae05bb6f2460be63d4/simulate\_time.py#L256). Additionally, this filter does not add much to the variance captured between non-control groups so it is always not included in the [scree plots](https://github.com/zainsouwei/ICASPADE/blob/21adaa891aab69852804d4ae05bb6f2460be63d4/simulate\_time.py#L290).
* The level of accuracy of the clustering is dependent on the frequency of the independent components in the subjects which is controlled by the number of groups, [number of shared components, number of filters, max number of places the filters occur, and the number of parcels/voxels](https://github.com/zainsouwei/ICASPADE/blob/21adaa891aab69852804d4ae05bb6f2460be63d4/simulate\_time.py#L250C3-L257C1)
* The reason we do dual in SPADE ICA pipeline is because we are working with VEV' covairance matrix initially where in classic ICA we have the data and can project onto SVD components of choosing form UEV'
*   All of these are valid/useful
*   However our main pipeline/generation of the independent filter analysis is
* When adding future extensions/variants, the main changes should be in the
* Discrimination - Current FKT
* Kernel Discriminatory Analysis
* Kernel PCA
* Explanation - Current ICA
* PROFUMO
* Dictionary Learning
