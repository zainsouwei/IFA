# Algorithm Questions

* [ ] Are the [spatial filters](https://github.com/zainsouwei/ICASPADE/blob/21adaa891aab69852804d4ae05bb6f2460be63d4/simulate\_time.py#L81C1-L83C6) always orthogonal with our data?
* [ ] Are SPADE Filters orthogonal
* [ ] How to best generalize to individual/no reference group case
* [x] LDA vs FKT vs LDA/FKT vs Bhattacharyya distance/Fischer Discriminant Criteria
* [ ] Bhattacharyya distance
* [ ] Computational cost of running ICA number of subject tiems vs once and then doing dual regression. Maybe doesnt matter cause can donit for subject of interest.
* [ ] Do the discrimination before ICA. Which means we are looking for low order discrimination. Which then means is the benefit of CC ICA that we look for higher order discrimination? How does CC-ICA performance compare/enforcing learning rule in the criterion
* [ ] How should we be denoising? This method seems prone to site effects
* [ ] What happens if all the discriminatory information is represented by higher-order information? I think it still might work in the case you view it as a linear approximation
* [ ] What is our extension of SPADE/what part of the pipeline? What are the works related to it? How does it differ/what are the benefits of this method?
* [ ] How to best generalize to individual/no reference group case?
* [ ] Is the best way to generalize to no reference group case one v all, pairwise comparisons, or add all separate covariance matrices?
* [ ] Look into group FKT paper
* [ ] What is the best distance similarity measure to form the within-subject FC? What is the best similarity measure to measure subject similarity? Is it right to say the parcellated covariance matrix is implicitly based on Euclidean distance and the SPADE algorithm is related to Bhattacharya distance since it uses two different covariance matrices? Are there other kernels we can use in formulating the parcellated covariance matrices and in the FKT/dissimilarity calculation? In forming the covariance matrices it is important to ask what space the vectors/observations live in. For doing the dissimilarity analysis it is important to figure out what space the covariance matrices live in. Should we relate the vectors/observations based on mean, polynomials, variance, and higher-order statistics? the same question applies to relating the separate covariance matrices
