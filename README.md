# [Direct Fitting of GMMs project website](https://leonidk.github.io/direct_gmm/)
This is the source code and project history for the following publication

**Direct Fitting of Gaussian Mixture Models** by Leonid Keselman and Martial Hebert ([arXiv version here](https://arxiv.org/abs/1904.05537))

## Overview
Almost all files used in the development and testing of this project are in this folder. The data files for the Stanford Bunny is included in `bunny`. 

* `mixture` contains the modifed version of scikit-learn with the proposed techniques. 
* `gmm_fit.py` and `gmm_fit2.py` contain the two sets of the bunny likelihood experiments
* `registration_test.py` contains the mesh registration (P2D) experiments
* Files with `_extra` are usually just copies for non-Stanford Bunny experiments
* `gen_gmm.ipynb` and `gen_gmm_mine.ipynb` generate GMM models from the TUM dataset, with and without uncertainty models
* `reg_results.ipynb` performs D2D registration between the GMM models built from the TUM dataset. 
