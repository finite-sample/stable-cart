# Stable CART Benchmark Report

**Generated:** 2026-08-14 16:37:32
**Random Seed:** 42
**Bootstrap Samples:** 20

## Executive Summary

This report compares stable CART methods against standard CART and ensemble baselines 
across multiple datasets. The primary focus is on **out-of-sample prediction variance** 
as a measure of model stability, complemented by standard discrimination metrics.

### Key Findings

- **Average variance reduction:** -3.2% across stable methods
- **Datasets evaluated:** 14
- **Models compared:** 8

- **Best stable method:** StableTree (16.8% average reduction)

## Classification Results

Performance comparison for classification tasks. **Bold** values indicate best performance 
within each dataset.

| Dataset | BootstrapVariancePenalized | CART | CART_Pruned | CentroidTree | LessGreedyHybrid | RandomForest | RobustPrefixHonest | StableTree |
|---|---|---|---|---|---|---|---|---|
| **Pred Variance** |  |  |  |  |  |  |  |  |
| breast_cancer | 0.033 | 0.016 | 0.017 | 0.020 | 0.038 | **0.001** | 0.033 | 0.019 |
| digits_binary | 0.004 | 0.004 | 0.004 | 0.004 | 0.006 | **2.40e-04** | 0.007 | 0.005 |
| digits_multiclass | N/A | 0.027 | 0.022 | 0.034 | N/A | 0.003 | N/A | **8.20e-04** |
| iris | N/A | 6.81e-04 | 5.74e-04 | 7.55e-04 | N/A | 0.008 | N/A | **0.00e+00** |
| wine | N/A | 0.006 | 0.004 | **0.002** | N/A | 0.004 | N/A | 0.005 |
| |  |  |  |  |  |  |  |  |
| **Accuracy** |  |  |  |  |  |  |  |  |
| breast_cancer | 0.912 | 0.918 | 0.918 | 0.918 | 0.912 | **0.930** | 0.871 | 0.626 |
| digits_binary | 0.985 | 0.987 | 0.985 | 0.985 | 0.989 | **0.998** | 0.985 | 0.974 |
| digits_multiclass | N/A | 0.752 | 0.739 | 0.757 | N/A | **0.920** | N/A | 0.393 |
| iris | N/A | 0.889 | 0.889 | **0.933** | N/A | **0.933** | N/A | 0.889 |
| wine | N/A | 0.889 | 0.889 | 0.870 | N/A | **0.963** | N/A | 0.907 |
| |  |  |  |  |  |  |  |  |
| **F1-Macro** |  |  |  |  |  |  |  |  |
| breast_cancer | 0.905 | 0.913 | 0.913 | 0.913 | 0.905 | **0.925** | 0.856 | 0.385 |
| digits_binary | 0.960 | 0.963 | 0.960 | 0.960 | 0.970 | **0.995** | 0.959 | 0.926 |
| digits_multiclass | N/A | 0.742 | 0.731 | 0.753 | N/A | **0.921** | N/A | 0.370 |
| iris | N/A | 0.889 | 0.889 | **0.933** | N/A | **0.933** | N/A | 0.889 |
| wine | N/A | 0.897 | 0.897 | 0.880 | N/A | **0.964** | N/A | 0.909 |
| |  |  |  |  |  |  |  |  |
| **AUC** |  |  |  |  |  |  |  |  |
| breast_cancer | 0.899 | 0.969 | 0.926 | 0.965 | 0.899 | **0.984** | 0.844 | 0.500 |
| digits_binary | 0.991 | 0.996 | 0.983 | 0.985 | 0.983 | **0.999** | 0.982 | 0.965 |
| digits_multiclass | N/A | 0.940 | 0.928 | 0.933 | N/A | **0.995** | N/A | N/A |
| iris | N/A | 0.944 | 0.944 | 0.961 | N/A | **0.990** | N/A | N/A |
| wine | N/A | 0.976 | 0.967 | 0.932 | N/A | **0.999** | N/A | N/A |
| |  |  |  |  |  |  |  |  |


## Regression Results

Performance comparison for regression tasks. **Bold** values indicate best performance 
within each dataset.

| Dataset | BootstrapVariancePenalized | CART | CART_Pruned | CentroidTree | LessGreedyHybrid | RandomForest | RobustPrefixHonest | StableTree |
|---|---|---|---|---|---|---|---|---|
| **Pred Variance** |  |  |  |  |  |  |  |  |
| california_housing | 0.085 | 0.077 | 0.061 | 0.095 | 0.032 | **0.016** | 0.028 | 0.071 |
| diabetes | 819.902 | 822.824 | 822.824 | 1026.901 | 823.973 | **211.573** | 845.999 | 796.857 |
| friedman1 | 2.730 | 2.381 | 2.383 | 2.974 | 2.652 | **0.400** | 2.606 | 2.350 |
| friedman2 | 3496.071 | 1972.885 | 1972.885 | 2288.108 | 3894.245 | **405.351** | 4549.673 | 2008.510 |
| friedman3 | 0.070 | 0.116 | 0.048 | 0.134 | 0.059 | **0.020** | 0.043 | 0.101 |
| heteroscedastic | 0.936 | 0.918 | 0.904 | 1.065 | 0.984 | **0.177** | 0.788 | 0.856 |
| high_dim_sparse | 2.592 | 2.537 | 2.537 | 2.945 | 2.513 | **0.380** | 2.313 | 2.371 |
| quadrant_interaction | 0.076 | 0.072 | 0.055 | 0.092 | 0.095 | **0.023** | 0.087 | 0.065 |
| xor_nonlinear | 1.509 | 2.575 | 2.563 | 2.576 | 1.714 | **0.258** | 0.801 | 2.189 |
| |  |  |  |  |  |  |  |  |
| **MSE** |  |  |  |  |  |  |  |  |
| california_housing | 0.498 | 0.478 | 0.598 | 0.471 | **0.363** | 0.411 | 0.364 | 0.481 |
| diabetes | 2972.947 | 3177.665 | 3177.665 | 3168.119 | 2966.157 | **2707.559** | 3296.489 | 3410.149 |
| friedman1 | 8.791 | 7.141 | 7.141 | 7.858 | 9.203 | **5.463** | 8.746 | 7.124 |
| friedman2 | 4430.907 | 2317.675 | 2317.675 | 3123.242 | 4002.204 | **940.707** | 6025.061 | 2427.998 |
| friedman3 | 0.988 | 1.003 | 0.992 | 1.096 | 0.988 | **0.952** | 1.004 | 1.012 |
| heteroscedastic | 5.024 | 4.460 | 4.451 | 5.182 | 5.026 | **4.038** | 4.777 | 4.467 |
| high_dim_sparse | 6.725 | 5.161 | 5.161 | 6.368 | 5.321 | **3.481** | 5.543 | 5.475 |
| quadrant_interaction | 0.427 | 0.354 | 0.385 | 0.389 | 0.486 | **0.312** | 0.443 | 0.348 |
| xor_nonlinear | 10.013 | 9.665 | 9.665 | 11.079 | 10.026 | **6.092** | 9.894 | 7.815 |
| |  |  |  |  |  |  |  |  |
| **R²** |  |  |  |  |  |  |  |  |
| california_housing | 0.621 | 0.636 | 0.545 | 0.641 | **0.723** | 0.687 | 0.723 | 0.634 |
| diabetes | 0.449 | 0.411 | 0.411 | 0.413 | 0.451 | **0.498** | 0.389 | 0.368 |
| friedman1 | 0.658 | 0.722 | 0.722 | 0.694 | 0.642 | **0.788** | 0.660 | 0.723 |
| friedman2 | 0.969 | 0.984 | 0.984 | 0.978 | 0.972 | **0.993** | 0.958 | 0.983 |
| friedman3 | 0.053 | 0.038 | 0.049 | -0.051 | 0.052 | **0.087** | 0.037 | 0.029 |
| heteroscedastic | 0.501 | 0.557 | 0.557 | 0.485 | 0.500 | **0.599** | 0.525 | 0.556 |
| high_dim_sparse | 0.690 | 0.762 | 0.762 | 0.706 | 0.755 | **0.839** | 0.744 | 0.748 |
| quadrant_interaction | 0.972 | 0.977 | 0.975 | 0.974 | 0.968 | **0.979** | 0.971 | 0.977 |
| xor_nonlinear | 0.005 | 0.039 | 0.039 | -0.101 | 0.003 | **0.395** | 0.017 | 0.223 |
| |  |  |  |  |  |  |  |  |


## Stability Analysis

Prediction variance reduction compared to CART baseline. Positive values indicate 
more stable predictions (lower variance).

| Model | Avg Variance Reduction (%) | Std Dev | Datasets | Relative to CART |
|-------|---------------------------|---------|----------|------------------|
| StableTree | **+16.8** | ±36.0 | 14 | 0.83× |
| CART_Pruned | **+12.1** | ±17.8 | 14 | 0.88× |
| CART | 0.00e+00 | ±0.00e+00 | 14 | 1.00× |
| RandomForest | -2.8 | ±306.2 | 14 | 1.03× |
| RobustPrefixHonest | -12.1 | ±68.7 | 11 | 1.12× |
| CentroidTree | -12.5 | ±23.6 | 14 | 1.12× |
| BootstrapVariancePenalized | -13.4 | ±44.6 | 11 | 1.13× |
| LessGreedyHybrid | -18.1 | ±60.5 | 11 | 1.18× |


## Model Characteristics

Computational and structural properties of the models.

| Model | Avg Leaves | Avg Fit Time (s) |
|---|---|---|---|---|
| BootstrapVariancePenalized | nan ± nan | 0.30 ± 0.33 |
| CART | 32 ± 21 | 0.01 ± 0.01 |
| CART_Pruned | 22 ± 21 | 0.01 ± 0.01 |
| CentroidTree | nan ± nan | 0.09 ± 0.13 |
| LessGreedyHybrid | nan ± nan | 0.30 ± 0.37 |
| RandomForest | nan ± nan | 0.32 ± 0.53 |
| RobustPrefixHonest | nan ± nan | 0.23 ± 0.28 |
| StableTree | nan ± nan | 0.29 ± 0.37 |


## Dataset Insights

### Best Datasets for Stability Improvements
- **iris**: 100.0% reduction with StableTree
- **digits_multiclass**: 96.9% reduction with StableTree
- **digits_binary**: 94.3% reduction with RandomForest

### Challenging Datasets
- **wine**: Limited improvement (63.6%)
- **quadrant_interaction**: Limited improvement (68.0%)
- **diabetes**: Limited improvement (74.3%)

### Most Consistent Methods
- **CART_Pruned**: 12.1% ± 17.8% improvement
- **StableTree**: 16.8% ± 36.0% improvement


## Methodology

### Stability Measurement
- **Bootstrap prediction variance**: Models trained on bootstrap samples of training data
- **Test set consistency**: All models evaluated on same held-out test set
- **Bootstrap samples**: 20 per model

### Datasets
- **Selected datasets**: friedman1, friedman2, friedman3, quadrant_interaction, high_dim_sparse, heteroscedastic, xor_nonlinear, california_housing, diabetes, breast_cancer, wine, iris, digits_binary, digits_multiclass...
- **Train/test split**: 70/30 with stratification for classification
- **Feature standardization**: Applied to real-world datasets

### Models
- **CART**: Standard sklearn DecisionTreeRegressor/Classifier
- **CART_Pruned**: Cost-complexity pruning with CV-selected alpha
- **RandomForest**: 100-tree ensemble baseline
- **LessGreedyHybrid**: Honest splits + lookahead + oblique root
- **BootstrapVariancePenalized**: Explicit variance penalty in splitting
- **RobustPrefixHonest**: Robust prefix + honest leaves (classification)

## Recommendations

### When to Use Stable CART Methods

**Use stable methods when:**
- Prediction consistency is more important than marginal accuracy gains
- Model will be retrained frequently with new data
- Predictions are used for critical decision-making requiring reliability

**Recommended stable methods:**
- **RandomForest**: -2.8% average variance reduction
- **StableTree**: 16.8% average variance reduction
- **CART_Pruned**: 12.1% average variance reduction
- **LessGreedyHybrid**: -18.1% average variance reduction
- **BootstrapVariancePenalized**: -13.4% average variance reduction
- **RobustPrefixHonest**: -12.1% average variance reduction
- **CentroidTree**: -12.5% average variance reduction

### Trade-offs
- Stable methods may have slightly higher computational cost
- Accuracy differences are typically small (< 5%)
- Stability benefits are most apparent with limited training data

---

*Report generated by stable-cart benchmark suite v0.1.0*
