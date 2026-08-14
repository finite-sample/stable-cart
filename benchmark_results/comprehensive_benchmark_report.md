# Stable CART Benchmark Report

**Generated:** 2026-08-14 00:51:51
**Random Seed:** 42
**Bootstrap Samples:** 20

## Executive Summary

This report compares stable CART methods against standard CART and ensemble baselines 
across multiple datasets. The primary focus is on **out-of-sample prediction variance** 
as a measure of model stability, complemented by standard discrimination metrics.

### Key Findings

- **Average variance reduction:** -38.1% across stable methods
- **Datasets evaluated:** 14
- **Models compared:** 6

- **Best stable method:** CART_Pruned (12.1% average reduction)

## Classification Results

Performance comparison for classification tasks. **Bold** values indicate best performance 
within each dataset.

| Dataset | BootstrapVariancePenalized | CART | CART_Pruned | LessGreedyHybrid | RandomForest | RobustPrefixHonest |
|---|---|---|---|---|---|---|
| **Pred Variance** |  |  |  |  |  |  |
| breast_cancer | 0.031 | 0.016 | 0.017 | 0.036 | **0.001** | 0.036 |
| digits_binary | 0.007 | 0.004 | 0.004 | 0.007 | **2.40e-04** | 0.006 |
| digits_multiclass | N/A | 0.027 | 0.022 | N/A | **0.003** | N/A |
| iris | N/A | 6.81e-04 | **5.74e-04** | N/A | 0.008 | N/A |
| wine | N/A | 0.006 | **0.004** | N/A | 0.004 | N/A |
| |  |  |  |  |  |  |
| **Accuracy** |  |  |  |  |  |  |
| breast_cancer | 0.924 | 0.918 | 0.918 | 0.912 | **0.930** | 0.871 |
| digits_binary | 0.989 | 0.987 | 0.985 | 0.989 | **0.998** | 0.991 |
| digits_multiclass | N/A | 0.752 | 0.739 | N/A | **0.920** | N/A |
| iris | N/A | 0.889 | 0.889 | N/A | **0.933** | N/A |
| wine | N/A | 0.889 | 0.889 | N/A | **0.963** | N/A |
| |  |  |  |  |  |  |
| **F1-Macro** |  |  |  |  |  |  |
| breast_cancer | 0.917 | 0.913 | 0.913 | 0.905 | **0.925** | 0.856 |
| digits_binary | 0.970 | 0.963 | 0.960 | 0.970 | **0.995** | 0.974 |
| digits_multiclass | N/A | 0.742 | 0.731 | N/A | **0.921** | N/A |
| iris | N/A | 0.889 | 0.889 | N/A | **0.933** | N/A |
| wine | N/A | 0.897 | 0.897 | N/A | **0.964** | N/A |
| |  |  |  |  |  |  |
| **AUC** |  |  |  |  |  |  |
| breast_cancer | 0.944 | 0.969 | 0.926 | 0.930 | **0.984** | 0.844 |
| digits_binary | 0.993 | 0.996 | 0.983 | 0.992 | **0.999** | 0.994 |
| digits_multiclass | N/A | 0.940 | 0.928 | N/A | **0.995** | N/A |
| iris | N/A | 0.944 | 0.944 | N/A | **0.990** | N/A |
| wine | N/A | 0.976 | 0.967 | N/A | **0.999** | N/A |
| |  |  |  |  |  |  |


## Regression Results

Performance comparison for regression tasks. **Bold** values indicate best performance 
within each dataset.

| Dataset | BootstrapVariancePenalized | CART | CART_Pruned | LessGreedyHybrid | RandomForest | RobustPrefixHonest |
|---|---|---|---|---|---|---|
| **Pred Variance** |  |  |  |  |  |  |
| california_housing | 0.111 | 0.077 | 0.061 | 0.032 | **0.016** | 0.106 |
| diabetes | 896.375 | 822.824 | 822.824 | 849.827 | **211.573** | 815.503 |
| friedman1 | 2.645 | 2.381 | 2.383 | 2.507 | **0.400** | 2.742 |
| friedman2 | 12722.731 | 1972.885 | 1972.885 | 3753.311 | **405.351** | 11846.308 |
| friedman3 | 0.036 | 0.116 | 0.048 | 0.063 | 0.020 | **0.018** |
| heteroscedastic | 0.913 | 0.918 | 0.904 | 0.910 | **0.177** | 0.842 |
| high_dim_sparse | 2.528 | 2.537 | 2.537 | 2.538 | **0.380** | 2.186 |
| quadrant_interaction | 0.433 | 0.072 | 0.055 | 0.075 | **0.023** | 0.437 |
| xor_nonlinear | 2.264 | 2.575 | 2.563 | 2.222 | **0.258** | 1.686 |
| |  |  |  |  |  |  |
| **MSE** |  |  |  |  |  |  |
| california_housing | 0.539 | 0.478 | 0.598 | **0.364** | 0.411 | 0.545 |
| diabetes | 3285.084 | 3177.665 | 3177.665 | 2966.157 | **2707.559** | 3391.053 |
| friedman1 | 9.334 | 7.141 | 7.141 | 8.565 | **5.463** | 9.253 |
| friedman2 | 26388.533 | 2317.675 | 2317.675 | 3706.741 | **940.707** | 42319.083 |
| friedman3 | 1.021 | 1.003 | 0.992 | 1.006 | **0.952** | 1.018 |
| heteroscedastic | 5.654 | 4.460 | 4.451 | 5.091 | **4.038** | 5.523 |
| high_dim_sparse | 8.421 | 5.161 | 5.161 | 5.321 | **3.481** | 8.005 |
| quadrant_interaction | 0.754 | 0.354 | 0.385 | 0.432 | **0.312** | 0.619 |
| xor_nonlinear | 10.095 | 9.665 | 9.665 | 10.016 | **6.092** | 9.919 |
| |  |  |  |  |  |  |
| **R²** |  |  |  |  |  |  |
| california_housing | 0.589 | 0.636 | 0.545 | **0.723** | 0.687 | 0.585 |
| diabetes | 0.391 | 0.411 | 0.411 | 0.451 | **0.498** | 0.372 |
| friedman1 | 0.637 | 0.722 | 0.722 | 0.667 | **0.788** | 0.640 |
| friedman2 | 0.815 | 0.984 | 0.984 | 0.974 | **0.993** | 0.704 |
| friedman3 | 0.021 | 0.038 | 0.049 | 0.036 | **0.087** | 0.024 |
| heteroscedastic | 0.438 | 0.557 | 0.557 | 0.494 | **0.599** | 0.451 |
| high_dim_sparse | 0.612 | 0.762 | 0.762 | 0.755 | **0.839** | 0.631 |
| quadrant_interaction | 0.950 | 0.977 | 0.975 | 0.972 | **0.979** | 0.959 |
| xor_nonlinear | -0.003 | 0.039 | 0.039 | 0.004 | **0.395** | 0.014 |
| |  |  |  |  |  |  |


## Stability Analysis

Prediction variance reduction compared to CART baseline. Positive values indicate 
more stable predictions (lower variance).

| Model | Avg Variance Reduction (%) | Std Dev | Datasets | Relative to CART |
|-------|---------------------------|---------|----------|------------------|
| CART_Pruned | **+12.1** | ±17.8 | 14 | 0.88× |
| CART | 0.00e+00 | ±0.00e+00 | 14 | 1.00× |
| RandomForest | -2.8 | ±306.2 | 14 | 1.03× |
| LessGreedyHybrid | -16.4 | ±56.8 | 11 | 1.16× |
| RobustPrefixHonest | -98.7 | ±205.9 | 11 | 1.99× |
| BootstrapVariancePenalized | -108.1 | ±208.9 | 11 | 2.08× |


## Model Characteristics

Computational and structural properties of the models.

| Model | Avg Leaves | Avg Fit Time (s) |
|---|---|---|---|---|
| BootstrapVariancePenalized | nan ± nan | 1.22 ± 1.43 |
| CART | 32 ± 21 | 0.01 ± 0.02 |
| CART_Pruned | 22 ± 21 | 0.01 ± 0.01 |
| LessGreedyHybrid | nan ± nan | 0.38 ± 0.57 |
| RandomForest | nan ± nan | 0.53 ± 0.98 |
| RobustPrefixHonest | nan ± nan | 0.87 ± 0.97 |


## Dataset Insights

### Best Datasets for Stability Improvements
- **digits_binary**: 94.3% reduction with RandomForest
- **breast_cancer**: 91.2% reduction with RandomForest
- **xor_nonlinear**: 90.0% reduction with RandomForest

### Challenging Datasets
- **iris**: Limited improvement (15.6%)
- **wine**: Limited improvement (30.2%)
- **quadrant_interaction**: Limited improvement (68.0%)

### Most Consistent Methods
- **CART_Pruned**: 12.1% ± 17.8% improvement
- **LessGreedyHybrid**: -16.4% ± 56.8% improvement


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
- **CART_Pruned**: 12.1% average variance reduction
- **LessGreedyHybrid**: -16.4% average variance reduction
- **BootstrapVariancePenalized**: -108.1% average variance reduction
- **RobustPrefixHonest**: -98.7% average variance reduction

### Trade-offs
- Stable methods may have slightly higher computational cost
- Accuracy differences are typically small (< 5%)
- Stability benefits are most apparent with limited training data

---

*Report generated by stable-cart benchmark suite v0.1.0*
