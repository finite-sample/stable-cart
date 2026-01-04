# Stable CART Benchmark Report

**Generated:** 2025-10-28 22:19:59
**Random Seed:** 42
**Bootstrap Samples:** 20

## Executive Summary

This report compares stable CART methods against standard CART and ensemble baselines 
across multiple datasets. The primary focus is on **out-of-sample prediction variance** 
as a measure of model stability, complemented by standard discrimination metrics.

### Key Findings

- **Average variance reduction:** 60.1% across stable methods
- **Datasets evaluated:** 2
- **Models compared:** 6

- **Best stable method:** BootstrapVariancePenalized (98.8% average reduction)

## Classification Results

Performance comparison for classification tasks. **Bold** values indicate best performance 
within each dataset.

| Dataset | BootstrapVariancePenalized | CART | CART_Pruned | LessGreedyHybrid | RandomForest | RobustPrefixHonest |
|---|---|---|---|---|---|---|
| **Pred Variance** |  |  |  |  |  |  |
| breast_cancer | **2.31e-04** | 0.016 | 0.019 | 0.024 | 0.001 | 2.75e-04 |
| |  |  |  |  |  |  |
| **Accuracy** |  |  |  |  |  |  |
| breast_cancer | 0.626 | 0.918 | 0.918 | **0.988** | 0.930 | 0.626 |
| |  |  |  |  |  |  |
| **F1-Macro** |  |  |  |  |  |  |
| breast_cancer | 0.385 | 0.913 | 0.913 | **0.987** | 0.925 | 0.385 |
| |  |  |  |  |  |  |
| **AUC** |  |  |  |  |  |  |
| breast_cancer | 0.500 | 0.969 | 0.926 | **0.995** | 0.984 | 0.500 |
| |  |  |  |  |  |  |


## Regression Results

Performance comparison for regression tasks. **Bold** values indicate best performance 
within each dataset.

| Dataset | BootstrapVariancePenalized | CART | CART_Pruned | LessGreedyHybrid | RandomForest | RobustPrefixHonest |
|---|---|---|---|---|---|---|
| **Pred Variance** |  |  |  |  |  |  |
| friedman1 | **0.023** | 2.381 | 2.382 | **0.023** | 0.372 | 0.186 |
| |  |  |  |  |  |  |
| **MSE** |  |  |  |  |  |  |
| friedman1 | 25.728 | 7.141 | 7.141 | 25.728 | **5.463** | 25.710 |
| |  |  |  |  |  |  |
| **R²** |  |  |  |  |  |  |
| friedman1 | -6.97e-04 | 0.722 | 0.722 | -6.97e-04 | **0.788** | -8.32e-06 |
| |  |  |  |  |  |  |


## Stability Analysis

Prediction variance reduction compared to CART baseline. Positive values indicate 
more stable predictions (lower variance).

| Model | Avg Variance Reduction (%) | Std Dev | Datasets | Relative to CART |
|-------|---------------------------|---------|----------|------------------|
| BootstrapVariancePenalized | **+98.8** | ±0.3 | 2 | 0.01× |
| RobustPrefixHonest | **+95.2** | ±4.3 | 2 | 0.05× |
| RandomForest | **+87.9** | ±5.0 | 2 | 0.12× |
| LessGreedyHybrid | **+25.3** | ±104.3 | 2 | 0.75× |
| CART | 0.00e+00 | ±0.00e+00 | 2 | 1.00× |
| CART_Pruned | -6.7 | ±9.4 | 2 | 1.07× |


## Model Characteristics

Computational and structural properties of the models.

| Model | Avg Leaves | Avg Fit Time (s) |
|---|---|---|---|---|
| BootstrapVariancePenalized | nan ± nan | 0.06 ± 0.08 |
| CART | 30 ± 35 | 0.01 ± 0.01 |
| CART_Pruned | 28 ± 36 | 0.01 ± 0.00 |
| LessGreedyHybrid | nan ± nan | 0.05 ± 0.05 |
| RandomForest | nan ± nan | 0.32 ± 0.35 |
| RobustPrefixHonest | nan ± nan | 0.05 ± 0.06 |


## Dataset Insights

### Best Datasets for Stability Improvements
- **friedman1**: 99.0% reduction with LessGreedyHybrid
- **breast_cancer**: 98.6% reduction with BootstrapVariancePenalized

### Challenging Datasets
- **breast_cancer**: Limited improvement (98.6%)
- **friedman1**: Limited improvement (99.0%)

### Most Consistent Methods
- **BootstrapVariancePenalized**: 98.8% ± 0.3% improvement
- **RobustPrefixHonest**: 95.3% ± 4.3% improvement
- **RandomForest**: 87.9% ± 5.0% improvement


## Methodology

### Stability Measurement
- **Bootstrap prediction variance**: Models trained on bootstrap samples of training data
- **Test set consistency**: All models evaluated on same held-out test set
- **Bootstrap samples**: 20 per model

### Datasets
- **Selected datasets**: friedman1, breast_cancer...
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
- **RandomForest**: 87.9% average variance reduction
- **LessGreedyHybrid**: 25.3% average variance reduction
- **BootstrapVariancePenalized**: 98.8% average variance reduction
- **RobustPrefixHonest**: 95.3% average variance reduction

### Trade-offs
- Stable methods may have slightly higher computational cost
- Accuracy differences are typically small (< 5%)
- Stability benefits are most apparent with limited training data

---

*Report generated by stable-cart benchmark suite v0.1.0*
