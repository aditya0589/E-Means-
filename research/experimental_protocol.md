# Experimental Protocol and Benchmarking Standards

This document establishes the scientific protocol for benchmarking Traditional K-Means, Optimized K-Means (K-Means++ and Elkan's Algorithm), and E-Means across synthetic and real-world datasets.

---

## 1. Benchmarking Objectives
The experiments are designed to address the following core research questions:
1. **Convergence Speed & Epoch Efficiency:** Does E-Means reach Lloyd's final inertia in significantly fewer effective epochs/passes over the dataset?
2. **Clustering Quality:** Does E-Means achieve comparable or superior Within-Cluster Sum of Squares (Inertia), Silhouette Score, and Davies-Bouldin index relative to Lloyd's algorithm and K-Means++?
3. **Computational Scalability:** How do execution runtime, memory footprint, and distance calculations scale as a function of sample size $n$, dimensionality $d$, and cluster count $k$?
4. **Initialization Sensitivity:** How robust is each algorithm across multiple random seeds, and how does K-Means++ initialization impact stability compared to random seeding?
5. **Noise & Overlap Tolerance:** How do the algorithms perform in the presence of overlapping clusters, anisotropic variance, and background noise?

---

## 2. Experimental Suite Overview

| Experiment ID | Experiment Name | Independent Variable | Controlled Variables | Metrics Tracked |
|---|---|---|---|---|
| **Exp 1** | Basic Performance Comparison | Algorithm Variant | Dataset, Preprocessing, $k$, Seed | Inertia, Silhouette, DB, CH, ARI, Time |
| **Exp 2** | Convergence Trajectory | Training Iterations / Elapsed Time | Same dataset & $k$ | Inertia vs. Iterations, Inertia vs. Time |
| **Exp 3** | Initialization Sensitivity | Random Seeds (10+ seeds) | Dataset, Algorithm | Mean, Std, Boxplots of Inertia & ARI |
| **Exp 4** | Sample Size Scalability | Sample Size $n \in [500, 100000]$ | $d=10, k=5$ | Runtime (s), Memory Delta (MB) |
| **Exp 5** | Dimensionality Scalability | Features $d \in [2, 100]$ | $n=5000, k=5$ | Runtime (s), Distance Computations |
| **Exp 6** | Cluster Count Scalability | Clusters $k \in [2, 3, 5, 8, 10, 15, 20]$ | $n=5000, d=10$ | Runtime (s), Inertia, Silhouette |
| **Exp 7** | Noise & Cluster Overlap | Cluster Std Deviation $\sigma \in [0.5, 3.0]$ | $n=2000, k=4$ | Silhouette, ARI, Misclassification |
| **Exp 8** | Repeated Statistical Trials | 10 Random Seeds | All datasets & algorithms | Paired t-tests, Wilcoxon rank-sum |

---

## 3. Dataset Specifications

### Synthetic Datasets
1. **Gaussian Blobs:** Controlled spherical isotropic clusters with ground-truth labels.
2. **Anisotropic Blobs:** Linear transformations applied to Gaussian blobs to evaluate sensitivity to non-spherical clusters.
3. **Noisy Concentric Circles & Moons:** Non-linearly separable manifolds to test centroid positioning behavior.
4. **High-Dimensional Gaussians:** Scalable synthetic benchmarks ($d = 50, 100$) for scalability sweeps.

### Real-World Datasets
1. **Iris Dataset:** 150 samples, 4 features, 3 classes (classic taxonomic benchmark).
2. **Wine Dataset:** 178 samples, 13 chemical features, 3 classes.
3. **Breast Cancer Wisconsin:** 569 samples, 30 continuous clinical features, 2 classes.
4. **Optical Recognition of Handwritten Digits:** 1,797 samples, 64 pixel features, 10 classes.

### Preprocessing Protocol
- All numerical datasets undergo z-score standardization (`StandardScaler`) prior to clustering.
- Ground-truth labels are **strictly withheld** during fitting and are utilized solely for computing external validity metrics (Adjusted Rand Index, Normalized Mutual Information).

---

## 4. Evaluation Metrics

### Unsupervised Clustering Quality Metrics
1. **Inertia / Within-Cluster Sum of Squares (WCSS):**
   $$\text{Inertia} = \sum_{j=1}^k \sum_{x \in C_j} \|x - c_j\|_2^2$$
2. **Silhouette Coefficient:** Measures cohesion versus separation, bounded in $[-1, 1]$:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
3. **Davies-Bouldin Index:** Measures average similarity between each cluster and its most similar counterpart (lower is better):
   $$DB = \frac{1}{k} \sum_{i=1}^k \max_{j \ne i} \left( \frac{\sigma_i + \sigma_j}{\|c_i - c_j\|} \right)$$
4. **Calinski-Harabasz Index (Variance Ratio Criterion):** Ratio of between-cluster dispersion to within-cluster dispersion (higher is better).

### External Validation Metrics (with Ground Truth)
1. **Adjusted Rand Index (ARI):** Chance-corrected measure of pair agreements, bounded in $[-1, 1]$.
2. **Normalized Mutual Information (NMI):** Information-theoretic measure of shared mutual information, bounded in $[0, 1]$.
3. **Homogeneity and Completeness:** V-measure components verifying single-class clusters and all-class assignments.

### Computational Performance Metrics
1. **Total Execution Time (seconds):** Wall-clock duration of entire execution.
2. **Initialization Time (seconds):** Wall-clock duration of centroid initialization.
3. **Training Time (seconds):** Wall-clock duration of optimization loop.
4. **Peak Memory Consumption (MB):** RSS memory delta measured via `psutil`.
5. **Iterations / Epoch Passes:** Number of full data passes or mini-batch iterations.

---

## 5. Statistical Rigor Protocol
- Every non-deterministic configuration is repeated over a minimum of 10 independent trials with explicit seeds $\{42, 43, \dots, 51\}$.
- Aggregate summaries report mean, standard deviation, median, and 95% confidence intervals.
- Differences in inertia and runtime are evaluated for statistical significance using two-sided paired Student's t-tests and non-parametric Wilcoxon signed-rank tests.
