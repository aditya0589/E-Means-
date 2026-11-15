# Implementation Decisions and Architecture

This document outlines the architectural patterns, software engineering decisions, and optimization strategies adopted across the implementation of Traditional K-Means, Optimized K-Means, and E-Means.

---

## 1. Unified Estimator Interface

All clustering algorithms inherit from a common abstract base class `BaseClustering` adhering to the standard scikit-learn estimator protocol:

```python
class BaseClustering(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseClustering": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).predict(X)

    @abstractmethod
    def get_centroids(self) -> np.ndarray: ...

    @abstractmethod
    def get_metrics(self) -> dict: ...
```

### Benefits:
1. **Interchangeability:** Every algorithm can be passed to identical benchmarking, cross-validation, and visualization routines without adaptation layers.
2. **Standardized State Inspection:** Centroids, inertia, cluster labels, runtime, and convergence histories are accessed via uniform property and getter conventions.
3. **Out-of-Sample Generalization:** All algorithms support `.predict(X_new)` using the nearest-centroid assignment rule.

---

## 2. Distance Computation Optimization

Euclidean distance squared $\|x - c\|^2 = \|x\|^2 - 2 x \cdot c + \|c\|^2$ is the core primitive across all methods.
- We vectorize pairwise distance calculations using NumPy matrix multiplication:
  $$\|X - C\|^2 = \sum_d X_{i,d}^2 - 2 X C^T + \sum_d C_{j,d}^2$$
  This achieves BLAS-level throughput without allocating memory for 3D tensors $(n, k, d)$.
- Numerical clipping `np.maximum(dist_sq, 0.0)` prevents precision-induced negative squared distances.

---

## 3. Empty Cluster & Degenerate State Handling

In Lloyd's and Elkan's algorithms, an empty cluster (zero points assigned in an iteration) can lead to division by zero:
- **Decision:** Rather than failing or producing NaNs, we identify the sample point in $X$ having the maximum residual distance to its currently assigned center ($x^* = \arg\max_i \min_j \|x_i - c_j\|^2$) and reassign the empty centroid to this location.
- **Scientific Justification:** This maximizes immediate error reduction and guarantees valid partitions of cardinality $k$.

In E-Means, empty or dying clusters are handled natively via the **proportional donor splitting** protocol described in Section 4.

---

## 4. E-Means Specific Engineering Decisions

### A. Scaled Batch Size and Learning Rate
- While the paper evaluated massive embedding datasets ($n \in [10^7, 10^{10}]$, $k \in [10^4, 10^7]$) where $B$ ranges in $10^4$–$10^5$, practical benchmarking on standard machine learning benchmark datasets ($n \in [500, 100000]$, $k \in [2, 20]$) requires calibrated scaling bounds:
  $$B = \min\left(n, \max\left(128, 8 \cdot k\right)\right)$$
  $$\alpha_0 = \min\left(1.0, \frac{2 \cdot B}{n}\right)$$
- If the user explicitly sets `batch_size` or `alpha_0`, those values take precedence, allowing controlled ablation studies.

### B. Stable Sufficient Statistics Initialization
- $N$ is initialized to $(B / k)$ for all $k$ clusters, representing balanced initial mass.
- $S$ is initialized to $C_{\text{init}} \odot N[:, \text{None}]$, ensuring initial derived centroids $c_j = S_j / N_j$ match the initial centroids exactly without introducing numerical instability.

### C. Dead-Cluster Revival Timing and donor sampling
- Revival checks run every `revival_interval` batches (default: 5).
- A cluster is flagged if its running count $N_j < \tau \cdot \bar{N}$ with $\tau = 0.1$.
- Donors are sampled with probability proportional to $\max(0, N_d - \tau \bar{N})$, ensuring donor clusters have sufficient mass to split.
- Perturbation vector $\delta \sim \mathcal{N}(0, \sigma^2 I)$ with $\sigma = 0.01 \cdot \text{std}(X)$ ensures sibling centroids branch along separate local trajectories.

### D. Finalization Pass
- A single full-batch Lloyd update is executed after the scheduled epochs. This is measured as part of E-Means total training time to ensure fair, rigorous comparison against full-batch Lloyd and Elkan algorithms.

---

## 5. Memory and Time Profiling Integrity
- Execution time is measured using monotonic clocks (`time.perf_counter()`).
- Initialization time is isolated from training update time.
- Memory consumption is measured using `psutil.Process().memory_info().rss` before and after fitting, tracking peak resident set size delta.
- All random processes accept explicit `np.random.RandomState` or integer seeds for deterministic reproducibility.
