# Research Paper Summary: e-means

## Paper Details
- **Title:** e-means: outperforming k-means (in one-tenth the epochs)
- **Authors:** Swann Bessa, Pierre-Emmanuel Mazaré, Gergely Szilvasy, Hugo Touvron, David Picard, Matthijs Douze, Hervé Jégou
- **Affiliations:** Meta Fundamental AI Research (FAIR) and École Nationale des Ponts et Chaussées (ENPC)
- **Reference / Publication:** arXiv:2411.09501 [cs.LG] / Open-source release in FAISS (`faiss.contrib.e_means`)
- **Interactive Demonstration:** [e-means.github.io](https://e-means.github.io)

---

## 1. Executive Summary
Lloyd’s classic $k$-means algorithm has been the foundation of vector quantization and clustering since 1957. Despite its ubiquity, Lloyd's algorithm suffers from severe computational bottlenecks on large-scale datasets because each iteration requires a full pass over the entire dataset ($O(n \cdot k \cdot d)$ distance operations per epoch). While classical mini-batch $k$-means (e.g., Sculley, 2010; MacQueen, 1967) mitigates per-step compute, it typically trades off final clustering quality, suffers from severe cluster death, retains noisy early historical statistics indefinitely, and requires delicate manual hyperparameter tuning.

**e-means** revolutionizes clustering by reformulating mini-batch centroid optimization with modern deep learning principles:
1. **Exponential Moving Average (EMA) Statistics:** Tracks cluster counts ($N$) and sum vectors ($S$) using an exponential decay schedule, smoothly forgetting early stale assignments.
2. **Cosine-Scheduled Learning Rate:** Decays the effective update rate $\alpha(t)$ following a half-cosine annealing schedule with a minimum floor $\varepsilon = 0.1$.
3. **Dynamic Dead-Cluster Revival:** Monitors cluster occupancy and revives inactive/dead clusters by proportionally splitting dominant active donor clusters with directional perturbation.
4. **Zero-Tuning Scaling Laws:** Derives empirical power-law formulas relating dataset size ($n$), cluster count ($k$), and target epoch budget ($T$) to optimal mini-batch size ($B$) and initial step size ($\alpha_0$).
5. **Exact Finalization Pass:** Completes clustering with a single full-pass assignment to eliminate any lingering stochastic mini-batch noise.

---

## 2. Key Theoretical & Empirical Insights

### A. The Flaw in Classical Mini-Batch K-Means
- **MacQueen (1967) / Standard Mini-Batch:** Updates centroids as cumulative running averages $c_j \leftarrow \frac{N_j c_j + \sum x}{N_j + n_j}$. Because old points are never forgotten, early misallocations distort centroids throughout training.
- **Gradient Descent K-Means:** Can diverge or oscillate wildly if the step size is misconfigured, as centroids are not guaranteed to remain inside the data convex hull.
- **e-means EMA Solution:** By updating $N_j$ and $S_j$ with decay rate $\alpha(t)$, old assignments are exponentially decayed:
  $$N_j \leftarrow (1-\alpha)N_j + \alpha n_j, \quad S_j \leftarrow (1-\alpha)S_j + \alpha s_j, \quad c_j = \frac{S_j}{N_j}$$
  This naturally bounds centroids within the convex hull of recent data points while steadily refining spatial positioning.

### B. Speed & Epoch Efficiency
- Matches or exceeds the clustering distortion (WCSS / Inertia) of **25 full epochs of Lloyd's algorithm in just 1.2 to 2.6 epochs** (roughly a 10× reduction in data passes).
- Wall-clock speedup of **8×–15× on CPU** compared to standard Faiss / Scikit-Learn implementations, and **10×–45× on GPU** compared to Flash-KMeans.
- Successfully clusters massive-scale vision datasets such as **DINO10B** (10 billion 256-dimensional embeddings partitioned into 16 million clusters) in 2.5 passes.

### C. Cluster Utilization & Codebook Health
- In standard mini-batch clustering, up to 25–30% of centroids can "die" (receive 0 assignments) when trapped in sparse regions or shadowed by neighboring centroids.
- e-means reduces dead clusters to $< 0.3\%$ by periodically splitting donor clusters whose running counts exceed average occupancy.

---

## 3. Core Hyperparameters & Scaling Laws
The algorithm is designed to operate with zero manual hyperparameter search. Given:
- $n$: number of data samples
- $d$: feature dimension
- $k$: target number of clusters
- $T$: target epoch budget (default $T = 5$, achieving target loss in $T \le 3$)

The hyperparameters are resolved automatically:
- **Batch Size ($B$):** Governed by scaling law $B = \text{clip}\left(\text{round}(c_B \cdot k^{b_k} \cdot n^{b_n} \cdot T^{b_T}), B_{\min}, B_{\max}\right)$. For typical tabular/embedding regimes, $B \approx \max(256, \min(n, 8k))$.
- **Initial Step Size ($\alpha_0$):** Derived from empirical power laws $\alpha_0 = \min\left(1.0, c_\alpha \cdot k^{a_k} \cdot B^{a_B} \cdot n^{a_n} \cdot T^{a_T}\right)$, scaling proportionally to batch frequency and inverse epoch budget.
- **Decay Schedule:** Cosine decay from $\alpha_0$ to $\varepsilon \alpha_0$ ($\varepsilon = 0.1$) over $M = \lceil n / B \rceil \times T$ total iterations.
- **Revival Period ($P$):** Evaluated every $P$ iterations (e.g., every 5 mini-batches or per epoch) with dead threshold $\tau = 0.1 \cdot \bar{N}$.
