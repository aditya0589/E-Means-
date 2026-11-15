# Mathematical Methodology: Traditional K-Means, Optimized K-Means, and E-Means

This document provides complete mathematical derivations, objective formulations, convergence proofs, distance metrics, and algorithmic optimizations for:
1. Traditional Lloyd's K-Means
2. K-Means++ Probabilistic Initialization
3. Elkan's Triangle Inequality Accelerated K-Means
4. E-Means with Exponential Moving Average & Scaling Laws

---

## 1. Problem Formulation: The K-Means Objective

Let $X = \{x_1, x_2, \dots, x_n\} \subset \mathbb{R}^d$ be a set of $n$ observations in $d$-dimensional Euclidean space. The clustering objective is to partition the $n$ points into $k$ disjoint subsets $C = \{C_1, C_2, \dots, C_k\}$ such that:
$$\bigcup_{j=1}^k C_j = X \quad \text{and} \quad C_i \cap C_j = \emptyset \quad \forall i \ne j$$

Each cluster $C_j$ is parameterized by its centroid (mean vector) $c_j \in \mathbb{R}^d$. The standard $k$-means optimization objective is to minimize the Within-Cluster Sum of Squares (WCSS), also known as **Inertia** or **Distortion**:

$$\mathcal{J}(C, \{c_j\}_{j=1}^k) = \sum_{j=1}^k \sum_{x_i \in C_j} \|x_i - c_j\|_2^2$$

Equivalently, expressing cluster assignments via indicator variables $z_{ij} \in \{0, 1\}$ with $\sum_{j=1}^k z_{ij} = 1$:
$$\mathcal{J}(Z, C) = \sum_{i=1}^n \sum_{j=1}^k z_{ij} \|x_i - c_j\|_2^2$$

Minimizing $\mathcal{J}$ is an NP-hard combinatorial problem in general Euclidean space ($k \ge 2, d \ge 2$).

---

## 2. Lloyd's Algorithm (Traditional K-Means)

Lloyd's algorithm solves this problem using coordinate descent (Expectation-Maximization):

### Step 1: Expectation Step (Optimal Partitioning)
Given fixed centroids $C = \{c_1, \dots, c_k\}$, the optimal cluster indicator assignment that minimizes $\mathcal{J}$ is the nearest neighbor rule:
$$z_{ij}^{(t)} = \begin{cases} 1 & \text{if } j = \arg\min_{\ell \in \{1,\dots,k\}} \|x_i - c_\ell^{(t)}\|_2^2 \\ 0 & \text{otherwise} \end{cases}$$
Ties are broken arbitrarily or deterministically by lowest cluster index.

### Step 2: Maximization Step (Optimal Centroid Update)
Given fixed assignments $Z^{(t)}$, the objective $\mathcal{J}$ is strictly convex in each $c_j$. Setting the gradient with respect to $c_j$ to zero:
$$\nabla_{c_j} \mathcal{J} = -2 \sum_{i=1}^n z_{ij} (x_i - c_j) = 0 \implies c_j^{(t+1)} = \frac{\sum_{i=1}^n z_{ij}^{(t)} x_i}{\sum_{i=1}^n z_{ij}^{(t)}} = \frac{1}{|C_j^{(t)}|} \sum_{x \in C_j^{(t)}} x$$

### Convergence Guarantee
Since:
1. The number of possible partitions of $n$ points into $k$ clusters is finite ($S(n,k) \le k^n / k!$).
2. Both Step 1 and Step 2 strictly decrease or keep unchanged the objective value $\mathcal{J}$.
3. $\mathcal{J}$ is bounded below by 0.

Lloyd's algorithm is guaranteed to converge to a local minimum in a finite number of iterations.

### Computational Complexity
- **Time Complexity:** $O(I \cdot n \cdot k \cdot d)$ where $I$ is the number of iterations until convergence.
- **Space Complexity:** $O(n \cdot d + k \cdot d)$ to store data and centroids.

---

## 3. K-Means++ Initialization (Arthur & Vassilvitskii, 2007)

Standard Lloyd's algorithm with uniform random initialization can converge to arbitrarily poor local optima, with approximation ratio $\Omega(n)$. K-Means++ introduces a randomized seeding technique with an $O(\log k)$ competitive ratio.

### Mathematical Formulation
1. **First Centroid:** Choose the first center $c_1$ uniformly at random from $X$:
   $$\mathbb{P}(c_1 = x_i) = \frac{1}{n}$$
2. **Subsequent Centroids ($j = 2, \dots, k$):**
   For each data point $x \in X$, compute its shortest squared distance to any already chosen center:
   $$D(x)^2 = \min_{m=1}^{j-1} \|x - c_m\|_2^2$$
   Select the next center $c_j = x_i$ from $X$ with probability proportional to $D(x_i)^2$:
   $$\mathbb{P}(c_j = x_i) = \frac{D(x_i)^2}{\sum_{m=1}^n D(x_m)^2}$$
3. **Repeat** until $k$ centers are chosen, then proceed with Lloyd's iterations.

### Theoretical Bound
$$\mathbb{E}[\mathcal{J}_{\text{kmeans++}}] \le 8(\ln k + 2) \mathcal{J}_{\text{optimal}}$$
This initialization dramatically stabilizes convergence, reduces variance across random seeds, and lowers the required number of Lloyd iterations.

---

## 4. Elkan's Algorithm (Accelerated K-Means via Triangle Inequality)

Elkan (2003) observed that the vast majority of distance computations in Lloyd's algorithm are redundant because points rarely change clusters in later iterations.

### Mathematical Principles
By the triangle inequality in Euclidean space, for any point $x$ and any two centers $c, c'$:
$$\|x - c'\| \ge \|c - c'\| - \|x - c\|$$

#### Lemma 1 (Lower Bound Pruning):
Let $c(x)$ be the center currently assigned to $x$. If:
$$\|x - c(x)\| \le \frac{1}{2} \|c(x) - c'\|$$
then $\|x - c'\| \ge \|x - c(x)\|$, and center $c'$ cannot be closer to $x$ than $c(x)$. Thus, the explicit computation of $\|x - c'\|$ can be skipped completely!

#### Lemma 2 (Upper and Lower Bounds under Centroid Shift):
When centers move from $c$ to $c'$, let the shift distance be:
$$\Delta(c) = \|c - c'\|$$
- The upper bound $u(x)$ on $\|x - c(x)\|$ updates as:
  $$u(x) \leftarrow u(x) + \Delta(c(x))$$
- The lower bound $\ell(x, j)$ on $\|x - c_j\|$ updates as:
  $$\ell(x, j) \leftarrow \max(0, \ell(x, j) - \Delta(c_j))$$

### Maintained Bounds
1. $u(x) \ge \|x - c(x)\|$ for every $x \in X$.
2. $\ell(x, j) \le \|x - c_j\|$ for every $x \in X$ and each cluster $j \in \{1,\dots,k\}$.
3. $s(j) = \frac{1}{2} \min_{j' \ne j} \|c_j - c_{j'}\|$ precomputed at each iteration in $O(k^2 d)$.

If $u(x) \le s(c(x))$, point $x$ is guaranteed to remain in cluster $c(x)$, skipping all $k-1$ distance calculations for point $x$ in that iteration.

---

## 5. E-Means: Mini-Batch with Exponential Moving Average & Scaling Laws

E-Means (Bessa et al., Meta FAIR, 2024/2026) replaces full-batch EM updates with an Exponential Moving Average (EMA) formulation over mini-batches.

### Step 1: Mini-Batch Sampling & Assignment
At iteration $t$, draw a mini-batch $B_t \subset X$ of size $B$:
For each sample $x_i \in B_t$, compute the nearest centroid:
$$y_i = \arg\min_{j \in \{1,\dots,k\}} \|x_i - c_j^{(t)}\|_2^2$$

### Step 2: Batch Sufficient Statistics
For each cluster $j \in \{1,\dots,k\}$, compute the instantaneous batch mass $n_j^{(t)}$ and vector sum $s_j^{(t)}$:
$$n_j^{(t)} = \sum_{x_i \in B_t} \mathbb{I}(y_i = j)$$
$$s_j^{(t)} = \sum_{x_i \in B_t, y_i = j} x_i$$

### Step 3: EMA Sufficient Statistics Update
Instead of indefinite accumulation (which causes old stale statistics to persist indefinitely), E-Means tracks exponentially smoothed running statistics:
$$N_j^{(t+1)} = (1 - \alpha(t)) N_j^{(t)} + \alpha(t) n_j^{(t)}$$
$$S_j^{(t+1)} = (1 - \alpha(t)) S_j^{(t)} + \alpha(t) s_j^{(t)}$$

Centroids are obtained algebraically:
$$c_j^{(t+1)} = \begin{cases} \frac{S_j^{(t+1)}}{N_j^{(t+1)}} & \text{if } N_j^{(t+1)} > 0 \\ c_j^{(t)} & \text{otherwise} \end{cases}$$

### Step 4: Cosine Decay Schedule
The update rate $\alpha(t)$ starts at an initial peak $\alpha_0$ and decays gracefully following a half-cosine profile:
$$\alpha(t) = \alpha_0 \left[ \varepsilon + (1 - \varepsilon) \frac{1}{2} \left( 1 + \cos\left( \frac{\pi t}{M} \right) \right) \right]$$
where:
- $\varepsilon = 0.1$ is the floor ratio (ensuring learning rate ends at $10\%$ of peak).
- $M = T \cdot \lceil n / B \rceil$ is the total number of mini-batch update steps across $T$ epochs.

### Step 5: Dead-Cluster Detection & Revival
Every $P$ steps (e.g. $P = 5$ batches), check for under-utilized centroids:
$$\bar{N} = \frac{1}{k} \sum_{i=1}^k N_i^{(t)}$$
A cluster $j$ is identified as dead if:
$$N_j^{(t)} < \tau \cdot \bar{N} \quad (\text{default } \tau = 0.1)$$

When cluster $j$ is dead:
1. Select a donor cluster $d \in \{1,\dots,k\}$ proportional to its running mass $N_d^{(t)}$:
   $$\mathbb{P}(\text{donor} = d) = \frac{N_d^{(t)}}{\sum_{m=1}^k N_m^{(t)}}$$
2. Perturb the donor centroid to create two sibling centroids:
   $$c_j \leftarrow c_d + \delta, \quad c_d \leftarrow c_d - \delta$$
   where $\delta \sim \mathcal{N}(0, \sigma^2 I)$ with $\sigma = 0.01 \cdot \text{std}(X)$.
3. Split the sufficient statistics equally:
   $$N_d \leftarrow \frac{1}{2} N_d, \quad N_j \leftarrow N_d$$
   $$S_d \leftarrow N_d \cdot c_d, \quad S_j \leftarrow N_j \cdot c_j$$

### Step 6: Empirical Scaling Laws
E-Means avoids costly grid searches by setting $B$ and $\alpha_0$ automatically via power laws:
$$B = \text{clip}\left(\text{round}\left(c_B \cdot k^{b_k} \cdot n^{b_n} \cdot T^{b_T}\right), B_{\min}, n\right)$$
$$\alpha_0 = \min\left(1.0, c_\alpha \cdot k^{a_k} \cdot B^{a_B} \cdot n^{a_n} \cdot T^{a_T}\right)$$
For datasets with $n \le 10^5$, robust scaling sets $B = \min(n, \max(128, 8k))$ and $\alpha_0 = \min\left(1.0, \frac{2 B}{n}\right)$.

### Step 7: Finalization Pass
After $M$ mini-batch updates, a single full-dataset Lloyd assignment is performed:
$$c_j^* = \frac{1}{|C_j|} \sum_{x \in C_j} x \quad \text{where } C_j = \{x \in X : \arg\min_\ell \|x - c_\ell\| = j\}$$
This completely eliminates any remaining stochastic mini-batch variance and ensures that final centroids match the global optimum of the current cluster assignments.
