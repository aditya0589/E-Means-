# Algorithm Pseudocode: K-Means Variants

This document details the exact pseudocode for:
1. Traditional Lloyd's K-Means
2. K-Means++ Initialization
3. Elkan's Triangle Inequality K-Means
4. E-Means with EMA, Cosine Annealing, Dead-Cluster Revival, and Finalization

---

## 1. Algorithm 1: Traditional Lloyd's K-Means

```text
Algorithm: LloydKMeans
Input: 
    X: Data matrix of shape (n, d)
    k: Number of clusters
    max_iter: Maximum number of iterations (default: 300)
    tol: Relative tolerance for centroid convergence (default: 1e-4)
    seed: Random state seed

Output:
    centroids: Matrix of shape (k, d)
    labels: Array of length n containing cluster index for each sample
    inertia: Total Within-Cluster Sum of Squares (WCSS)
    n_iter: Number of iterations executed

1.  Initialize random number generator with seed.
2.  Sample k distinct indices from {0, ..., n-1} uniformly at random.
3.  centroids = X[indices]
4.  For iter = 1 to max_iter do:
5.      // Step 1: Expectation (Assign points)
6.      For each i from 0 to n-1:
7.          distances = [||x_i - centroids[j]||^2 for j in 0..k-1]
8.          labels[i] = argmin(distances)
9.      
10.     // Step 2: Maximization (Update centroids)
11.     new_centroids = zeros(k, d)
12.     counts = zeros(k)
13.     For each i from 0 to n-1:
14.         c = labels[i]
15.         new_centroids[c] += X[i]
16.         counts[c] += 1
17.     
18.     // Handle empty clusters
19.     For each j from 0 to k-1:
20.         If counts[j] > 0:
21.             new_centroids[j] = new_centroids[j] / counts[j]
22.         Else:
23.             // Reinitialize to the data point farthest from its current centroid
24.             farthest_idx = argmax(||X - centroids[labels]||^2)
25.             new_centroids[j] = X[farthest_idx]
26.     
27.     // Step 3: Check convergence
28.     shift = max(||new_centroids[j] - centroids[j]||^2 for j in 0..k-1)
29.     centroids = new_centroids
30.     If shift <= tol:
31.         break
32. End For
33. inertia = sum(||X[i] - centroids[labels[i]]||^2 for i in 0..n-1)
34. Return centroids, labels, inertia, iter
```

---

## 2. Algorithm 2: K-Means++ Initialization

```text
Algorithm: KMeansPlusPlusInit
Input:
    X: Data matrix of shape (n, d)
    k: Number of clusters
    seed: Random state seed

Output:
    centroids: Initialized centroid matrix of shape (k, d)

1.  Initialize RNG with seed.
2.  Pick first center c_0 uniformly at random from X.
3.  centroids = [c_0]
4.  min_dist_sq = [||x_i - c_0||^2 for i in 0..n-1]
5.  For j = 1 to k-1 do:
6.      total_dist_sq = sum(min_dist_sq)
7.      probs = min_dist_sq / total_dist_sq
8.      // Sample next center with probability proportional to D(x)^2
9.      chosen_idx = sample_index(probabilities=probs, rng=rng)
10.     c_j = X[chosen_idx]
11.     centroids.append(c_j)
12.     // Update minimum distances
13.     For each i from 0 to n-1:
14.         d_new = ||x_i - c_j||^2
15.         If d_new < min_dist_sq[i]:
16.             min_dist_sq[i] = d_new
17. End For
18. Return centroids
```

---

## 3. Algorithm 3: Elkan's Accelerated K-Means

```text
Algorithm: ElkanKMeans
Input:
    X: Data matrix of shape (n, d)
    k: Number of clusters
    max_iter: Maximum iterations
    tol: Convergence tolerance
    seed: Random seed

Output:
    centroids, labels, inertia, n_iter

1.  Initialize centroids using KMeansPlusPlusInit(X, k, seed).
2.  Initialize bounds:
        lower_bounds: matrix of shape (n, k), lower_bounds[i, j] = 0
        upper_bounds: array of length n, upper_bounds[i] = infinity
        r: boolean array of length n, r[i] = True (bounds are fresh)
3.  Compute pairwise inter-centroid distances:
        dist_cc[j, j'] = ||c_j - c_{j'}||
        s[j] = 0.5 * min_{j' != j} dist_cc[j, j']
4.  Assign initial clusters:
        For each point x_i, find closest center c(x_i), update upper_bound and lower_bounds.
5.  For iter = 1 to max_iter do:
6.      // Centroid pairwise distances
7.      dist_cc = compute_pairwise_distances(centroids)
8.      s[j] = 0.5 * min_{j' != j} dist_cc[j, j']
9.      
10.     // Point assignment with pruning
11.     For each i from 0 to n-1:
12.         c_curr = labels[i]
13.         If upper_bounds[i] <= s[c_curr]:
14.             continue  // Lemma 1: No other centroid can be closer!
15.         
16.         For each j from 0 to k-1 (j != c_curr):
17.             If upper_bounds[i] <= lower_bounds[i, j]:
18.                 continue  // Pruned by lower bound
19.             If upper_bounds[i] <= 0.5 * dist_cc[c_curr, j]:
20.                 continue  // Pruned by inter-center distance
21.             
22.             // Recompute exact distance to current center if upper bound is stale
23.             If r[i]:
24.                 d_curr = ||X[i] - centroids[c_curr]||
25.                 upper_bounds[i] = d_curr
26.                 lower_bounds[i, c_curr] = d_curr
27.                 r[i] = False
28.             Else:
29.                 d_curr = upper_bounds[i]
30.             
31.             If d_curr > lower_bounds[i, j] or d_curr > 0.5 * dist_cc[c_curr, j]:
32.                 d_cand = ||X[i] - centroids[j]||
33.                 lower_bounds[i, j] = d_cand
34.                 If d_cand < d_curr:
35.                     c_curr = j
36.                     upper_bounds[i] = d_cand
37.                     labels[i] = j
38.     
39.     // Update centroids and shift vectors
40.     new_centroids = recompute_means(X, labels, k)
41.     shift = [||new_centroids[j] - centroids[j]|| for j in 0..k-1]
42.     
43.     // Update bounds based on centroid movement
44.     For each i from 0 to n-1:
45.         upper_bounds[i] += shift[labels[i]]
46.         For j from 0 to k-1:
47.             lower_bounds[i, j] = max(0, lower_bounds[i, j] - shift[j])
48.         r[i] = True
49.     
50.     centroids = new_centroids
51.     If max(shift) <= tol:
52.         break
53. End For
54. Return centroids, labels, inertia, iter
```

---

## 4. Algorithm 4: E-Means (Meta FAIR)

```text
Algorithm: EMeans
Input:
    X: Data matrix of shape (n, d)
    k: Number of clusters
    n_epochs: Epoch budget (default: 5)
    batch_size: Mini-batch size B (optional, auto-computed via scaling law)
    alpha_0: Initial EMA step size (optional, auto-computed via scaling law)
    revival_interval: Interval P in batches for dead-cluster revival (default: 5)
    dead_threshold_ratio: Ratio tau of mean count to declare cluster dead (default: 0.1)
    seed: Random state seed

Output:
    centroids: Matrix of shape (k, d)
    labels: Array of length n
    inertia: Final WCSS
    history: Dictionary of training trajectories (inertia, time, centroids, dead_clusters)

1.  Initialize RNG with seed.
2.  Compute scaling-law parameters if not manually overridden:
        If batch_size is None:
            B = clip(round(c_B * (k^b_k) * (n^b_n) * (n_epochs^b_T)), min_val=128, max_val=n)
            Default heuristic: B = min(n, max(128, 8 * k))
        If alpha_0 is None:
            alpha_0 = min(1.0, 2.0 * B / n)
        
3.  Total mini-batch steps: M = n_epochs * ceil(n / B)
4.  Initialize centroids C:
        C = KMeansPlusPlusInit(X, k, seed)  // or random sample of k points
5.  Initialize EMA statistics:
        N = ones(k) * (B / k)               // Initial uniform cluster counts
        S = C * N[:, None]                  // Initial weighted sum vectors
6.  step = 0
7.  For epoch = 1 to n_epochs do:
8.      shuffled_indices = random_permutation(n, rng=rng)
9.      For batch_start = 0 to n-1 step B do:
10.         batch_idx = shuffled_indices[batch_start : min(batch_start + B, n)]
11.         X_batch = X[batch_idx]
12.         
13.         // Step 1: Assign batch points to current centroids
14.         batch_labels = argmin_pairwise_distance(X_batch, C)
15.         
16.         // Step 2: Compute instantaneous batch statistics
17.         n_batch = count_occurrences(batch_labels, k)
18.         s_batch = sum_by_cluster(X_batch, batch_labels, k)
19.         
20.         // Step 3: Compute cosine decay update rate alpha(step)
21.         alpha_t = alpha_0 * (0.1 + 0.9 * 0.5 * (1 + cos(pi * step / M)))
22.         
23.         // Step 4: EMA update of sufficient statistics
24.         N = (1 - alpha_t) * N + alpha_t * n_batch
25.         S = (1 - alpha_t) * S + alpha_t * s_batch
26.         
27.         // Step 5: Update centroids
28.         For j = 0 to k-1:
29.             If N[j] > 1e-8:
30.                 C[j] = S[j] / N[j]
31.         
32.         // Step 6: Dead-cluster revival (every P steps)
33.         If step > 0 and step % revival_interval == 0:
34.             mean_count = mean(N)
35.             dead_mask = (N < dead_threshold_ratio * mean_count)
36.             dead_indices = where(dead_mask)
37.             
38.             For each j_dead in dead_indices:
39.                 // Select donor proportional to cluster size N
40.                 donor = sample_donor(weights=N, exclude=dead_indices, rng=rng)
41.                 // Directional perturbation
42.                 noise = sample_gaussian(mean=0, std=0.01 * std(X), shape=(d,), rng=rng)
43.                 C[j_dead] = C[donor] + noise
44.                 C[donor]  = C[donor] - noise
45.                 // Split statistics
46.                 N[donor]  = N[donor] / 2.0
47.                 N[j_dead] = N[donor]
48.                 S[donor]  = N[donor] * C[donor]
49.                 S[j_dead] = N[j_dead] * C[j_dead]
50.         
51.         step += 1
52.     End For
53. End For
54. 
55. // Step 7: Finalization pass (Exact Full-Batch Update)
56. full_labels = argmin_pairwise_distance(X, C)
57. For j = 0 to k-1:
58.     assigned_pts = X[full_labels == j]
59.     If length(assigned_pts) > 0:
60.         C[j] = mean(assigned_pts, axis=0)
61. 
62. final_labels = argmin_pairwise_distance(X, C)
63. final_inertia = sum(||X[i] - C[final_labels[i]]||^2)
64. Return C, final_labels, final_inertia, history
```
