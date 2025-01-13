Below is a short reference for the most common clustering metrics. Each metric measures a different aspect of clustering quality, and they can be grouped into two categories:

1. **Metrics that do *not* use true labels** (often called *internal* metrics):  
   - Inertia (a.k.a. within-cluster SSE(Sum of the Squares))  
   - Silhouette Score  
   - Calinski–Harabasz Score  
   - Davies–Bouldin Score  

2. **Metrics that *do* use true labels** (often called *external* metrics):  
   - Homogeneity, Completeness, V-measure  
   - Adjusted Rand Index (ARI)  
   - Adjusted Mutual Information (AMI)  

---

## 1. Metrics *Without* True Labels (Internal)

### Inertia
- **Definition**: The sum of squared distances of each sample to its assigned cluster center (a within-cluster “sum of squares”).  
- **Range**: \( [0, \infty) \). (Data-scale dependent; there is no universal upper bound).  
- **Higher or Lower Better?**: **Lower** is better (tighter clusters).  
- **Caveat**: Generally decreases as the number of clusters \( k \) increases, so you often look for an “elbow.”

---

### Silhouette Score
- **Definition**: For each sample, compares cohesion (distance to its own cluster center) vs. separation (distance to nearest other-cluster center).  
- **Range**: \(-1\) to \(+1\).  
- **Higher or Lower Better?**: **Higher** is better (values near +1 mean well-separated clusters; negative values mean many points may be in the wrong cluster; Scores around zero indicate overlapping clusters.).  

---

### Calinski–Harabasz (CH) Score also known as the Variance Ratio Criterion
- **Definition**: Ratio of the between-cluster dispersion to the within-cluster dispersion.  
- **Range**: \([0, \infty)\).  
- **Higher or Lower Better?**: **Higher** is better (indicating more distinct clusters).  

---

### Davies–Bouldin (DB) Score  
- **Definition**: Average ratio of each cluster’s within-cluster scatter to its between-cluster separation from the nearest other cluster.  
- **Range**: \([0, \infty)\). 
- **Higher or Lower Better?**: **Lower** is better (clusters should be more compact and farther apart).  

---

### Gap Statistic  
   - **Definition**: Compares the within-cluster dispersion to that expected under an appropriate reference null distribution.  
   - **Higher** is generally better.  

---

## 2. Metrics *With* True Labels (External)

When you have “ground truth” class labels and want to see how well your clusters match those labels, you can use these:

### Homogeneity
- **Definition**: A cluster is “homogeneous” if it contains only data points from a single class. Homogeneity is 1 if each cluster only contains data points that are members of a single class.  
- **Range**: 0 to 1.  
- **Higher or Lower Better?**: **Higher** is better.  

---

### Completeness
- **Definition**: A cluster satisfies “completeness” if all data points of a given class are assigned to the same cluster. Completeness is 1 if each class appears in only one cluster.  
- **Range**: 0 to 1.  
- **Higher or Lower Better?**: **Higher** is better.  

---

### V-measure
- **Definition**: The harmonic mean of homogeneity and completeness. Balances both metrics.  
- **Range**: 0 to 1.  
- **Higher or Lower Better?**: **Higher** is better (1 if perfectly homogeneous and complete).  

---

### Adjusted Rand Index (ARI)
- **Definition**: Measures the similarity of cluster assignments to ground truth labels based on pair counting (pairs of points in the same or different clusters). It’s a “chance-adjusted” version of the Rand Index.  
- **Range**: \([-0.5, 1]\). Close to 0.0 values mean random.  
- **Higher or Lower Better?**: **Higher** is better (1 = perfect agreement).  

---

### Adjusted Mutual Information (AMI) 
- **Definition**: Mutual Information between the cluster assignments and the ground truth labels, adjusted for chance.  
- **Range**: \([0, 1]\).  Close to 0.0 values mean random. 
- **Higher or Lower Better?**: **Higher** is better (1 = perfect correlation, 0 = random labels).  

---

###  Fowlkes–Mallows Score   
   - **Definition**: The geometric mean of precision and recall on pairwise cluster assignments, comparing to ground truth.  
   - **Range**: \([0, 1]\).  
   - **Higher** is better.

---

### **Purity**  
   - **Definition**: Fraction of the total number of correctly assigned labels (i.e., the majority label in a cluster).  
   - **Range**: \([0, 1]\).  
   - **Higher** is better.

---

## Internal Metrics (No True Labels Required)

| **Metric**             | **Range**          | **Higher/Lower** | **In** scikit-learn?                        | **What It Measures**                                                                          |
|------------------------|--------------------|------------------|----------------------------------------------|------------------------------------------------------------------------------------------------|
| **Inertia**            | \([0, \infty)\)   | Lower is better  | Stored in: `KMeans(...).inertia_`           | Sum of squared distances to cluster centers (within-cluster “sum of squares”).                 |
| **Silhouette Score**   | \([-1, 1]\)       | Higher is better | Yes: `sklearn.metrics.silhouette_score`     | Cohesion vs. separation of clusters.  
| **Calinski–Harabasz**  | \([0, \infty)\)   | Higher is better | Yes: `sklearn.metrics.calinski_harabasz_score` | Ratio of between-cluster to within-cluster dispersion.                                        |
| **Davies–Bouldin**     | \([0, \infty)\)   | Lower is better  | Yes: `sklearn.metrics.davies_bouldin_score` | Average ratio of within-cluster scatter to between-cluster separation.                         |
| **Gap Statistic**      | *Varies*          | Higher is better | **Not** in scikit-learn                     | Compares a clustering’s within-cluster dispersion to what would be expected under a “null” reference (e.g., random/uniform). Often used to estimate an “optimal” \(k\). |

> **Note**: 
> - **Inertia** is typically accessed via the fitted KMeans or MiniBatchKMeans object as `estimator.inertia_`.  
> - The **Gap Statistic** is a method introduced by Tibshirani, Walther, and Hastie (2001). It compares the within-cluster dispersion of your actual data to that of a reference dataset (often uniformly distributed in the same bounding box). The idea is to see if the clustering structure is significantly better than random.  It’s **not** in scikit-learn by default. People often implement it themselves or use third-party libraries (e.g., `gap-stat`, `kneed`, or custom code).



### External Metrics (Require True Labels)

| **Metric**              | **Range**          | **Higher/Lower** | **In** scikit-learn?                               | **What It Measures**                                                                     |
|-------------------------|--------------------|------------------|-----------------------------------------------------|-------------------------------------------------------------------------------------------|
| **Homogeneity**         | \([0, 1]\)        | Higher is better | Yes: `sklearn.metrics.homogeneity_score`           | Whether each cluster contains only data from a single true class.                         |
| **Completeness**        | \([0, 1]\)        | Higher is better | Yes: `sklearn.metrics.completeness_score`          | Whether all data points of a given class are assigned to the same cluster.               |
| **V-measure**           | \([0, 1]\)        | Higher is better | Yes: `sklearn.metrics.v_measure_score`             | Harmonic mean of Homogeneity and Completeness.                                           |
| **Adjusted Rand Index** | \([-0.5, 1]\)     | Higher is better | Yes: `sklearn.metrics.adjusted_rand_score`         | Similarity of clustering to true labels, adjusting for chance.                           |
| **Adjusted Mutual Info**| \([0, 1]\)        | Higher is better | Yes: `sklearn.metrics.adjusted_mutual_info_score`  | Mutual information (between cluster and labels), adjusted for chance.                    |
| **Fowlkes–Mallows**     | \([0, 1]\)        | Higher is better | Yes: `sklearn.metrics.fowlkes_mallows_score`       | Geometric mean of precision and recall on pairs of points.                               |

> **Note**: 
> - **Purity** is another external metric (fraction of correctly assigned points when each cluster is labeled by its majority class), but **not** included in scikit-learn. You typically have to code that manually.

---

#### Summary  
- **Internal metrics** (no true labels) like **inertia**, **silhouette**, **Calinski–Harabasz**, **Davies–Bouldin** help you gauge how well your data is partitioned based on distances/geometry alone.  
- **External metrics** (require true labels) like **homogeneity**, **completeness**, **V-measure**, **ARI**, **AMI**, **Fowlkes–Mallows** measure how closely your clustering results match known ground-truth labels.  

#### Reference
[sklearn Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)

#### Related topics

- Deep clustering
- Contrastive learning for clustering
- Open-Set Recognition / Open-World Learning
- Novel class discovery / novel category discovery / open-world classification / discovering new classes in unlabeled data.
- Fine-grained classification
- time-series self-supervised learning
