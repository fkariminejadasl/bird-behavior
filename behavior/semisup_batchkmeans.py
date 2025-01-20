"""
GCD implementation (below link) is kmean based but I need batch-based kmean. So the code from GCD is changed for batch kmean.
https://github.com/sgvaze/generalized-category-discovery/blob/main/methods/clustering/faster_mix_k_means_pytorch.py
"""

import random

import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score


##############################################################################
# 1) Define Utility: pairwise_distance
##############################################################################
def pairwise_distance(data1, data2):
    """
    data1: shape (N, D)
    data2: shape (K, D)
    returns: (N, K) matrix of squared Euclidean distances
    """
    # A: (N, 1, D)
    A = data1.unsqueeze(dim=1)
    # B: (1, K, D)
    B = data2.unsqueeze(dim=0)
    dist = (A - B) ** 2
    return dist.sum(dim=-1)  # shape (N, K)


##############################################################################
# 2) Define the MiniBatchK_Means class (extended with n_init logic)
##############################################################################
class MiniBatchK_Means:
    def __init__(
        self,
        k=3,
        init="k-means++",
        max_iter=100,
        n_init=1,
        random_state=None,
        device="cpu",
    ):
        """
        A simple mini-batch K-Means (semi-supervised) that can handle labeled + unlabeled data
        via partial_fit style. We also allow multiple inits (n_init) to pick the best inertia.

        Parameters
        ----------
        k : int
            Number of clusters
        init : str
            Initialization method, e.g. "k-means++"
        max_iter : int
            Not strictly used like in standard K-Means but kept for consistency
        n_init : int
            How many times to restart (with different seeds) and choose best by inertia
        random_state : int or np.random.RandomState
            For reproducibility
        device : str
            "cpu" or "cuda"
        """
        self.k = k
        self.init = init
        self.max_iter = max_iter
        self.n_init = n_init
        self.rng = np.random.RandomState(random_state)
        self.device = device

        self.cluster_centers_ = None  # Will hold final chosen centers
        self.counts_ = None
        self.inertia_ = None
        self.labels_ = None  # If you want to store final labels
        self.n_iter_ = 0

    def _kpp_init(self, X, k, pre_centers=None):
        """
        A simplified K++ init that can optionally start with 'pre_centers'.
        X: (N, D) torch.Tensor
        pre_centers: (m, D) torch.Tensor or None
        """
        if pre_centers is not None and pre_centers.shape[0] > 0:
            centers = pre_centers.clone()
        else:
            # pick first center randomly
            idx = self.rng.randint(0, X.shape[0])
            centers = X[idx].unsqueeze(0)

        # Continue picking centers until we have k
        while centers.shape[0] < k:
            dist = pairwise_distance(X, centers)  # (N, current_num_centers)
            d2, _ = torch.min(dist, dim=1)  # (N,)
            prob = d2 / d2.sum()
            cum_prob = torch.cumsum(prob, dim=0)
            r = self.rng.rand()
            idx_new = (cum_prob >= r).nonzero()[0][0]
            c_new = X[idx_new].unsqueeze(0)
            centers = torch.cat((centers, c_new), dim=0)

        return centers

    def partial_fit_mix(self, u_feats_batch, l_feats, l_targets):
        """
        Process one mini-batch of unlabeled data (u_feats_batch)
        together with labeled data (l_feats, l_targets).
        Updates centers incrementally.
        """
        # Ensure device
        u_feats_batch = u_feats_batch.to(self.device)
        l_feats = l_feats.to(self.device)
        l_targets = l_targets.to(self.device)

        # If cluster_centers_ not initialized, do so now
        if self.cluster_centers_ is None:
            # 1) We assume each unique labeled class is assigned
            l_classes = torch.unique(l_targets)
            class_centers = []
            for c in l_classes:
                c_inds = (l_targets == c).nonzero().squeeze(1)
                c_points = l_feats[c_inds]
                class_centers.append(c_points.mean(dim=0, keepdim=True))
            class_centers = (
                torch.cat(class_centers, dim=0) if len(class_centers) else None
            )

            # 2) If leftover clusters remain, use K++ on the unlabeled mini-batch
            leftover_k = self.k - (
                class_centers.shape[0] if class_centers is not None else 0
            )
            if leftover_k > 0:
                # do k++ from unlabeled batch
                if class_centers is not None:
                    new_centers = self._kpp_init(u_feats_batch, leftover_k, None)
                    self.cluster_centers_ = torch.cat(
                        (class_centers, new_centers), dim=0
                    )
                else:
                    self.cluster_centers_ = self._kpp_init(u_feats_batch, self.k, None)
            else:
                # Edge case: we have more labeled classes than k, or exactly equal
                self.cluster_centers_ = class_centers[: self.k]

            self.cluster_centers_ = self.cluster_centers_.to(self.device)
            self.counts_ = torch.zeros(self.k, dtype=torch.float, device=self.device)

        # ======================
        # 1) Assign unlabeled to nearest center + incremental update
        # ======================
        dist_unlab = pairwise_distance(
            u_feats_batch, self.cluster_centers_
        )  # (batch_size, k)
        u_labels_batch = torch.argmin(dist_unlab, dim=1)  # (batch_size,)

        for c in range(self.k):
            mask = (u_labels_batch == c).nonzero().squeeze(1)
            if mask.numel() == 0:
                continue
            points_c = u_feats_batch[mask]
            mean_c = points_c.mean(dim=0)

            batch_count = float(points_c.shape[0])
            old_count = float(self.counts_[c].item())
            new_count = old_count + batch_count

            old_center = self.cluster_centers_[c]
            new_center = (old_center * old_count + mean_c * batch_count) / new_count

            self.cluster_centers_[c] = new_center
            self.counts_[c] = new_count

        # ======================
        # 2) Update centers based on labeled data
        #    We force labeled class i -> cluster i (if i < k).
        # ======================
        l_classes = torch.unique(l_targets)
        for i, c_val in enumerate(l_classes):
            if i >= self.k:
                break  # If there are more classes than k, handle externally
            c_inds = (l_targets == c_val).nonzero().squeeze(1)
            if c_inds.numel() == 0:
                continue
            points_c = l_feats[c_inds]
            mean_c = points_c.mean(dim=0)

            old_count = float(self.counts_[i].item())
            batch_count = float(points_c.shape[0])
            new_count = old_count + batch_count

            old_center = self.cluster_centers_[i]
            new_center = (old_center * old_count + mean_c * batch_count) / new_count

            self.cluster_centers_[i] = new_center
            self.counts_[i] = new_count

    def fit_mix_minibatch(
        self, u_feats, l_feats, l_targets, batch_size=64, n_epochs=5, shuffle=True
    ):
        """
        Perform mini-batch training over the unlabeled data multiple epochs.
        """
        u_feats = u_feats.to(self.device)
        l_feats = l_feats.to(self.device)
        l_targets = l_targets.to(self.device)

        dataset_size = u_feats.shape[0]
        idxs = np.arange(dataset_size)

        for _ in range(n_epochs):
            if shuffle:
                self.rng.shuffle(idxs)

            start = 0
            while start < dataset_size:
                end = min(start + batch_size, dataset_size)
                batch_idx = idxs[start:end]
                u_feats_batch = u_feats[batch_idx]
                self.partial_fit_mix(u_feats_batch, l_feats, l_targets)
                start = end

    def compute_total_inertia(self, u_feats, l_feats, l_targets):
        """
        Compute "semi-supervised" inertia:
          - sum of distances of unlabeled data to nearest center
          - plus sum of distances of labeled data to their 'forced' center (class i => cluster i)
        Returns: (inertia_value, all_labels)
        """
        # 1) Unlabeled part
        dist_unlab = pairwise_distance(u_feats, self.cluster_centers_)
        u_mindist, u_labels = torch.min(dist_unlab, dim=1)
        u_inertia = u_mindist.sum()  # sum of minimal distances

        # 2) Labeled part
        # Force each labeled class i -> cluster i, if possible
        l_classes = torch.unique(l_targets)
        labeled_dist_sum = 0.0
        labeled_labels = torch.empty_like(l_targets, dtype=torch.long)
        for i, c_val in enumerate(l_classes):
            if i >= self.k:
                # If # of labeled classes > k, this logic won't strictly hold
                # but let's just break
                break
            c_inds = (l_targets == c_val).nonzero().squeeze(1)
            if c_inds.numel() == 0:
                continue
            points_c = l_feats[c_inds]
            center_i = self.cluster_centers_[i]
            dists_sq = torch.sum((points_c - center_i) ** 2, dim=1)
            labeled_dist_sum += dists_sq.sum().item()
            labeled_labels[c_inds] = i

        total_inertia = float(u_inertia.item() + labeled_dist_sum)

        # Combine labeled and unlabeled labels if you want an overall label array
        # For convenience, let's define them in a single vector:
        # cat_feats = [l_feats, u_feats] so let's do the same for labels
        all_labels = torch.cat([labeled_labels, u_labels])
        return total_inertia, all_labels

    def predict(self, X):
        """
        Predict cluster labels for new data X.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet!")
        X = X.to(self.device)
        dist = pairwise_distance(X, self.cluster_centers_)
        labels = torch.argmin(dist, dim=1)
        return labels.cpu().numpy()

    ############################################################################
    # NEW: The method that does multiple inits => best inertia
    ############################################################################
    def fit_mix_minibatch_n_init(
        self, u_feats, l_feats, l_targets, batch_size=64, n_epochs=5, shuffle=True
    ):
        """
        Similar to the original GCD code's approach: run mini-batch training
        'n_init' times, each with a different random seed, and select the best
        model based on minimal inertia.
        """
        best_inertia = None
        best_centers = None
        best_labels = None
        best_iters = 0

        seeds = self.rng.randint(0, 2**31 - 1, size=self.n_init)

        for run_idx in range(self.n_init):
            # Create fresh model state for each initialization
            # (We reset self.cluster_centers_, counts_, etc.)
            self.cluster_centers_ = None
            self.counts_ = None

            # Optionally, set up a new RNG per run:
            local_rng = np.random.RandomState(seeds[run_idx])
            self.rng = local_rng  # reassign so that _kpp_init uses this local RNG

            # Train with mini-batches
            self.fit_mix_minibatch(
                u_feats,
                l_feats,
                l_targets,
                batch_size=batch_size,
                n_epochs=n_epochs,
                shuffle=shuffle,
            )

            # Compute inertia
            inertia_val, labels_val = self.compute_total_inertia(
                u_feats, l_feats, l_targets
            )

            # Check if best
            if best_inertia is None or inertia_val < best_inertia:
                best_inertia = inertia_val
                best_centers = self.cluster_centers_.clone()
                best_labels = labels_val.clone()
                best_iters = n_epochs

        # Finally, store the best results
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia
        self.labels_ = best_labels
        self.n_iter_ = best_iters


##############################################################################
# 3) TEST the code on synthetic data
##############################################################################
if __name__ == "__main__":
    # Fix random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Generate synthetic data: 4 clusters in 2D
    N = 800
    K = 4
    X_data, y_data = make_blobs(
        n_samples=N, n_features=2, centers=K, cluster_std=1.0, random_state=seed
    )

    # Suppose we treat clusters 0 and 1 as "labeled" => classes 0 and 1
    # Clusters 2 and 3 as "unlabeled"
    labeled_mask = (y_data == 0) | (y_data == 1)
    unlabeled_mask = ~labeled_mask

    l_feats_np = X_data[labeled_mask]  # shape (some_l, 2)
    l_targets_np = y_data[labeled_mask]
    u_feats_np = X_data[unlabeled_mask]  # shape (some_u, 2)
    u_targets_np = y_data[unlabeled_mask]  # just for evaluating

    print("Number of labeled samples:", len(l_feats_np))
    print("Number of unlabeled samples:", len(u_feats_np))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l_feats_t = torch.from_numpy(l_feats_np).float().to(device)
    l_targets_t = torch.from_numpy(l_targets_np).long().to(device)
    u_feats_t = torch.from_numpy(u_feats_np).float().to(device)

    # Create our mini-batch KMeans object with multiple inits:
    mbk = MiniBatchK_Means(
        k=K,
        n_init=5,  # We'll do 5 different random initializations
        random_state=seed,
        device=device,
    )

    # Run "fit_mix_minibatch_n_init"
    mbk.fit_mix_minibatch_n_init(
        u_feats_t, l_feats_t, l_targets_t, batch_size=64, n_epochs=5, shuffle=True
    )

    # Evaluate on the entire dataset:
    X_all_torch = torch.from_numpy(X_data).float().to(device)
    pred_labels = mbk.predict(X_all_torch)

    nmi_val = normalized_mutual_info_score(y_data, pred_labels)
    ari_val = adjusted_rand_score(y_data, pred_labels)
    print(f"\nBest inertia after n_init: {mbk.inertia_:.4f}")
    print(f"NMI = {nmi_val:.4f}, ARI = {ari_val:.4f}")

    # Just to see final centers:
    print("\nFinal cluster centers:")
    print(mbk.cluster_centers_.cpu().numpy())

    import matplotlib.pylab as plt

    # 5) Plot
    center_labels = mbk.cluster_centers_.cpu().numpy()
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap="tab20", s=5)
    plt.scatter(center_labels[:, 0], center_labels[:, 1], c="black", marker="X", s=50)
    plt.colorbar(scatter, label="Cluster Label")
    plt.show(block=True)
    print("done")
