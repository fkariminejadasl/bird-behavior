import numpy as np
import torch
from sklearn.utils import check_random_state


##############################################################################
# 1) Utility for pairwise distance in PyTorch
##############################################################################
def pairwise_distance(data1, data2):
    """
    data1: (N, D) Tensor
    data2: (K, D) Tensor
    Returns: (N, K) of squared Euclidean distances
    """
    A = data1.unsqueeze(dim=1)  # (N, 1, D)
    B = data2.unsqueeze(dim=0)  # (1, K, D)
    return (A - B).pow(2).sum(dim=-1)  # (N, K)


class MiniBatchK_Means:
    def __init__(
        self, k=3, init="k-means++", n_init=1, random_state=None, device="cpu"
    ):
        """
        A semi-supervised mini-batch K-Means that:
          - handles labeled data by forcing class i => cluster i
          - trains in partial_fit style
          - can do multiple random initializations (n_init) and pick best inertia

        Parameters
        ----------
        k : int
            Number of clusters
        init : str
            "k-means++" or "random"
        n_init : int
            How many random restarts to try, picking best inertia
        random_state : int or None
            For reproducibility
        device : str
            "cpu" or "cuda"
        """
        self.k = k
        self.init = init
        self.n_init = n_init
        self.rng = check_random_state(random_state)
        self.device = device

        self.cluster_centers_ = None
        self.counts_ = None
        self.inertia_ = None
        self.labels_ = None  # Combined final labels (labeled + unlabeled) if desired
        self.best_seed_ = None

    def _kpp_init(self, X, leftover_k, pre_centers=None):
        """
        K-Means++ initialization on X (shape: (N, D)) - picks 'leftover_k' new centers.
        Optionally starts with pre_centers (shape: (m, D)) if given.
        """
        if pre_centers is None or pre_centers.shape[0] == 0:
            # pick the first center randomly
            idx = self.rng.randint(0, X.shape[0])
            centers = X[idx].unsqueeze(0)
            used = 1
        else:
            centers = pre_centers.clone()
            used = centers.shape[0]

        while used < leftover_k + (
            pre_centers.shape[0] if pre_centers is not None else 0
        ):
            dist = pairwise_distance(X, centers)  # (N, used)
            d2, _ = torch.min(dist, dim=1)  # (N,)
            prob = d2 / d2.sum()
            cum_prob = torch.cumsum(prob, dim=0)
            r = self.rng.rand()
            idx_new = (cum_prob >= r).nonzero()[0][0]
            c_new = X[idx_new].unsqueeze(0)
            centers = torch.cat((centers, c_new), dim=0)
            used += 1

        return centers

    def partial_fit_mix(self, u_feats_batch, l_feats, l_targets):
        """
        Process one mini-batch of unlabeled data (u_feats_batch) plus all labeled data in memory.
        Updates cluster centers incrementally.
        """
        u_feats_batch = u_feats_batch.to(self.device)
        l_feats = l_feats.to(self.device)
        l_targets = l_targets.to(self.device)

        # 1) If cluster centers not inited => do so now
        if self.cluster_centers_ is None:
            # gather one center per labeled class (force class i => center i)
            l_classes = torch.unique(l_targets)
            labeled_centers = []
            for c_val in l_classes:
                c_inds = (l_targets == c_val).nonzero().squeeze(1)
                if c_inds.numel() > 0:
                    center = l_feats[c_inds].mean(dim=0, keepdim=True)
                    labeled_centers.append(center)
            if labeled_centers:
                labeled_centers = torch.cat(labeled_centers, dim=0)  # (#l_classes, D)
            else:
                labeled_centers = None

            # if leftover clusters remain, do K++ on the unlabeled batch
            leftover_k = self.k - (
                labeled_centers.shape[0] if labeled_centers is not None else 0
            )
            if leftover_k > 0:
                new_centers = self._kpp_init(u_feats_batch, leftover_k, None)
                if labeled_centers is not None and labeled_centers.shape[0] > 0:
                    self.cluster_centers_ = torch.cat(
                        (labeled_centers, new_centers), dim=0
                    )
                else:
                    self.cluster_centers_ = new_centers
            else:
                # If labeled classes >= k, just take the first k
                self.cluster_centers_ = labeled_centers[: self.k]

            self.cluster_centers_ = self.cluster_centers_.to(self.device)
            self.counts_ = torch.zeros(self.k, dtype=torch.float, device=self.device)

        # 2) Assign unlabeled batch => nearest center, incremental update
        dist_unlab = pairwise_distance(
            u_feats_batch, self.cluster_centers_
        )  # (batch_size, k)
        u_labels = torch.argmin(dist_unlab, dim=1)
        for c_idx in range(self.k):
            c_mask = (u_labels == c_idx).nonzero().squeeze(1)
            if c_mask.numel() == 0:
                continue
            points = u_feats_batch[c_mask]
            old_count = float(self.counts_[c_idx].item())
            new_count = old_count + points.shape[0]
            mean_batch = points.mean(dim=0)
            old_center = self.cluster_centers_[c_idx]
            new_center = (
                old_center * old_count + mean_batch * points.shape[0]
            ) / new_count

            self.cluster_centers_[c_idx] = new_center
            self.counts_[c_idx] = new_count

        # 3) Force labeled data => class i => cluster i (if i < k)
        l_classes = torch.unique(l_targets)
        for i, c_val in enumerate(l_classes):
            if i >= self.k:
                break
            c_inds = (l_targets == c_val).nonzero().squeeze(1)
            if c_inds.numel() == 0:
                continue
            pts = l_feats[c_inds]
            old_count = float(self.counts_[i].item())
            new_count = old_count + pts.shape[0]
            mean_pts = pts.mean(dim=0)
            old_center = self.cluster_centers_[i]
            new_center = (old_center * old_count + mean_pts * pts.shape[0]) / new_count
            self.cluster_centers_[i] = new_center
            self.counts_[i] = new_count

    def _compute_streaming_inertia_pass(self, loader, model, layer_to_hook):
        """
        One streaming pass over 'loader' to compute sum of distances of unlabeled data
        to final cluster centers.  We do hooking to get embeddings on-the-fly.
        Returns: (float total_dist, torch.LongTensor of assigned cluster labels)
        """
        device = self.device
        total_dist = 0.0
        all_assigned_labels = []

        activation = []

        def hook_fn(module, inp, out):
            activation.append(out.detach())

        hook_handle = layer_to_hook.register_forward_hook(hook_fn)

        with torch.no_grad():
            for batch_data in loader:
                batch_data = batch_data.to(device)
                _ = model(batch_data)  # forward => fill activation
                if activation:
                    feats = (
                        activation.pop()
                    )  # shape (B, something, embed_dim) or (B, embed_dim)
                    # If shape is (B,1,D), flatten the middle
                    if feats.ndim == 3:
                        feats = feats[:, 0, :]
                    dist = pairwise_distance(feats, self.cluster_centers_)
                    mindist, minidx = torch.min(dist, dim=1)
                    total_dist += mindist.sum().item()
                    all_assigned_labels.append(minidx.cpu())

        hook_handle.remove()

        if len(all_assigned_labels) > 0:
            all_assigned_labels = torch.cat(all_assigned_labels, dim=0)
        else:
            all_assigned_labels = torch.tensor([], dtype=torch.long)
        return total_dist, all_assigned_labels

    def compute_streaming_inertia(
        self, loader, model, layer_to_hook, l_feats, l_targets
    ):
        """
        Does a streaming pass over the unlabeled loader + forces labeled data
        => computes total inertia = (sum of unlabeled distances) + (labeled distances).
        Returns: (float inertia, torch.LongTensor of final labels [labeled+unlabeled])
        """
        device = self.device
        # 1) unlabeled portion
        unl_dist_sum, unl_labels = self._compute_streaming_inertia_pass(
            loader, model, layer_to_hook
        )

        # 2) labeled portion: force class i => cluster i
        l_feats = l_feats.to(device)
        l_targets = l_targets.to(device)
        labeled_dist_sum = 0.0
        labeled_labels = torch.empty_like(l_targets, dtype=torch.long)

        l_classes = torch.unique(l_targets)
        for i, c_val in enumerate(l_classes):
            if i >= self.k:
                break
            c_inds = (l_targets == c_val).nonzero().squeeze(1)
            if c_inds.numel() == 0:
                continue
            pts = l_feats[c_inds]
            center_i = self.cluster_centers_[i]
            dist_sq = (pts - center_i).pow(2).sum(dim=1)
            labeled_dist_sum += dist_sq.sum().item()
            labeled_labels[c_inds] = i

        total_inertia = unl_dist_sum + labeled_dist_sum
        all_labels = torch.cat([labeled_labels.cpu(), unl_labels], dim=0)
        return total_inertia, all_labels

    def reset_model(self):
        """
        Clears cluster centers, counts, inertia, labels for a fresh run.
        """
        self.cluster_centers_ = None
        self.counts_ = None
        self.inertia_ = None
        self.labels_ = None

    def fit_mix_epochs_n_init_streaming(
        self,
        create_unlabeled_loader_func,
        model,
        layer_to_hook,
        l_feats,
        l_targets,
        n_epochs=5,
    ):
        """
        Repeatedly do:
          For each init in n_init:
            1) reset cluster centers
            2) for epoch in 1..n_epochs:
                 - create a fresh unlabeled loader via create_unlabeled_loader_func()
                 - partial_fit_mix(...) in mini-batches (hooking in your code's style)
            3) do a final streaming pass over unlabeled data => compute inertia
          Keep the best run (lowest inertia).

        create_unlabeled_loader_func: a callback that returns a *fresh* DataLoader for unlabeled
        model, layer_to_hook: used for hooking to get embeddings
        l_feats, l_targets: entire labeled data in memory
        n_epochs: how many passes over unlabeled data
        """
        best_inertia = None
        best_centers = None
        best_labels = None
        best_seed = None

        seeds = self.rng.randint(0, 2**31 - 1, size=self.n_init)

        for run_idx in range(self.n_init):
            run_seed = seeds[run_idx]
            local_rng = np.random.RandomState(run_seed)
            self.rng = local_rng  # so k++ uses run_seed

            # Reset
            self.reset_model()

            # 2) multiple epochs partial-fitting
            for ep in range(n_epochs):
                unl_loader = create_unlabeled_loader_func()
                activation = []

                def hook_fn(module, inp, out):
                    activation.append(out.detach())

                hook_handle = layer_to_hook.register_forward_hook(hook_fn)

                with torch.no_grad():
                    for batch_data in unl_loader:
                        batch_data = batch_data.to(self.device)
                        _ = model(batch_data)
                        if activation:
                            feats = activation.pop()  # shape e.g. (B,1,D) or (B,D)
                            if feats.ndim == 3:
                                feats = feats[:, 0, :]
                            # Now partial fit
                            self.partial_fit_mix(feats, l_feats, l_targets)

                hook_handle.remove()

            # 3) final streaming inertia pass
            unl_loader = create_unlabeled_loader_func()
            inertia_val, labels_val = self.compute_streaming_inertia(
                unl_loader, model, layer_to_hook, l_feats, l_targets
            )

            if best_inertia is None or inertia_val < best_inertia:
                best_inertia = inertia_val
                best_centers = self.cluster_centers_.clone()
                best_labels = labels_val.clone()
                best_seed = run_seed

        # store final best
        self.cluster_centers_ = best_centers
        self.inertia_ = best_inertia
        self.labels_ = best_labels
        self.best_seed_ = best_seed

    def predict(self, X):
        """
        Predict cluster labels for new data X: shape (N, D)
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet!")
        X = X.to(self.device)
        dist = pairwise_distance(X, self.cluster_centers_)
        labels = torch.argmin(dist, dim=1)
        return labels.cpu().numpy()
