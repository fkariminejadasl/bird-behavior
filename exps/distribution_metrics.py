# Comparing Two Distributions — Corrected & Annotated
# --------------------------------------------------
# This file consolidates your original script with the requested fixes:
#   1) Remove incorrect t-SNE "sim_*" KL part; rely on proper P/Q.
#   2) Use bandwidths (h1, h2) with (X1, X2) and (hp, hq) with (Xp, Xq).
#   3) Avoid implicit matrix inverse in Gaussian quadratic forms (use solve once).
#   4) Add clear, annotated printouts (units, interpretations).

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import slogdet, solve
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import StandardScaler

# -------------------------
# Helper: Gaussian measures
# -------------------------


def mean_and_cov(X, eps=1e-6):
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1])
    return mu, Sigma


def kl_gaussians(mu0, S0, mu1, S1):
    """KL divergence D_KL( N0 || N1 ) for multivariate Gaussians.
    Uses solves instead of explicit matrix inverse.
    Returns a Python float (nats).
    """
    k = mu0.shape[0]

    # log det terms
    sign0, logdet0 = slogdet(S0)
    sign1, logdet1 = slogdet(S1)
    if sign0 <= 0 or sign1 <= 0:
        raise ValueError("Covariance not PD after regularization.")

    # trace( S1^{-1} S0 ) via solve
    X = solve(S1, S0)
    trace_term = float(np.trace(X))

    # quadratic term (mu1 - mu0)^T S1^{-1} (mu1 - mu0)
    diff = mu1 - mu0
    v = solve(S1, diff)
    q = float(diff @ v)

    return 0.5 * ((logdet1 - logdet0) - k + trace_term + q)


def bhattacharyya_gaussians(mu0, S0, mu1, S1):
    """Bhattacharyya distance between two multivariate Gaussians."""
    Sm = 0.5 * (S0 + S1)

    sign_m, logdet_m = slogdet(Sm)
    sign_0, logdet_0 = slogdet(S0)
    sign_1, logdet_1 = slogdet(S1)
    if min(sign_m, sign_0, sign_1) <= 0:
        raise ValueError("Covariance not PD after regularization.")

    diff = mu1 - mu0
    v = solve(Sm, diff)
    q = float(diff @ v)

    term1 = 0.125 * q
    term2 = 0.5 * (logdet_m - 0.5 * (logdet_0 + logdet_1))
    return term1 + term2


# -------------------------------
# KDE-based divergences & helpers
# -------------------------------


def select_bandwidth(X, cv_splits=5):
    n, d = X.shape
    h0 = n ** (-1.0 / (d + 4.0))  # Scott
    grid = h0 * np.logspace(-0.5, 0.5, 25)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=0)
    search = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": grid}, cv=cv, n_jobs=1
    )
    search.fit(X)
    return float(search.best_params_["bandwidth"])


def kde_kl_divergence(Xp, Xq, hp, hq, grid_bins=200, margin=3.0):
    """Non-parametric estimate of D_KL(P||Q) using KDE on a finite grid (2D)."""
    X = np.vstack([Xp, Xq])
    mins = X.min(axis=0) - margin
    maxs = X.max(axis=0) + margin

    xs = np.linspace(mins[0], maxs[0], grid_bins)
    ys = np.linspace(mins[1], maxs[1], grid_bins)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.column_stack([XX.ravel(), YY.ravel()])

    kde_p = KernelDensity(bandwidth=hp, kernel="gaussian").fit(Xp)
    kde_q = KernelDensity(bandwidth=hq, kernel="gaussian").fit(Xq)

    logp = kde_p.score_samples(grid)
    logq = kde_q.score_samples(grid)

    p = np.exp(logp)
    q = np.exp(logq)
    dx = (maxs[0] - mins[0]) / (grid_bins - 1)
    dy = (maxs[1] - mins[1]) / (grid_bins - 1)
    cell_area = dx * dy

    p = p / (p.sum() * cell_area)
    q = q / (q.sum() * cell_area)

    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    dkl = np.sum(p * (np.log(p) - np.log(q))) * cell_area
    return float(dkl)


def kde_kl_divergence_mc(Xp, Xq, hp, hq, n_samples=50_000, seed=0):
    kde_p = KernelDensity(bandwidth=hp, kernel="gaussian").fit(Xp)
    kde_q = KernelDensity(bandwidth=hq, kernel="gaussian").fit(Xq)
    Z = kde_p.sample(n_samples, random_state=seed)  # X ~ P
    lp = kde_p.score_samples(Z)
    lq = kde_q.score_samples(Z)
    return float(np.mean(lp - lq))


def kde_js_divergence_mc(Xp, Xq, hp, hq, n_samples=50_000, seed=0):
    kde_p = KernelDensity(bandwidth=hp, kernel="gaussian").fit(Xp)
    kde_q = KernelDensity(bandwidth=hq, kernel="gaussian").fit(Xq)

    n_p = n_samples // 2
    n_q = n_samples - n_p

    Zp = kde_p.sample(n_p, random_state=seed)
    lp_p = kde_p.score_samples(Zp)
    lq_p = kde_q.score_samples(Zp)
    logm_p = np.logaddexp(lp_p, lq_p) - np.log(2.0)

    Zq = kde_q.sample(n_q, random_state=seed)
    lq_q = kde_q.score_samples(Zq)
    lp_q = kde_p.score_samples(Zq)
    logm_q = np.logaddexp(lp_q, lq_q) - np.log(2.0)

    kl_p_m = np.mean(lp_p - logm_p)
    kl_q_m = np.mean(lq_q - logm_q)
    return float(0.5 * (kl_p_m + kl_q_m))


def kde_overlap_coefficient_mc(Xp, Xq, hp, hq, n_samples=50_000, seed=0):
    """Returns (overlap, tv) where overlap = ∫ min(p,q) dx ∈ [0,1] and tv is total variation."""
    kde_p = KernelDensity(bandwidth=hp, kernel="gaussian").fit(Xp)
    kde_q = KernelDensity(bandwidth=hq, kernel="gaussian").fit(Xq)

    n_p = n_samples // 2
    n_q = n_samples - n_p

    Zp = kde_p.sample(n_p, random_state=seed)
    Zq = kde_q.sample(n_q, random_state=seed)
    Z = np.vstack([Zp, Zq])

    lp = kde_p.score_samples(Z)
    lq = kde_q.score_samples(Z)

    tv = 0.5 * np.mean(np.abs(np.tanh(0.5 * (lp - lq))))
    overlap = 1.0 - tv
    return float(overlap), float(tv)


# -----------------------------------
# kNN-based KL, JS; MMD and Energy U
# -----------------------------------


def _kth_nn_dists(A, B, k=1, exclude_self=False, metric="euclidean"):
    extra = 1 if exclude_self else 0
    neigh = NearestNeighbors(n_neighbors=k + extra, metric=metric).fit(B)
    dists, idxs = neigh.kneighbors(A, return_distance=True)
    if exclude_self:
        mask_nonzero = dists > 0
        out = np.empty(len(A))
        for i in range(len(A)):
            di = dists[i, mask_nonzero[i]]
            if len(di) < k:
                di = np.pad(
                    di, (0, k - len(di)), constant_values=di.max() if di.size else 1.0
                )
            out[i] = di[k - 1]
        return out
    else:
        return dists[:, k - 1]


def knn_kl_perezcruz(X, Y, k=1, x_subset_of_y=False, metric="euclidean"):
    n, m = len(X), len(Y)
    d = X.shape[1]
    rho = _kth_nn_dists(X, X, k=k, exclude_self=True, metric=metric)
    nu = _kth_nn_dists(X, Y, k=k, exclude_self=x_subset_of_y, metric=metric)
    eps = np.finfo(float).tiny
    kl = d * np.mean(np.log((nu + eps) / (rho + eps))) + np.log(m / (n - 1.0))
    return float(kl)


def js_knn(X, Y, k=1, metric="euclidean", seed=0):
    rng = np.random.default_rng(seed)
    n, m = len(X), len(Y)
    t = min(n, m)
    Xb = X[rng.choice(n, size=t, replace=False)]
    Yb = Y[rng.choice(m, size=t, replace=False)]
    M = np.vstack([Xb, Yb])

    d_pm = knn_kl_perezcruz(Xb, M, k=k, x_subset_of_y=True, metric=metric)
    d_qm = knn_kl_perezcruz(Yb, M, k=k, x_subset_of_y=True, metric=metric)
    return 0.5 * (d_pm + d_qm)


def mmd_rbf(X, Y, sigma=None, unbiased=True):
    Z = np.vstack([X, Y])
    D2 = pairwise_distances(Z, squared=True)
    if sigma is None:
        n = len(X)
        m = len(Y)
        D2_xy = D2[:n, n:]
        med = np.median(D2_xy[D2_xy > 0])
        sigma = np.sqrt(0.5 * med) if med > 0 else 1.0
    K = np.exp(-D2 / (2.0 * sigma**2))
    n = len(X)
    m = len(Y)
    Kxx = K[:n, :n]
    Kyy = K[n:, n:]
    Kxy = K[:n, n:]

    if unbiased:
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        mmd2 = Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1)) - 2.0 * Kxy.mean()
    else:
        mmd2 = Kxx.mean() + Kyy.mean() - 2.0 * Kxy.mean()
    return float(max(mmd2, 0.0)), sigma


def energy_distance_u(X, Y):
    n, m = len(X), len(Y)
    Dxx = pairwise_distances(X)
    Dyy = pairwise_distances(Y)
    Dxy = pairwise_distances(X, Y)
    np.fill_diagonal(Dxx, 0.0)
    np.fill_diagonal(Dyy, 0.0)
    term_xy = 2.0 * Dxy.mean()
    term_xx = Dxx.sum() / (n * (n - 1)) if n > 1 else 0.0
    term_yy = Dyy.sum() / (m * (m - 1)) if m > 1 else 0.0
    ed = term_xy - term_xx - term_yy
    return float(max(ed, 0.0))


# -------------------------
# t-SNE helpers (correct P/Q)
# -------------------------


def build_P(X, perplexity=30.0, tol=1e-5, max_iter=50):
    n = X.shape[0]
    D2 = pairwise_distances(X, squared=True)
    np.fill_diagonal(D2, np.inf)  # exclude i=j when building rows

    P = np.zeros((n, n), dtype=float)
    target = np.log(perplexity)  # match entropy = log(perplexity)

    for i in range(n):
        Di = D2[i].copy()
        beta = 1.0  # beta = 1/(2*sigma^2)
        betamin, betamax = -np.inf, np.inf

        for _ in range(max_iter):
            Pi = np.exp(-beta * Di)
            Pi[i] = 0.0
            s = Pi.sum()
            if s == 0.0:
                betamax = beta
                beta = beta / 2.0 if np.isinf(betamin) else (beta + betamin) / 2.0
                continue

            Pi /= s
            H = -np.sum(Pi * np.log(Pi + 1e-12))
            diff = H - target
            if abs(diff) < tol:
                break
            if diff > 0:
                betamin = beta
                beta = 2 * beta if np.isinf(betamax) else (beta + betamax) / 2.0
            else:
                betamax = beta
                beta = (beta + betamin) / 2.0 if not np.isinf(betamin) else beta / 2.0
        P[i] = Pi

    P = (P + P.T) / (2.0 * n)
    P = np.clip(P, 1e-12, None)
    P /= P.sum()
    return P


def build_Q(Y):
    D2 = pairwise_distances(Y, squared=True)
    num = 1.0 / (1.0 + D2)
    np.fill_diagonal(num, 0.0)
    num = np.clip(num, 1e-12, None)
    Q = num / num.sum()
    return Q


def kl_divergence(P, Q):
    eps = 1e-12
    return float(np.sum(P * (np.log(P + eps) - np.log(Q + eps))))


# ==================================
# Demo scenarios and annotated output
# ==================================
if __name__ == "__main__":
    seed = 42

    # --- Small t-SNE sanity check ---
    X_hd, _ = make_blobs(
        n_samples=50, centers=[(1.2,) * 5], cluster_std=[1.1], random_state=seed
    )
    reducer = TSNE(n_components=2, random_state=seed)
    Y_ld = reducer.fit_transform(X_hd)
    P = build_P(X_hd, perplexity=15)
    Q = build_Q(Y_ld)
    KL_tsne = kl_divergence(P, Q)
    tw = trustworthiness(X_hd, Y_ld, n_neighbors=10)
    neigh_X = NearestNeighbors(n_neighbors=10, metric="euclidean").fit(X_hd)
    dists_X, idxs_X = neigh_X.kneighbors(X_hd, return_distance=True)
    neigh_Y = NearestNeighbors(n_neighbors=10, metric="euclidean").fit(Y_ld)
    dists_Y, idxs_Y = neigh_Y.kneighbors(Y_ld, return_distance=True)

    print("t-SNE alignment metrics:")
    print(f"  KL(P||Q): {KL_tsne:.6f} nats  (lower is better)")
    print(f"  Trustworthiness@10: {tw:.4f}  (1.0 is best)\n")
    print(f"  High-dimensional neighbors: {idxs_X[0]}")
    print(f"  Low-dimensional neighbors: {idxs_Y[0]}")

    # --- Overlapping cluster scenarios ---
    # You can toggle among these three setups

    # Case A: moderately overlapping isotropic blobs
    X, y = make_blobs(
        n_samples=400,
        centers=[(0, 0), (1.2, 1.2)],
        cluster_std=[1.1, 1.1],
        random_state=42,
    )

    # Case B: well-separated isotropic blobs
    # X, y = make_blobs(n_samples=400, centers=[(0, 0), (2.2, 2.2)], cluster_std=[1.1, 1.1], random_state=42)

    # Case C: anisotropic blobs
    # X, y = make_blobs(n_samples=400, centers=[(0, 0), (1.2, 1.2)], cluster_std=[(0.7, 2.3), (0.5, 3.5)], random_state=42)

    X1 = X[y == 0].copy()
    X2 = X[y == 1].copy()

    # Bandwidths (unscaled and scaled)
    h1 = select_bandwidth(X1)
    h2 = select_bandwidth(X2)

    scaler = StandardScaler().fit(np.vstack([X1, X2]))
    Xp = scaler.transform(X1)
    Xq = scaler.transform(X2)
    hp = select_bandwidth(Xp)
    hq = select_bandwidth(Xq)

    # --- Gaussian parametrics ---
    mu1, S1 = mean_and_cov(X1)
    mu2, S2 = mean_and_cov(X2)
    kl_12 = kl_gaussians(mu1, S1, mu2, S2)
    kl_21 = kl_gaussians(mu2, S2, mu1, S1)
    bhatt = bhattacharyya_gaussians(mu1, S1, mu2, S2)

    # --- KDE nonparametrics (UNSCALED) ---
    js_mc = kde_js_divergence_mc(X1, X2, h1, h2)
    overlap, tv = kde_overlap_coefficient_mc(X1, X2, h1, h2)
    kl12_mc = kde_kl_divergence_mc(X1, X2, h1, h2)
    kl21_mc = kde_kl_divergence_mc(X2, X1, h2, h1)
    kl12_grid = kde_kl_divergence(X1, X2, h1, h2)
    kl21_grid = kde_kl_divergence(X2, X1, h2, h1)

    # --- KDE nonparametrics (SCALED) ---
    js_mc_scaled = kde_js_divergence_mc(Xp, Xq, hp, hq)
    overlap_s, tv_s = kde_overlap_coefficient_mc(Xp, Xq, hp, hq)
    kl12_mc_s = kde_kl_divergence_mc(Xp, Xq, hp, hq)
    kl21_mc_s = kde_kl_divergence_mc(Xq, Xp, hq, hp)
    kl12_grid_s = kde_kl_divergence(Xp, Xq, hp, hq)
    kl21_grid_s = kde_kl_divergence(Xq, Xp, hq, hp)

    # --- kNN, MMD, Energy ---
    js_knn_ = js_knn(X1, X2, k=3)
    mmd2, sigma_used = mmd_rbf(X1, X2, sigma=None, unbiased=True)
    ed = energy_distance_u(X1, X2)

    # ------------
    # Annotations
    # ------------
    print("Parametric Gaussian comparisons:")
    print(f"  KL(1||2): {kl_12:.6f} nats\n  KL(2||1): {kl_21:.6f} nats")
    print(
        f"  Bhattacharyya distance: {bhatt:.6f}  (0 = identical, larger = more separable)\n"
    )

    print("KDE (unscaled space):")
    print(
        f"  JS_MC: {js_mc:.6f} nats   (0 = identical; ln(2)≈0.693 = max for disjoint)"
    )
    print(f"  Overlap coefficient: {overlap:.4f}  (1 = identical, 0 = no overlap)")
    print(f"  Total variation: {tv:.4f}       (0 = identical, 1 = disjoint)")
    print(f"  KL_MC  (1||2): {kl12_mc:.6f} nats,  KL_MC  (2||1): {kl21_mc:.6f} nats")
    print(
        f"  KL_Grid(1||2): {kl12_grid:.6f} nats,  KL_Grid(2||1): {kl21_grid:.6f} nats\n"
    )

    print("KDE (scaled space: StandardScaler):")
    print(f"  JS_MC: {js_mc_scaled:.6f} nats")
    print(f"  Overlap: {overlap_s:.4f},  TV: {tv_s:.4f}")
    print(
        f"  KL_MC  (1||2): {kl12_mc_s:.6f} nats,  KL_MC  (2||1): {kl21_mc_s:.6f} nats"
    )
    print(
        f"  KL_Grid(1||2): {kl12_grid_s:.6f} nats,  KL_Grid(2||1): {kl21_grid_s:.6f} nats\n"
    )

    print("kNN / kernel / metric distances:")
    print(f"  JS_kNN(k=3): {js_knn_:.6f} nats")
    print(f"  MMD^2 (RBF, unbiased): {mmd2:.6f}  (sigma≈{sigma_used:.4f})")
    print(f"  Energy distance (unbiased): {ed:.6f}\n")

    # -------------
    # Quick visuals
    # -------------
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], s=18, c=y)
    plt.title("Two Partially Overlapping Clusters (make_blobs)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.gca().set_aspect("equal", adjustable="box")

    plt.figure()
    plt.imshow(P, interpolation="nearest")
    plt.title("t-SNE: P (high-D affinities)")

    plt.figure()
    plt.imshow(Q, interpolation="nearest")
    plt.title("t-SNE: Q (low-D affinities)")

    plt.show()
