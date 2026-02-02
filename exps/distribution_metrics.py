import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import slogdet, solve
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull
from scipy.stats import chi2
from shapely.geometry import MultiPoint, Polygon
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import StandardScaler

"""
Bounded metrics
js:  Jensen-Shannon Divergence
bc:  Bhattacharyya Coefficient
bc:  Bhattacharyya Distance
jac: Jaccard Index or IOU
ovl: Overlap coefficient / Szymkiewicz-Simpson

Bounds:
- js  [0,ln2] 0 iff P=Q (not overlap), ln2 iff P,Q disjoint
- bc  [0,1] 1 iff P=Q (not overlap), 0 iff P,Q disjoint
- bd  [0,inf] 0 iff P=Q (not overlap), inf iff P,Q disjoint
- jac [0,1] 1 iff P=Q (not overlap), 0 iff P,Q disjoint
- ovl [0,1] 1 iff P=Q (and overlap), 0 iff P,Q disjoint
"""


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


def bhattacharyya_distance_gaussians(mu0, S0, mu1, S1):
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


def bhattacharyya_coefficient_from_distance(distance):
    # Bhattacharyya coefficient [0,1] (1 = identical, 0 = disjoint)
    return np.exp(-distance)


def _in_ellipse(points, mu, S, alpha):
    """
    Stable and efficient way to calculate Mahalanobis distance using Cholesky.

    Returns a boolean mask indicating which points lie inside the α-level ellipse
    (confidence region) of a d-dimensional Gaussian N(mu, S).

    Mathematically, the α-level set is
        E = { x : (x - mu)^T S^{-1} (x - mu) <= χ²_{d,α} },
    where χ²_{d,α} is the α quantile (percent point function) of the chi-square
    distribution with d degrees of freedom.
    """
    d = mu.shape[0]
    thresh = chi2.ppf(alpha, df=d)  # inv cdf
    L = np.linalg.cholesky(S)
    Xm = points - mu
    v = np.linalg.solve(L, Xm.T)
    v = np.linalg.solve(L.T, v)
    maha2 = np.sum(v**2, axis=0)  # Cholesky-based Mahalanobis distance
    return maha2 <= thresh


def sample_box(mu0, S0, mu1, S1, pad=3.5):
    std0 = np.sqrt(np.diag(S0))
    std1 = np.sqrt(np.diag(S1))
    lo = np.minimum(mu0 - pad * std0, mu1 - pad * std1)
    hi = np.maximum(mu0 + pad * std0, mu1 + pad * std1)
    return lo, hi


def overlap_coeff_gaussian_alpha(
    mu0, S0, mu1, S1, alpha=0.95, n_grid=120000, pad=3.5, seed=0
):
    """Szymkiewicz-Simpson overlap coefficient on α–level sets."""
    lo, hi = sample_box(mu0, S0, mu1, S1, pad)
    rng = np.random.default_rng(seed)
    P = rng.uniform(lo, hi, size=(n_grid, 2))
    m0 = _in_ellipse(P, mu0, S0, alpha)
    m1 = _in_ellipse(P, mu1, S1, alpha)
    inter = np.count_nonzero(m0 & m1)
    a0 = np.count_nonzero(m0)
    a1 = np.count_nonzero(m1)
    denom = min(a0, a1)
    ovl = 0.0 if denom == 0 else inter / denom
    return round(ovl, 3)


def containment_alpha(mu0, S0, mu1, S1, alpha=0.95, n_grid=120000, pad=3.5, seed=0):
    """Directed containment (A→B, B→A) on α–level sets."""
    lo, hi = sample_box(mu0, S0, mu1, S1, pad)
    rng = np.random.default_rng(seed)
    P = rng.uniform(lo, hi, size=(n_grid, 2))
    m0 = _in_ellipse(P, mu0, S0, alpha)
    m1 = _in_ellipse(P, mu1, S1, alpha)
    inter = np.count_nonzero(m0 & m1)
    a0 = np.count_nonzero(m0)
    a1 = np.count_nonzero(m1)
    C01 = 0.0 if a0 == 0 else inter / a0  # fraction of A inside B
    C10 = 0.0 if a1 == 0 else inter / a1  # fraction of B inside A
    return C01, C10


def jaccard_alpha(mu0, S0, mu1, S1, alpha=0.95, n_grid=120000, pad=0.0, seed=0):
    lo, hi = sample_box(mu0, S0, mu1, S1, pad)
    rng = np.random.default_rng(seed)
    P = rng.uniform(lo, hi, size=(n_grid, 2))
    m0 = _in_ellipse(P, mu0, S0, alpha)
    m1 = _in_ellipse(P, mu1, S1, alpha)
    inter = np.count_nonzero(m0 & m1)
    union = np.count_nonzero(m0 | m1)
    return 0.0 if union == 0 else inter / union


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


def kde_kl_divergence_grid(Xp, Xq, bandwidth="silverman", grid_bins=200, margin=3.0):
    """Non-parametric estimate of D_KL(P||Q) using KDE on a finite grid (2D)."""
    if isinstance(bandwidth, (tuple, list)):
        hp, hq = bandwidth
    else:
        hp = hq = bandwidth  # float or {"silverman","scott"}
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

    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    kl = np.sum(p * (np.log(p) - np.log(q))) * cell_area
    return float(kl)


def kde_kl_divergence_mc(Xp, Xq, bandwidth="silverman", n_samples=50_000, seed=0):
    if isinstance(bandwidth, (tuple, list)):
        hp, hq = bandwidth
    else:
        hp = hq = bandwidth  # float or {"silverman","scott"}

    kde_p = KernelDensity(bandwidth=hp, kernel="gaussian").fit(Xp)
    kde_q = KernelDensity(bandwidth=hq, kernel="gaussian").fit(Xq)
    Z = kde_p.sample(n_samples, random_state=seed)  # X ~ P
    lp = kde_p.score_samples(Z)
    lq = kde_q.score_samples(Z)
    return float(np.mean(lp - lq))


def kde_js_divergence_mc(Xp, Xq, bandwidth="silverman", n_samples=50_000, seed=0):
    """
    Monte Carlo JS divergence using Gaussian KDEs.
    n_samples is the TOTAL number of samples (split ~half from each KDE).

    Other metrics: KL divergence, JS divergence, Bhattacharyya coefficient/distance, Earth mover's distance
    JS divergence and Bhattacharyya coefficient are bounded.
    """
    if isinstance(bandwidth, (tuple, list)):
        hp, hq = bandwidth
    else:
        hp = hq = bandwidth  # float or {"silverman","scott"}

    kde_p = KernelDensity(bandwidth=hp, kernel="gaussian").fit(Xp)
    kde_q = KernelDensity(bandwidth=hq, kernel="gaussian").fit(Xq)

    # Split total sample budget between P and Q
    n_p = n_samples // 2
    n_q = n_samples - n_p

    # Samples from P
    Zp = kde_p.sample(n_p, random_state=seed)
    lp_p = kde_p.score_samples(Zp)  # log p(z) for z~P
    lq_p = kde_q.score_samples(Zp)  # log q(z) for z~P
    logm_p = np.logaddexp(lp_p, lq_p) - np.log(2.0)  # log((p+q)/2)

    # Samples from Q
    Zq = kde_q.sample(n_q, random_state=seed)
    lq_q = kde_q.score_samples(Zq)  # log q(z) for z~Q
    lp_q = kde_p.score_samples(Zq)  # log p(z) for z~Q
    logm_q = np.logaddexp(lp_q, lq_q) - np.log(2.0)

    # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    kl_p_m = np.mean(lp_p - logm_p)
    kl_q_m = np.mean(lq_q - logm_q)
    js = 0.5 * (kl_p_m + kl_q_m)

    return float(js)


def kde_js_divergence(reduced, labels, l1, l2):
    import behavior.utils as bu

    X1 = reduced[labels == l1]
    X2 = reduced[labels == l2]

    # Normalize points
    scaler = StandardScaler().fit(np.vstack([X1, X2]))
    Xp = scaler.transform(X1)
    Xq = scaler.transform(X2)

    # Calculate bandwidths
    hp = select_bandwidth(Xp)
    hq = select_bandwidth(Xq)

    # Calculate KDE divergence
    divergence = kde_js_divergence_mc(Xp, Xq, (hp, hq))
    return divergence


def kde_overlap_coefficient_mc(Xp, Xq, bandwidth="silverman", n_samples=50_000, seed=0):
    """Returns (overlap, tv) where overlap = ∫ min(p,q) dx ∈ [0,1] and tv is total variation."""
    if isinstance(bandwidth, (tuple, list)):
        hp, hq = bandwidth
    else:
        hp = hq = bandwidth  # float or {"silverman","scott"}
    kde_p = KernelDensity(bandwidth=hp, kernel="gaussian").fit(Xp)
    kde_q = KernelDensity(bandwidth=hq, kernel="gaussian").fit(Xq)

    n_p = n_samples // 2
    n_q = n_samples - n_p

    Zp = kde_p.sample(n_p, random_state=seed)
    Zq = kde_q.sample(n_q, random_state=seed)
    Z = np.vstack([Zp, Zq])

    lp = kde_p.score_samples(Z)
    lq = kde_q.score_samples(Z)

    tv = np.mean(np.abs(np.tanh(0.5 * (lp - lq))))
    overlap = 1.0 - tv
    return float(overlap), float(tv)


def overlap_coeff_kde_is(X0, X1, bandwidth="silverman", n_samples=50000, seed=0):
    """
    Estimate OVL = ∫ min(f,g) dx where f,g are KDEs of samples X0, X1.
    Uses importance sampling (is) from the mixture q = 0.5 f + 0.5 g.

    Parameters
    ----------
    X0, X1 : array-like of shape (n_i, d)
        Samples belonging to cluster 0 and 1.
    bandwidth0, bandwidth1 : float or None
        KDE bandwidths (sklearn isotropic). If None, use Scott/Silverman-like rule.
    n_samples : int
        Total number of samples to draw from the mixture proposal.
    seed : int
        RNG seed.

    Returns
    -------
    ovl : float in [0, 1]
    """

    if isinstance(bandwidth, (tuple, list)):
        hp, hq = bandwidth
    else:
        hp = hq = bandwidth  # float or {"silverman","scott"}

    kde0 = KernelDensity(bandwidth=hp, kernel="gaussian").fit(X0)
    kde1 = KernelDensity(bandwidth=hq, kernel="gaussian").fit(X1)

    # sample from the mixture q = 0.5 f + 0.5 g
    n0 = n_samples // 2
    n1 = n_samples - n0
    S0 = kde0.sample(n0, random_state=seed)
    S1 = kde1.sample(n1, random_state=seed)
    S = np.vstack([S0, S1])

    # log densities under f and g
    logf = kde0.score_samples(S)
    logg = kde1.score_samples(S)

    # log q = log(0.5*exp(logf) + 0.5*exp(logg)) = logsumexp([logf,logg]) - log(2)
    # implement stable logsumexp for two terms
    m = np.maximum(logf, logg)
    logq = m + np.log(0.5 * np.exp(logf - m) + 0.5 * np.exp(logg - m))

    # min(f,g) in log-domain
    log_min_fg = np.minimum(logf, logg)

    # importance weights: min(f,g)/q
    w = np.exp(log_min_fg - logq)

    # Monte Carlo estimate of the integral (bounded in [0,1])
    ovl = float(np.mean(w))
    # numerical guard
    return min(max(ovl, 0.0), 1.0)


def overlap_coeff_kde_grid(
    X0, X1, bandwidth="silverman", n_grid=60000, pad=0.0, seed=0
):
    """
    Uniform-grid Monte Carlo on a padded bounding box (most convenient for d<=3).
    """
    if isinstance(bandwidth, (tuple, list)):
        hp, hq = bandwidth
    else:
        hp = hq = bandwidth  # float or {"silverman","scott"}

    kde0 = KernelDensity(bandwidth=hp, kernel="gaussian").fit(X0)
    kde1 = KernelDensity(bandwidth=hq, kernel="gaussian").fit(X1)

    # padded box around both clusters
    X0, X1 = np.asarray(X0), np.asarray(X1)
    d = X0.shape[1]
    allX = np.vstack([X0, X1])
    sd = np.std(allX, axis=0, ddof=1)
    sd[sd == 0] = 1.0
    lo = np.min(allX, axis=0) - pad * sd
    hi = np.max(allX, axis=0) + pad * sd

    rng = np.random.default_rng(seed)
    U = rng.uniform(lo, hi, size=(n_grid, d))
    logf = kde0.score_samples(U)
    logg = kde1.score_samples(U)
    min_fg = np.exp(np.minimum(logf, logg))

    # integral ≈ volume * average(min_fg)
    volume = float(np.prod(hi - lo))
    ovl = volume * float(np.mean(min_fg))
    # clamp to [0,1] to be safe numerically
    return min(max(ovl, 0.0), 1.0)


def _hdr_log_threshold(kde, alpha=0.95, n_samples=200000, seed=0):
    """
    Return log t such that P_kde{ log f(X) >= log t } = alpha.
    Estimated by sampling from the KDE with the given seed.
    """
    S = kde.sample(n_samples, random_state=seed)
    logd = kde.score_samples(S)
    # HDR keeps top alpha mass -> threshold is the (1 - alpha) quantile of log-densities
    log_t = np.quantile(logd, 1.0 - alpha)
    return float(log_t)


def overlap_coeff_kde_hdr(
    XP, XQ, alpha=0.95, n_samples=200000, bandwidth="silverman", seed=0
):
    """
    KDE-based containment overlap on alpha-HDRs (returns value in [0,1]).
    = 1 when one alpha-HDR is contained in the other; = 0 when HDRs are disjoint.
    HDR (High Density Region)

    bandwidth: str|float or (hp, hq)
        Pass "silverman" / "scott" (new sklearn behavior) or a float.
        Tuple/list applies different bandwidths to P and Q.
    """
    if isinstance(bandwidth, (tuple, list)):
        hp, hq = bandwidth
    else:
        hp = hq = bandwidth

    kde_p = KernelDensity(bandwidth=hp, kernel="gaussian").fit(np.asarray(XP))
    kde_q = KernelDensity(bandwidth=hq, kernel="gaussian").fit(np.asarray(XQ))

    # thresholds for each alpha-HDR (sample from each KDE)
    log_t_p = _hdr_log_threshold(
        kde_p, alpha=alpha, n_samples=n_samples // 2, seed=seed
    )
    log_t_q = _hdr_log_threshold(
        kde_q, alpha=alpha, n_samples=n_samples // 2, seed=seed
    )

    # Estimate P_f(S_g): draw from f (P), test against g's (Q) HDR
    SP = kde_p.sample(n_samples, random_state=seed)
    in_q = kde_q.score_samples(SP) >= log_t_q
    p_pq = float(np.mean(in_q))  # P_P( S_Q )

    # Estimate P_g(S_f): draw from g (Q), test against f's (P) HDR
    SQ = kde_q.sample(n_samples, random_state=seed)
    in_p = kde_p.score_samples(SQ) >= log_t_p
    p_qp = float(np.mean(in_p))  # P_Q( S_P )

    c_pq = min(1.0, p_pq / alpha)
    c_qp = min(1.0, p_qp / alpha)
    ovl = max(c_pq, c_qp)

    return round(ovl, 3)


def _rule_bandwidth(X, rule="scott"):
    X = np.asarray(X)
    n, d = X.shape
    if n < 2:  # fallback
        return 1.0
    sd = np.std(X, axis=0, ddof=1)
    sd[sd == 0] = 1.0
    if rule == "silverman":
        # Silverman’s multivariate rule (scalar isotropic)
        s = float(np.mean(sd))
        return s * (n * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0))
    # default: Scott
    s = float(np.mean(sd))
    return s * n ** (-1.0 / (d + 4.0))


def _kde(X, bandwidth):
    # bandwidth: "scott" | "silverman" | float
    return KernelDensity(bandwidth=bandwidth, kernel="gaussian").fit(np.asarray(X))


def _hdr_log_threshold(kde, alpha=0.95, n_samples=200000, seed=0):
    S = kde.sample(n_samples, random_state=seed)
    logd = kde.score_samples(S)
    return float(np.quantile(logd, 1.0 - alpha))


def _containment_fraction(kde_from, kde_to, log_t_to, n_samples=200000, seed=0):
    S = kde_from.sample(n_samples, random_state=seed)
    return float(np.mean(kde_to.score_samples(S) >= log_t_to))


def overlap_coeff_kde_hdr_containment(
    XP,
    XQ,
    alpha=0.95,
    n_samples=200000,
    bandwidth="scott",
    seed=0,
    ensure_containment=True,
    inflate_cap=8.0,
):
    """
    KDE-based HDR containment overlap in [0,1].
    If ensure_containment=True, adaptively inflates bandwidths just enough to avoid
    fragmented HDRs (cap controls the max inflation factor).
    """
    # unpack bandwidth
    if isinstance(bandwidth, (tuple, list)):
        hp_raw, hq_raw = bandwidth
    else:
        hp_raw = hq_raw = bandwidth

    # turn "scott"/"silverman" into floats (so we can inflate), else keep float
    hp0 = _rule_bandwidth(XP, hp_raw) if isinstance(hp_raw, str) else float(hp_raw)
    hq0 = _rule_bandwidth(XQ, hq_raw) if isinstance(hq_raw, str) else float(hq_raw)

    # start with no inflation
    mult_p = 1.0
    mult_q = 1.0

    def fit_with(mult_p, mult_q):
        kde_p = _kde(XP, hp0 * mult_p)
        kde_q = _kde(XQ, hq0 * mult_q)
        log_t_p = _hdr_log_threshold(kde_p, alpha, n_samples // 2, seed)
        log_t_q = _hdr_log_threshold(kde_q, alpha, n_samples // 2, seed)
        p_pq = _containment_fraction(kde_p, kde_q, log_t_q, n_samples, seed)  # P_P(S_Q)
        p_qp = _containment_fraction(kde_q, kde_p, log_t_p, n_samples, seed)  # P_Q(S_P)
        c_pq = min(1.0, p_pq / alpha)
        c_qp = min(1.0, p_qp / alpha)
        return kde_p, kde_q, log_t_p, log_t_q, p_pq, p_qp, c_pq, c_qp

    kde_p, kde_q, log_t_p, log_t_q, p_pq, p_qp, c_pq, c_qp = fit_with(mult_p, mult_q)

    if ensure_containment:
        # If one cluster should be inside the other but HDR fragmentation hurts,
        # softly inflate each side until its containment ≥ alpha (or we hit the cap).
        # Geometric growth keeps it fast; we only inflate the *target* side.
        # First try to make P inside Q (raise Q’s smoothing if needed)
        while c_pq < 1.0 and mult_q < inflate_cap:
            mult_q *= 1.5
            kde_p, kde_q, log_t_p, log_t_q, p_pq, p_qp, c_pq, c_qp = fit_with(
                mult_p, mult_q
            )
        # Then try the other direction
        while c_qp < 1.0 and mult_p < inflate_cap:
            mult_p *= 1.5
            kde_p, kde_q, log_t_p, log_t_q, p_pq, p_qp, c_pq, c_qp = fit_with(
                mult_p, mult_q
            )

    ovl = max(c_pq, c_qp)
    return float(ovl)


def hull_polygon(points: np.ndarray) -> Polygon:
    # Handle degenerate small sets gracefully
    if len(points) < 3:
        return MultiPoint(points).convex_hull  # a point/segment/polygon
    hull = ConvexHull(points)
    return Polygon(points[hull.vertices])


def overlap_coeff_convex_hull(X1, X2, buffer=False):
    # list(P1.exterior.coords) # list[tuple] coords
    P1 = hull_polygon(X1)
    P2 = hull_polygon(X2)

    if buffer:
        min_area = min(P1.area, P2.area)
        buffer_size = 0.01 * min_area
        P1 = P1.buffer(buffer_size)
        P2 = P2.buffer(buffer_size)
    intersection_area = P1.intersection(P2).area
    min_area = min(P1.area, P2.area)
    ovl = 0.0 if min_area == 0 else intersection_area / min_area
    return round(ovl, 3)
    # union_area        = P1.union(P2).area
    # iou = 0.0 if union_area == 0 else intersection_area / union_area


def _grid_bounds(X0, X1, pad=0.02):
    """Tight combined bounds with a small relative padding."""
    X = np.vstack([X0, X1])
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    return lo - pad * span, hi + pad * span


def _disk_structuring_element(radius_px):
    r = int(np.ceil(radius_px))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= radius_px * radius_px


def overlap_coeff_raster_occupancy(
    X0,
    X1,
    cell_size=None,  # size of a grid cell in data units (float)
    bins=64,  # or use number of bins per axis
    bounds=None,
    pad=0.02,
    dilate_by=0.0,  # dilation radius in data units (0 = no dilation)
):
    """
    Rasterized Szymkiewicz-Simpson overlap coefficient (occupancy) for 2D points.
    Adds an optional dilation radius (in data units) to make the metric robust
    to sampling sparsity and grid aliasing.
    """
    X0 = np.asarray(X0, float)
    X1 = np.asarray(X1, float)

    if bounds is None:
        lo, hi = _grid_bounds(X0, X1, pad=pad)
    else:
        lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)

    # Build grid
    if cell_size is not None:
        xedges = np.arange(lo[0], hi[0] + cell_size, cell_size)
        yedges = np.arange(lo[1], hi[1] + cell_size, cell_size)
    else:
        # fall back to 'bins' (int or (nx, ny))
        if np.isscalar(bins):
            nx = ny = int(bins)
        else:
            nx, ny = int(bins[0]), int(bins[1])
        xedges = np.linspace(lo[0], hi[0], nx + 1)
        yedges = np.linspace(lo[1], hi[1], ny + 1)

    H0, _, _ = np.histogram2d(X0[:, 0], X0[:, 1], bins=[xedges, yedges])
    H1, _, _ = np.histogram2d(X1[:, 0], X1[:, 1], bins=[xedges, yedges])
    B0 = H0 > 0
    B1 = H1 > 0

    # Optional dilation in *data units*
    if dilate_by > 0.0:
        # convert data distance to pixel radius separately for x and y, use max
        cell_w = np.diff(xedges).mean()
        cell_h = np.diff(yedges).mean()
        radius_px = max(dilate_by / cell_w, dilate_by / cell_h)
        se = _disk_structuring_element(radius_px)
        B0 = binary_dilation(B0, structure=se)
        B1 = binary_dilation(B1, structure=se)

    inter = np.count_nonzero(B0 & B1)
    denom = min(np.count_nonzero(B0), np.count_nonzero(B1))
    return 0.0 if denom == 0 else inter / denom


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

    # disjoin
    X, y = make_blobs(
        n_samples=50,
        centers=[(1.1, 1.1), (10.2, 10.2)],
        cluster_std=[1.1, 1.2],
        random_state=seed,
    )
    # complete overlap
    X, y = make_blobs(
        n_samples=50,
        centers=[(1.1, 1.1), (1.2, 1.2)],
        cluster_std=[1.1, 5.6],
        random_state=seed,
    )
    mask0 = y == 0
    mask1 = y == 1
    plt.figure()
    # plt.scatter(X[:, 0], X[:, 1], s=18, c=y)
    plt.plot(X[mask0, 0], X[mask0, 1], "r*")
    plt.plot(X[mask1, 0], X[mask1, 1], "g*")
    plt.show()

    X1 = X[y == 0]
    X2 = X[y == 1]
    mu1, S1 = mean_and_cov(X1)
    mu2, S2 = mean_and_cov(X2)
    overlap_coeff_convex_hull(X1, X2)
    overlap_coeff_raster_occupancy(X1, X2, bins=80, dilate_by=0.5, pad=0)
    overlap_coeff_gaussian_alpha(mu1, S1, mu2, S2, alpha=0.95)
    overlap_coeff_kde_is(X1, X2)
    overlap_coeff_kde_hdr(X1, X2)
    overlap_coeff_kde_hdr_containment(X1, X2)
    overlap_coeff_kde_grid(X1, X2)
    kde_overlap_coefficient_mc(X1, X2)
    bhatt = bhattacharyya_coefficient_from_distance(
        bhattacharyya_distance_gaussians(mu1, S1, mu2, S2)
    )
    jac = jaccard_alpha(mu1, S1, mu2, S2, alpha=0.95)
    ca = containment_alpha(mu1, S1, mu2, S2, alpha=0.95)
    js = kde_js_divergence(X, y, 0, 1)
    js_test = kde_js_divergence_mc(X1, X2)
    kde = kde_kl_divergence_grid(X1, X2)
    print("done")

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
    bhatt = bhattacharyya_distance_gaussians(mu1, S1, mu2, S2)

    # --- KDE nonparametrics (UNSCALED) ---
    js_mc = kde_js_divergence_mc(X1, X2, (h1, h2))
    overlap, tv = kde_overlap_coefficient_mc(X1, X2, (h1, h2))
    kl12_mc = kde_kl_divergence_mc(X1, X2, (h1, h2))
    kl21_mc = kde_kl_divergence_mc(X2, X1, (h1, h2))
    kl12_grid = kde_kl_divergence_grid(X1, X2, (h1, h2))
    kl21_grid = kde_kl_divergence_grid(X2, X1, (h1, h2))

    # --- KDE nonparametrics (SCALED) ---
    js_mc_scaled = kde_js_divergence_mc(Xp, Xq, (h1, h2))
    overlap_s, tv_s = kde_overlap_coefficient_mc(Xp, Xq, (h1, h2))
    kl12_mc_s = kde_kl_divergence_mc(Xp, Xq, (h1, h2))
    kl21_mc_s = kde_kl_divergence_mc(Xq, Xp, (h1, h2))
    kl12_grid_s = kde_kl_divergence_grid(Xp, Xq, (h1, h2))
    kl21_grid_s = kde_kl_divergence_grid(Xq, Xp, (h1, h2))

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
