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


def select_bandwidth(X, cv_splits=5):
    n, d = X.shape
    h0 = n ** (-1.0 / (d + 4.0))  # Scott
    # h0 *= (4/(d + 2.0))** (1/(d + 4.0)) # Silverman
    grid = h0 * np.logspace(-0.5, 0.5, 25)
    # grid = np.logspace(-2, 1, 40)
    cv = KFold(n_splits=min(cv_splits, n), shuffle=True, random_state=0)
    search = GridSearchCV(
        KernelDensity(kernel="gaussian"), {"bandwidth": grid}, cv=cv, n_jobs=1
    )
    search.fit(X)
    return float(search.best_params_["bandwidth"])


def kde_js_divergence_mc(Xp, Xq, bandwidth=None, n_samples=50_000, seed=0):
    """
    Monte Carlo estimate of JS divergence using Gaussian KDEs.
    n_samples is the TOTAL number of samples (split ~half from each KDE).

    Other metrics: KL divergence, JS divergence, Bhattacharyya coefficient/distance, Earth mover's distance
    JS divergence and Bhattacharyya coefficient are bounded.

    Parameters
    ----------
    Xp, Xq : array-like
        Samples from distributions P and Q.
    bandwidth : float, str, or tuple(float|str, float|str), optional
        Bandwidth(s) for KDEs.
        - If None: uses 'silverman'.
        - If single value float or {"scott", "silverman"}: used for both P and Q.
        - If tuple: interpreted as (bandwidth_P, bandwidth_Q).
    n_samples : int, optional
        Number of Monte Carlo samples.
    seed : int, optional
        Random seed for reproducibility.
    """

    if bandwidth is None:
        hp = hq = "silverman"
    elif isinstance(bandwidth, (tuple, list)):
        hp, hq = bandwidth
    else:
        hp = hq = bandwidth

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


def js_overlap_label(js):
    LN2 = np.log(2.0)
    x = js / LN2  # normalize to [0,1]
    if x < 0.2:
        return "very large overlap"
    elif x < 0.5:
        return "large overlap"
    elif x < 0.8:
        return "small overlap"
    elif x < 0.97:
        return "very small overlap"
    else:
        return "essentially no overlap"


def mean_and_cov(X, eps=1e-6):
    mu = X.mean(axis=0)
    Sigma = np.cov(X, rowvar=False) + eps * np.eye(X.shape[1])
    return mu, Sigma


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


def min_js_divergence_clusters(
    query_feats, keys_feats, keys_labels, bandwidth="silverman"
):
    """Calculate the minimum KDE-based JS divergence between the query class and each class in keys.

    Parameters
    ----------
    query_feats : np.ndarray
        Feature vectors for the query class.
    keys_feats : np.ndarray
        Feature vectors for the key classes.
    keys_labels : np.ndarray
        Labels for the key classes.
    bandwidth : float | str | tuple[float|str, float|str], optional
        Bandwidth(s) to use:
          - float: same numeric bandwidth for both P and Q
          - "silverman" or "scott": same rule-of-thumb for both
          - (hq, hk): separate values/rules for query and key
          - "auto": use a bandwidth selector to pick hq/hk per class

    Returns
    -------
    float
        Minimum JS divergence score across all class pairs.
    """
    min_divergence = float("inf")
    Xq_raw = query_feats

    key_labels = np.unique(keys_labels)
    for i in key_labels:
        Xk_raw = keys_feats[keys_labels == i]

        Xq, Xk = Xq_raw, Xk_raw
        if bandwidth == "auto":
            # Normalize points
            scaler = StandardScaler().fit(np.vstack([Xq_raw, Xk_raw]))
            Xq = scaler.transform(Xq_raw)
            Xk = scaler.transform(Xk_raw)

            # Calculate bandwidths
            hq = select_bandwidth(Xq)
            hk = select_bandwidth(Xk)
        elif isinstance(bandwidth, (tuple, list)):
            hq, hk = bandwidth
        else:
            hq = hk = bandwidth  # float or {"silverman","scott"}

        # Calculate KDE divergence
        divergence = kde_js_divergence_mc(Xq, Xk, (hq, hk))
        min_divergence = min(min_divergence, divergence)

    return min_divergence


def max_ovl_gaussian_clusters(query_feats, keys_feats, keys_labels):
    """Calculate the maximum overlap coefficient between the query class and each class in keys.

    Parameters
    ----------
    query_feats : np.ndarray
        Feature vectors for the query class.
    keys_feats : np.ndarray
        Feature vectors for the key classes.
    keys_labels : np.ndarray
        Labels for the key classes.

    Returns
    -------
    float
        Mmaximum overlap coefficient score across all class pairs.
    """
    max_ovl = 0
    Xq = query_feats
    muq, Sq = mean_and_cov(Xq)

    key_labels = np.unique(keys_labels)
    for i in key_labels:
        Xk = keys_feats[keys_labels == i]
        muk, Sk = mean_and_cov(Xk)

        # Calculate KDE divergence
        ovl = overlap_coeff_gaussian_alpha(muq, Sq, muk, Sk, alpha=0.95)
        max_ovl = max(max_ovl, ovl)

    return max_ovl


def max_ovl_convex_hull_clusters(query_feats, keys_feats, keys_labels):
    """Calculate the maximum overlap coefficient between the query class and each class in keys.

    Parameters
    ----------
    query_feats : np.ndarray
        Feature vectors for the query class.
    keys_feats : np.ndarray
        Feature vectors for the key classes.
    keys_labels : np.ndarray
        Labels for the key classes.

    Returns
    -------
    float
        Mmaximum overlap coefficient score across all class pairs.
    """
    max_ovl = 0
    Xq = query_feats

    key_labels = np.unique(keys_labels)
    for i in key_labels:
        Xk = keys_feats[keys_labels == i]

        # Calculate KDE divergence
        ovl = overlap_coeff_convex_hull(Xq, Xk)
        max_ovl = max(max_ovl, ovl)

    return max_ovl


def max_ovl_kde_hdr(query_feats, keys_feats, keys_labels):
    """Calculate the maximum overlap coefficient between the query class and each class in keys.

    Parameters
    ----------
    query_feats : np.ndarray
        Feature vectors for the query class.
    keys_feats : np.ndarray
        Feature vectors for the key classes.
    keys_labels : np.ndarray
        Labels for the key classes.

    Returns
    -------
    float
        Mmaximum overlap coefficient score across all class pairs.
    """
    max_ovl = 0
    Xq = query_feats

    key_labels = np.unique(keys_labels)
    for i in key_labels:
        Xk = keys_feats[keys_labels == i]

        # Calculate KDE divergence
        ovl = overlap_coeff_kde_hdr(Xq, Xk)
        max_ovl = max(max_ovl, ovl)

    return max_ovl
