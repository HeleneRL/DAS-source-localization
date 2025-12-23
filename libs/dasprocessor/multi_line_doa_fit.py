import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression



'''
Ignore this for now...


'''






def _svd_axis(channel_positions_enu: np.ndarray) -> np.ndarray:
    center = channel_positions_enu.mean(axis=0)
    P = channel_positions_enu - center
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    axis = Vt[0]
    axis /= (np.linalg.norm(axis) + 1e-12)
    # enforce consistent direction (first -> last)
    if np.dot(axis, channel_positions_enu[-1] - channel_positions_enu[0]) < 0:
        axis = -axis
    return axis

def _compute_s(channel_positions_enu: np.ndarray, axis: np.ndarray) -> np.ndarray:
    s = channel_positions_enu @ axis
    s = s - s.min()
    return s

def _fit_shared_slope_and_intercepts(s, t, labels, k):
    """
    Fit t_i ≈ m*s_i + b_{label(i)} with least squares.
    Returns m, intercepts[k], and per-point residuals.
    """
    s = np.asarray(s, float)
    t = np.asarray(t, float)
    n = len(t)

    # Build design matrix: [s, one-hot(cluster)]
    # params = [m, b0, b1, ... b_{k-1}]
    A = np.zeros((n, 1 + k), float)
    A[:, 0] = s
    for i in range(n):
        A[i, 1 + labels[i]] = 1.0

    # Least squares
    params, *_ = np.linalg.lstsq(A, t, rcond=None)
    m = float(params[0])
    intercepts = params[1:].astype(float)

    t_hat = m * s + intercepts[labels]
    residuals = t - t_hat
    return m, intercepts, residuals

def _reassign_by_residuals(s, t, m, intercepts):
    """
    Assign each point to the line (same slope m) with smallest abs residual.
    """
    s = np.asarray(s, float)
    t = np.asarray(t, float)
    k = len(intercepts)
    # residuals to each line: shape (n,k)
    res = np.abs(t[:, None] - (m * s[:, None] + intercepts[None, :]))
    return np.argmin(res, axis=1).astype(int)

def _cluster_initial(s, t, k, alpha_s=0.2, random_state=0):
    """
    Initial clustering to separate 'bands' (paths).
    Uses features [alpha*s_norm, residual_to_global_fit].
    """
    s = np.asarray(s, float)
    t = np.asarray(t, float)

    # global slope baseline
    lr0 = LinearRegression().fit(s.reshape(-1,1), t)
    m0 = float(lr0.coef_[0])
    r0 = t - (m0 * s + float(lr0.intercept_))

    s_norm = (s - s.mean()) / (s.std() + 1e-12)
    feats = np.column_stack([alpha_s * s_norm, r0])

    km = KMeans(n_clusters=k, n_init=30, random_state=random_state)
    labels = km.fit_predict(feats)
    return labels

def _leave_one_out_refine(s, t, labels, k, min_points_per_line=3):
    """
    For each cluster with > min_points_per_line, try leaving out each point
    in that cluster and pick the refit (shared slope + intercepts) with
    smallest total SSE. This prevents a single bad point from skewing m.
    """
    s = np.asarray(s, float)
    t = np.asarray(t, float)
    n = len(t)

    # if any cluster is too small, just return normal fit
    counts = np.bincount(labels, minlength=k)
    if np.any(counts < 2):
        return None  # signal: skip LOO

    best = None  # (SSE, m, intercepts)
    # candidate masks: default "use all", plus leave-one-out per cluster
    candidate_masks = [np.ones(n, dtype=bool)]

    for c in range(k):
        idx_c = np.where(labels == c)[0]
        if idx_c.size > min_points_per_line:
            for j in idx_c:
                msk = np.ones(n, dtype=bool)
                msk[j] = False
                candidate_masks.append(msk)

    for msk in candidate_masks:
        s_m = s[msk]; t_m = t[msk]; lab_m = labels[msk]

        # reindex labels to 0..k-1 remains consistent
        m_fit, b_fit, res = _fit_shared_slope_and_intercepts(s_m, t_m, lab_m, k)
        sse = float(np.sum(res**2))

        if (best is None) or (sse < best[0]):
            best = (sse, m_fit, b_fit)

    return best  # may be None

def _physical_slope_repair_using_middle_y(s, t, labels, m, intercepts, slope_max, min_points_per_line=3):
    """
    Your requested rule:
    - identify the cluster 'responsible' for unphysical slope
      (we interpret as the cluster with largest SSE)
    - sort the y-values (times) in that cluster and use the two middle values
      to build a local slope estimate using the corresponding x-values.
    - refit shared slope + intercepts using that repaired slope as an anchor.

    Note: two middle y-values alone don't define a slope unless we pair them
    with x-values. We use their associated x-values at those indices.
    """
    k = len(intercepts)

    # compute per-cluster SSE under current model
    t_hat = m * s + intercepts[labels]
    res = t - t_hat
    sse_c = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            sse_c.append(-np.inf)
        else:
            sse_c.append(float(np.sum(res[idx]**2)))
    worst_c = int(np.argmax(sse_c))

    idx = np.where(labels == worst_c)[0]
    if idx.size < 2:
        # can't do much
        m_new = np.clip(m, -0.99*slope_max, 0.99*slope_max)
        return m_new, intercepts

    # sort by y (times) as you requested (not by s)
    order = np.argsort(t[idx])
    idx_sorted = idx[order]

    # take the two middle indices
    mid = len(idx_sorted) // 2
    if len(idx_sorted) % 2 == 0:
        i1 = idx_sorted[mid - 1]
        i2 = idx_sorted[mid]
    else:
        i1 = idx_sorted[mid - 1]
        i2 = idx_sorted[mid + 1] if (mid + 1) < len(idx_sorted) else idx_sorted[mid]

    ds = float(s[i2] - s[i1])
    dt = float(t[i2] - t[i1])

    if abs(ds) < 1e-12:
        # fallback: clamp
        m_new = np.clip(m, -0.99*slope_max, 0.99*slope_max)
    else:
        m_new = dt / ds
        # enforce physical limit
        m_new = float(np.clip(m_new, -0.99*slope_max, 0.99*slope_max))

    # With repaired slope, recompute intercepts as per-cluster medians (robust)
    intercepts_new = np.zeros(k, float)
    for c in range(k):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            intercepts_new[c] = 0.0
        else:
            intercepts_new[c] = float(np.median(t[idx_c] - m_new * s[idx_c]))

    return m_new, intercepts_new

def fit_doa_multiline_shared_slope(
    times,
    channel_positions_enu,
    n_lines_target=3,
    min_points_per_line=3,
    alpha_s=0.2,
    n_refine_iter=8,
    sound_speed=1475.0,
    random_state=0,
):
    """
    Fit multiple arrivals as K lines with the SAME slope m (DOA), but different intercepts b_k.

    Returns (slope_final, residuals_final, model_final, extra)

    extra contains:
      - axis, s
      - shared_slope
      - lines: [{"slope": m, "intercept": b_k, "points":[...]}]
      - labels: per-point assignments
    """
    t = np.asarray(times, float)
    pos = np.asarray(channel_positions_enu, float)

    axis = _svd_axis(pos)
    s = _compute_s(pos, axis)

    n = len(t)
    slope_max = 1.0 / sound_speed

    # if too few points: fallback to single line
    if n < n_lines_target * min_points_per_line:
        lr = LinearRegression().fit(s.reshape(-1,1), t)
        m = float(lr.coef_[0])
        b = float(lr.intercept_)
        class SimpleModel:
            def __init__(self, a, b):
                self.coef_ = np.array([a])
                self.intercept_ = b
            def predict(self, X_in):
                X_in = np.asarray(X_in).reshape(-1, 1)
                return X_in[:,0] * self.coef_[0] + self.intercept_
        model = SimpleModel(m, b)
        res = t - model.predict(s.reshape(-1,1))
        extra = {
            "n_lines": 1,
            "axis": axis.tolist(),
            "s": s.tolist(),
            "shared_slope": m,
            "lines": [{"slope": m, "intercept": b, "points": list(range(n))}],
            "labels": np.zeros(n, int).tolist(),
        }
        return m, res, model, extra

    # choose k
    max_k = n // min_points_per_line
    k = int(min(n_lines_target, max_k))
    k = max(k, 1)

    # -------------------------
    # Stage 1: initial clustering
    # -------------------------
    labels = _cluster_initial(s, t, k, alpha_s=alpha_s, random_state=random_state)

    # -------------------------
    # Stage 2: iterative refinement (shared slope + reassignment)
    # -------------------------
    m, intercepts, res = _fit_shared_slope_and_intercepts(s, t, labels, k)

    # physical check + repair (your rule)
    if abs(m) > slope_max:
        m, intercepts = _physical_slope_repair_using_middle_y(
            s, t, labels, m, intercepts, slope_max, min_points_per_line=min_points_per_line
        )

    for _ in range(n_refine_iter):
        # reassign to nearest line
        new_labels = _reassign_by_residuals(s, t, m, intercepts)

        # stop if stable
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # refit shared slope/intercepts
        m, intercepts, res = _fit_shared_slope_and_intercepts(s, t, labels, k)

        # physical check + repair again if needed
        if abs(m) > slope_max:
            m, intercepts = _physical_slope_repair_using_middle_y(
                s, t, labels, m, intercepts, slope_max, min_points_per_line=min_points_per_line
            )

        # -------------------------
        # Leave-one-out robustness (your outlier rejection request)
        # -------------------------
        best = _leave_one_out_refine(s, t, labels, k, min_points_per_line=min_points_per_line)
        if best is not None:
            _, m_lo, b_lo = best
            # accept only if physical
            if abs(m_lo) <= 1.01 * slope_max:
                m = float(np.clip(m_lo, -0.99*slope_max, 0.99*slope_max))
                intercepts = b_lo

    # final residuals
    t_hat = m * s + intercepts[labels]
    residuals_final = t - t_hat

    # return a "single-line style" model so your existing pipeline still works
    # (it will represent the shared slope and one representative intercept)
    # We'll use the median intercept across clusters.
    b_rep = float(np.median(intercepts))

    class SimpleModel:
        def __init__(self, a, b):
            self.coef_ = np.array([a])
            self.intercept_ = b
        def predict(self, X_in):
            X_in = np.asarray(X_in).reshape(-1, 1)
            return X_in[:,0] * self.coef_[0] + self.intercept_

    model_final = SimpleModel(m, b_rep)

    # package lines
    lines = []
    for c in range(k):
        idx = np.where(labels == c)[0].tolist()
        lines.append({"slope": float(m), "intercept": float(intercepts[c]), "points": idx})

    extra = {
        "n_lines": k,
        "axis": axis.tolist(),
        "s": s.tolist(),
        "shared_slope": float(m),
        "lines": lines,
        "labels": labels.tolist(),
        "intercepts": intercepts.tolist(),
        "slope_max": float(slope_max),
    }

    return float(m), residuals_final, model_final, extra
