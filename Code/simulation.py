import numpy as np
import pandas as pd


def _standardised_shock(rng: np.random.Generator, dist, size: int) -> np.ndarray:
    """
    Draw `size` i.i.d. shocks with mean 0 and variance 1 from the family `dist`.

    dist : None                -> N(0, 1)
           ("t", df)           -> Student t with df > 2, scaled to unit variance
           ("pareto", alpha)   -> Pareto with tail index alpha > 2 (x_m = 1),
                                  centred and scaled to unit variance; strongly
                                  right-skewed, moments of order >= alpha infinite
           ("mixture", eps, s) -> Tukey (1960) contaminated normal: with prob
                                  1-eps draw N(0,1), with prob eps draw
                                  N(0, s^2), scaled to unit variance. All
                                  moments finite, but kurtosis grows with s
                                  (eps=0.1, s=3 gives kurtosis 8.33 vs 3
                                  for the Gaussian)

    All families have exactly unit variance, so the moment calibration in
    generate_data is exact regardless of the family.
    """
    if dist is None:
        return rng.standard_normal(size)
    family, *params = dist
    if family == "mixture":
        eps, scale = params
        if not 0.0 < eps < 1.0:
            raise ValueError(f"mixture contamination eps must be in (0, 1), got {eps}")
        x = rng.standard_normal(size)
        x[rng.random(size) < eps] *= scale
        var = (1.0 - eps) + eps * scale**2
        return x / np.sqrt(var)
    (param,) = params
    if family == "t":
        if param <= 2:
            raise ValueError(f"t dof must be > 2 for finite variance (got {param})")
        return rng.standard_t(param, size=size) / np.sqrt(param / (param - 2))
    if family == "pareto":
        alpha = param
        if alpha <= 2:
            raise ValueError(f"Pareto tail index must be > 2 for finite variance (got {alpha})")
        x = rng.pareto(alpha, size=size) + 1.0          # Pareto(alpha) with x_m = 1
        mean = alpha / (alpha - 1.0)
        var = alpha / ((alpha - 1.0) ** 2 * (alpha - 2.0))
        return (x - mean) / np.sqrt(var)
    raise ValueError(f"unknown shock family {family!r}")


def generate_data(
    n: int,
    beta: float,
    mu_ZX: float,       # E[Z_i X_i]  — relevance (A2): must be nonzero
    sigma2_ZX: float,   # Var(Z_i X_i)
    sigma2_Ze: float,   # Var(Z_i eps_i)  — exogeneity (A1): E[Z eps]=0, but Var can be >0
    rho: float = 0.0,          # Corr(eps_Y, eps_X) — endogeneity; 0 = exogenous X
    eps_Y_df: float | None = None,  # dof for t-distributed eps_Y; None = Normal
    eps_X_df: float | None = None,  # dof for t-distributed eps_X; None = Normal
    eps_Y_dist=None,   # shock family spec, e.g. ("t", 2.1), ("pareto", 2.5) or ("mixture", 0.1, 3.0); overrides eps_Y_df
    eps_X_dist=None,   # shock family spec for eps_X; overrides eps_X_df
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """
    Generate an i.i.d. sample from the IV structural equation

        Y_i = beta * X_i + eps_i

    where (Y_i, X_i, Z_i) satisfy (A1)–(A3) from the paper.

    The DGP is constructed so that the cross-moment and variance parameters
    match the requested values exactly in population:

        E[Z X]    = mu_ZX
        Var(Z X)  = sigma2_ZX
        Var(Z eps)= sigma2_Ze

    Strategy
    --------
    Draw Z ~ N(0, 1) independently, then build X and eps as linear functions
    of Z plus plus noise that is independent of Z but may be correlated across the two equations, calibrated to hit the target moments.

    Parameters
    ----------
    n          : sample size
    beta       : structural parameter
    mu_ZX      : E[Z X], controls instrument relevance (must be != 0)
    sigma2_ZX  : Var(Z X)
    sigma2_Ze  : Var(Z eps_Y)
    rho        : Corr(eps_Y, eps_X); 0 = exogenous X, nonzero = endogenous X
    eps_Y_df   : dof for t-distributed eps_Y (must be > 2); None = Normal
    eps_X_df   : dof for t-distributed eps_X (must be > 2); None = Normal
    eps_Y_dist : shock family spec ("t", df) / ("pareto", alpha) /
                 ("mixture", eps, scale); overrides eps_Y_df
    eps_X_dist : shock family spec for eps_X; overrides eps_X_df
    rng        : numpy Generator; created from a fresh seed if None

    Returns
    -------
    pd.DataFrame with columns ['Y', 'X', 'Z', 'eps_Y', 'eps_X'] and n rows
        eps_Y : structural error in Y = beta*X + eps_Y
        eps_X : structural error in X = mu_ZX*Z + eps_X
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- instrument ---
    # Z ~ N(0,1); all cross-moments are computed w.r.t. Var(Z)=1
    Z = rng.standard_normal(n)

    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")

    # --- eps_Y and eps_X ---
    # Draw two independent unit-variance base shocks, then mix via Cholesky:
    #   eps_Y = sqrt(sigma2_Ze)      * u1
    #   eps_X = sqrt(sigma2_noise_x) * (rho*u1 + sqrt(1-rho^2)*u2)
    # This gives Corr(eps_Y, eps_X) = rho exactly, with the target variances.
    # Family spec: eps_*_dist takes precedence; eps_*_df kept for backward compat.
    if eps_Y_dist is None and eps_Y_df is not None:
        eps_Y_dist = ("t", eps_Y_df)
    if eps_X_dist is None and eps_X_df is not None:
        eps_X_dist = ("t", eps_X_df)

    u1 = _standardised_shock(rng, eps_Y_dist, n)
    u2 = _standardised_shock(rng, eps_X_dist, n)

    eps_Y = u1 * np.sqrt(sigma2_Ze)

    # --- X ---
    # E[Z X]   = a_x = mu_ZX
    # Var(Z X) = a_x^2 * Var(Z^2) + sigma2_nx  =>  sigma2_nx = sigma2_ZX - 2*a_x^2
    a_x = mu_ZX
    sigma2_noise_x = sigma2_ZX - 2.0 * a_x**2
    if sigma2_noise_x < 0:
        raise ValueError(
            f"sigma2_ZX={sigma2_ZX} is too small for mu_ZX={mu_ZX}: "
            f"need sigma2_ZX >= 2*mu_ZX^2 = {2*a_x**2:.4g}"
        )
    eps_X = (rho * u1 + np.sqrt(1 - rho**2) * u2) * np.sqrt(sigma2_noise_x)
    X = a_x * Z + eps_X

    # --- Y from structural equation ---
    Y = beta * X + eps_Y

    return pd.DataFrame({"Y": Y, "X": X, "Z": Z, "eps_Y": eps_Y, "eps_X": eps_X})


def iv_estimate(data: pd.DataFrame) -> dict[str, float]:
    """
    Standard just-identified IV estimator for the model Y = beta * X + eps,
    using the single instrument Z.

    The estimator is the sample analog of beta = mu_ZY / mu_ZX:

        beta_hat = mean(Z * Y) / mean(Z * X)

    A heteroskedasticity-robust standard error is also returned, based on the
    asymptotic variance of the method-of-moments estimator:

        sqrt(n) (beta_hat - beta) -> N(0, mu_ZX^{-2} * Var(Z * eps))

    estimated by plugging in residuals eps_hat = Y - beta_hat * X.

    Parameters
    ----------
    data : DataFrame with at least columns 'Y', 'X', 'Z' (e.g. from generate_data)

    Returns
    -------
    dict with keys 'beta_hat', 'se', 'n'
    """
    Y = data["Y"].to_numpy()
    X = data["X"].to_numpy()
    Z = data["Z"].to_numpy()
    n = len(Y)

    mZX = np.mean(Z * X)
    if mZX == 0:
        raise ValueError("mean(Z * X) is zero; instrument is not relevant (A2 violated)")

    beta_hat = np.mean(Z * Y) / mZX

    # Robust SE: avar = Var(Z * eps_hat) / mean(Z*X)^2, then se = sqrt(avar / n)
    eps_hat = Y - beta_hat * X
    avar = np.mean((Z * eps_hat) ** 2) / mZX**2
    se = np.sqrt(avar / n)

    return {"beta_hat": float(beta_hat), "se": float(se), "n": n}


def iv_estimate_rm(
    data: pd.DataFrame,
    delta: float = 0.05,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
) -> dict[str, float]:
    """
    Ratio-of-Medians (RM) IV estimator (Algorithm 2).

    The number of blocks is set from the confidence parameter delta as

        k = ceil(8 * ln(2 / delta)),

    and each block has size m = floor(n / k). For each block B_j compute the
    block means

        S_ZY^(j) = (1/m) sum_{i in B_j} Z_i Y_i
        S_ZX^(j) = (1/m) sum_{i in B_j} Z_i X_i

    then take the coordinate-wise medians across blocks and return their ratio:

        beta_RM = median_j(S_ZY^(j)) / median_j(S_ZX^(j))

    The median of block means is robust to heavy-tailed Z*Y and Z*X, so this
    estimator can outperform the plain mean-based IV estimator when the
    error terms have heavy tails (e.g. small t degrees of freedom).

    Parameters
    ----------
    data    : DataFrame with columns 'Y', 'X', 'Z' (e.g. from generate_data)
    delta   : confidence parameter; number of blocks k = ceil(8 ln(2/delta))
    rng     : numpy Generator used for shuffling; fresh seed if None
    shuffle : if True, randomly permute rows before blocking so block
              assignment does not depend on the (arbitrary) row order

    Returns
    -------
    dict with keys 'beta_hat', 'delta', 'k', 'm', 'n'
    """
    Y = data["Y"].to_numpy()
    X = data["X"].to_numpy()
    Z = data["Z"].to_numpy()
    n = len(Y)

    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    k = int(np.ceil(8 * np.log(2 / delta)))
    if k > n:
        raise ValueError(
            f"k = ceil(8 ln(2/delta)) = {k} exceeds n={n}; "
            "increase n or delta"
        )

    m = n // k  # block size; the last n - k*m rows are dropped
    if m == 0:
        raise ValueError(f"k={k} too large: block size floor(n/k) is 0")

    ZY = Z * Y
    ZX = Z * X

    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        perm = rng.permutation(n)
        ZY = ZY[perm]
        ZX = ZX[perm]

    # Reshape the first k*m elements into k blocks of size m, then mean over each block.
    block_means_ZY = ZY[: k * m].reshape(k, m).mean(axis=1)
    block_means_ZX = ZX[: k * m].reshape(k, m).mean(axis=1)

    S_ZY = np.median(block_means_ZY)
    S_ZX = np.median(block_means_ZX)
    if S_ZX == 0:
        raise ValueError("median of block means(Z * X) is zero; instrument not relevant")

    beta_hat = S_ZY / S_ZX

    return {"beta_hat": float(beta_hat), "delta": delta, "k": k, "m": m, "n": n}


def iv_estimate_mr(
    data: pd.DataFrame,
    delta: float = 0.05,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
) -> dict[str, float]:
    """
    Median-of-Ratios (MoR) IV estimator (Algorithm 3).

    Per Theorem 4.2, the number of blocks is set from the confidence
    parameter delta as

        k = ceil(8 * ln(1 / delta)),

    and each block has size m = floor(n / k). For each block B_j compute the
    block means

        S_ZY^(j) = (1/m) sum_{i in B_j} Z_i Y_i
        S_ZX^(j) = (1/m) sum_{i in B_j} Z_i X_i

    form a block-level IV estimate, then take the median across blocks:

        beta^(j) = S_ZY^(j) / S_ZX^(j)
        beta_MR  = median_j(beta^(j))

    Differs from the Ratio-of-Medians estimator (iv_estimate_rm) in that the
    ratio is taken inside each block *before* the median, rather than after.

    Parameters
    ----------
    data    : DataFrame with columns 'Y', 'X', 'Z' (e.g. from generate_data)
    delta   : confidence parameter; number of blocks k = ceil(8 ln(1/delta))
    rng     : numpy Generator used for shuffling; fresh seed if None
    shuffle : if True, randomly permute rows before blocking so block
              assignment does not depend on the (arbitrary) row order

    Returns
    -------
    dict with keys 'beta_hat', 'delta', 'k', 'm', 'n'
    """
    Y = data["Y"].to_numpy()
    X = data["X"].to_numpy()
    Z = data["Z"].to_numpy()
    n = len(Y)

    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    k = int(np.ceil(8 * np.log(1 / delta)))
    if k > n:
        raise ValueError(
            f"k = ceil(8 ln(1/delta)) = {k} exceeds n={n}; "
            "increase n or delta"
        )

    m = n // k  # block size; the last n - k*m rows are dropped
    if m == 0:
        raise ValueError(f"k={k} too large: block size floor(n/k) is 0")

    ZY = Z * Y
    ZX = Z * X

    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        perm = rng.permutation(n)
        ZY = ZY[perm]
        ZX = ZX[perm]

    # Reshape the first k*m elements into k blocks of size m, then mean over each block.
    block_means_ZY = ZY[: k * m].reshape(k, m).mean(axis=1)
    block_means_ZX = ZX[: k * m].reshape(k, m).mean(axis=1)

    # Block-level IV estimates, then median.
    if np.any(block_means_ZX == 0):
        raise ValueError("a block mean(Z * X) is zero; instrument not relevant")
    block_betas = block_means_ZY / block_means_ZX
    beta_hat = np.median(block_betas)

    return {"beta_hat": float(beta_hat), "delta": delta, "k": k, "m": m, "n": n}


def catoni_mean(
    x: np.ndarray,
    delta: float = 0.05,
    v: float | None = None,
    max_iter: int = 100,
) -> float:
    """ 
    Catoni's M-estimator of the mean (Catoni, 2012).

    Solves sum_i psi(alpha * (x_i - theta)) = 0 for theta, where psi is
    Catoni's narrowest influence function

        psi(u) =  log(1 + u + u^2/2)   for u >= 0,
               = -log(1 - u + u^2/2)   for u <  0.

    The tuning parameter is chosen from the target confidence level delta and
    a variance (bound) v, following Catoni's optimal choice:

        alpha = sqrt( 2 ln(2/delta) / ( n * v * (1 + 2 ln(2/delta) / (n - 2 ln(2/delta))) ) ),

    which yields P(|theta_hat - mu| > t_delta) <= delta with a sub-Gaussian
    deviation t_delta, provided Var(x) <= v and n > 2 ln(2/delta).

    If v is None, it is estimated robustly: a scalar MoM estimate of the mean
    (k = ceil(8 ln(2/delta)) blocks), then a MoM estimate of the mean of the
    squared deviations. Using MoM rather than the sample variance keeps the
    tuning itself heavy-tail robust.

    The left-hand side is strictly decreasing in theta, so the root is found
    by bisection on [min(x), max(x)].
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    ln_term = np.log(2.0 / delta)
    if n <= 2 * ln_term:
        raise ValueError(f"Catoni requires n > 2 ln(2/delta) = {2 * ln_term:.1f}, got n={n}")

    if v is None:
        # Robust preliminary variance: MoM of the squared deviations from a MoM mean.
        k = int(np.ceil(8 * np.log(2.0 / delta)))
        m = n // k
        if m == 0:
            raise ValueError(f"n={n} too small for MoM variance pre-estimate with k={k}")
        mu0 = np.median(x[: k * m].reshape(k, m).mean(axis=1))
        v = float(np.median(((x - mu0) ** 2)[: k * m].reshape(k, m).mean(axis=1)))
        v = max(v, 1e-12)

    alpha = np.sqrt(2.0 * ln_term / (n * v * (1.0 + 2.0 * ln_term / (n - 2.0 * ln_term))))

    def psi_sum(theta: float) -> float:
        u = alpha * (x - theta)
        return float(np.sum(np.sign(u) * np.log1p(np.abs(u) + 0.5 * u * u)))

    lo, hi = float(np.min(x)) - 1.0, float(np.max(x)) + 1.0
    # psi_sum is decreasing: positive at lo, negative at hi.
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if psi_sum(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * (1.0 + abs(mid)):
            break
    return 0.5 * (lo + hi)


def iv_estimate_catoni(data: pd.DataFrame, delta: float = 0.05) -> dict[str, float]:
    """
    Ratio-of-Catoni-means IV estimator: the natural robust competitor to the
    Ratio-of-Medians estimator, replacing each coordinate-wise MoM median by
    Catoni's M-estimator of the mean:

        beta_Cat = Catoni(Z*Y; delta/2) / Catoni(Z*X; delta/2)

    The confidence budget delta is split evenly across the two coordinates
    (mirroring RoM's k = ceil(8 ln(2/delta))), so each Catoni estimate is
    tuned for confidence level delta/2. Each estimate uses its own robust
    (MoM-based) scale pre-estimate; see catoni_mean.

    Returns
    -------
    dict with keys 'beta_hat', 'delta', 'n'
    """
    Y = data["Y"].to_numpy()
    X = data["X"].to_numpy()
    Z = data["Z"].to_numpy()
    n = len(Y)

    mu_ZY = catoni_mean(Z * Y, delta=delta / 2.0)
    mu_ZX = catoni_mean(Z * X, delta=delta / 2.0)
    if mu_ZX == 0:
        raise ValueError("Catoni estimate of E[Z X] is zero; instrument not relevant")

    return {"beta_hat": float(mu_ZY / mu_ZX), "delta": delta, "n": n}


def trimmed_mean(
    x: np.ndarray,
    delta: float = 0.05,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
) -> float:
    """
    Trimmed-mean estimator of the mean (Oliveira and Orenstein, 2019;
    Lugosi and Mendelson, 2021).

    The sample is split into two halves. The first half supplies the empirical
    quantiles at levels eps and 1 - eps,

        eps = 8 ln(4/delta) / (3n),

    and the second half is truncated to the resulting interval [a, b] before
    being averaged:

        mu_hat       = (1/m) sum_{i in second half} phi_{a,b}(x_i),
        phi_{a,b}(u) = min(max(u, a), b).

    Splitting the sample is what makes the guarantee work: the truncation
    levels are independent of the points they are applied to, which yields a
    sub-Gaussian deviation bound |mu_hat - mu| <= C sigma sqrt(ln(4/delta)/n)
    under nothing more than a finite variance.

    Note the contrast with catoni_mean: no variance proxy enters anywhere. The
    truncation levels are empirical quantiles of the data itself, so the
    estimator is scale-equivariant by construction and needs no preliminary
    scale estimate to be feasible.

    Parameters
    ----------
    x       : sample
    delta   : confidence parameter; sets the trimming level eps
    rng     : numpy Generator used for shuffling; fresh seed if None
    shuffle : if True, randomly permute before splitting so the two halves do
              not depend on the (arbitrary) row order

    Returns
    -------
    float, the trimmed mean
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    eps = 8.0 * np.log(4.0 / delta) / (3.0 * n)
    if eps >= 0.5:
        raise ValueError(
            f"n={n} too small for delta={delta}: trimming level "
            f"eps = 8 ln(4/delta)/(3n) = {eps:.3g} must be < 1/2"
        )

    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        x = x[rng.permutation(n)]

    # First half sets the truncation window, second half is averaged.
    h = n // 2
    a, b = np.quantile(x[:h], [eps, 1.0 - eps])
    return float(np.mean(np.clip(x[h:], a, b)))


def iv_estimate_trimmed(
    data: pd.DataFrame,
    delta: float = 0.05,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
) -> dict[str, float]:
    """
    Ratio-of-trimmed-means IV estimator: the same ratio construction as
    iv_estimate_catoni, with each coordinate mean replaced by the trimmed mean
    of Oliveira and Orenstein (2019) and Lugosi and Mendelson (2021):

        beta_Trim = TrimmedMean(Z*Y; delta/2) / TrimmedMean(Z*X; delta/2)

    The confidence budget delta is split evenly across the two coordinates,
    mirroring the allocation used by iv_estimate_catoni and RoM. A single
    permutation is shared by both coordinates, so the numerator and the
    denominator are trimmed on the same split of the sample.

    Returns
    -------
    dict with keys 'beta_hat', 'delta', 'n'
    """
    Y = data["Y"].to_numpy()
    X = data["X"].to_numpy()
    Z = data["Z"].to_numpy()
    n = len(Y)

    ZY = Z * Y
    ZX = Z * X
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        perm = rng.permutation(n)
        ZY = ZY[perm]
        ZX = ZX[perm]

    mu_ZY = trimmed_mean(ZY, delta=delta / 2.0, shuffle=False)
    mu_ZX = trimmed_mean(ZX, delta=delta / 2.0, shuffle=False)
    if mu_ZX == 0:
        raise ValueError("trimmed estimate of E[Z X] is zero; instrument not relevant")

    return {"beta_hat": float(mu_ZY / mu_ZX), "delta": delta, "n": n}


if __name__ == "__main__":
    df = generate_data(
        n=10_000,
        beta=1.5,
        mu_ZX=0.8,
        sigma2_ZX=1.5,
        sigma2_Ze=0.625,
        rho=0.7,
        eps_Y_df=5,
        eps_X_df=5,
        rng=np.random.default_rng(seed=42),
    )

    delta = 0.05
    print(df.head(5))

    result = iv_estimate(df)
    print(f"\nMean IV estimate:      beta_hat = {result['beta_hat']:.4f} "
          f"(se = {result['se']:.4f}, true beta = 1.5)")

    result_rm = iv_estimate_rm(df, delta=delta, rng=np.random.default_rng(seed=0))
    print(f"Ratio-of-Medians (delta={result_rm['delta']}, k={result_rm['k']}): "
          f"beta_hat = {result_rm['beta_hat']:.4f} (true beta = 1.5)")

    result_mr = iv_estimate_mr(df, delta=delta, rng=np.random.default_rng(seed=0))
    print(f"Median-of-Ratios (delta={result_mr['delta']}, k={result_mr['k']}): "
          f"beta_hat = {result_mr['beta_hat']:.4f} (true beta = 1.5)")
