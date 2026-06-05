import numpy as np
import pandas as pd


def generate_data(
    n: int,
    beta: float,
    mu_ZX: float,       # E[Z_i X_i]  — relevance (A2): must be nonzero
    sigma2_ZX: float,   # Var(Z_i X_i)
    sigma2_Ze: float,   # Var(Z_i eps_i)  — exogeneity (A1): E[Z eps]=0, but Var can be >0
    rho: float = 0.0,          # Corr(eps_Y, eps_X) — endogeneity; 0 = exogenous X
    eps_Y_df: float | None = None,  # dof for t-distributed eps_Y; None = Normal
    eps_X_df: float | None = None,  # dof for t-distributed eps_X; None = Normal
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
    # Draw two independent unit-variance base shocks (Normal or t), then mix via Cholesky:
    #   eps_Y = sqrt(sigma2_Ze)      * u1
    #   eps_X = sqrt(sigma2_noise_x) * (rho*u1 + sqrt(1-rho^2)*u2)
    # This gives Corr(eps_Y, eps_X) = rho exactly, with the target variances.
    def _draw(df, size):
        if df is None:
            return rng.standard_normal(size)
        if df <= 2:
            raise ValueError(f"dof must be > 2 for finite variance (got {df})")
        return rng.standard_t(df, size=size) / np.sqrt(df / (df - 2))

    u1 = _draw(eps_Y_df, n)
    u2 = _draw(eps_X_df, n)

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
