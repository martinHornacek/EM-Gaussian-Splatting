"""
em_utils.py - Shared utilities for EM fitting and rendering

Contains: load_config, download_kodak_dataset, render_gaussians,
          and fit functions for standard and minibatch EM.
"""

import os
import time
import yaml
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

def load_config(config_path='config.yml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_kodak_dataset(output_dir="./kodak"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    existing = list(Path(output_dir).glob("kodim*.png"))
    if len(existing) >= 24:
        print(f"✓ Kodak dataset present ({len(existing)} images)")
        return output_dir
    print("Downloading Kodak dataset...")
    import urllib.request
    base_url = "http://r0k.us/graphics/kodak/kodak/"
    for i in range(1, 25):
        filename = f"kodim{i:02d}.png"
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            try:
                print(f"  Downloading {filename}...", end=' ')
                urllib.request.urlretrieve(base_url + filename, filepath)
                print("✓")
            except Exception as e:
                print(f"✗ {e}")
    return output_dir


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _prepare_data(img_np):
    """
    Build the 5D joint pixel array [x_norm, y_norm, r, g, b] used by all variants.
    Returns (data, h, w).
    """
    h, w, _ = img_np.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
    pixels = img_np.reshape(-1, 3)
    coords_norm = coords / np.array([[w, h]], dtype=np.float32)
    data = np.concatenate([coords_norm, pixels], axis=1).astype(np.float32)
    return data, h, w


def _resolve_n_comp(config, n_components):
    n_comp = n_components if n_components is not None else (config or {}).get('n_components')
    if n_comp is None:
        raise ValueError("n_components must be provided either as argument or in config")
    return n_comp


def render_gaussians(means, covs, colors, weights, image_size):
    """Alpha-composite Gaussian splats onto a canvas."""
    h, w = image_size
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    pos = np.dstack((X, Y))
    for mean, cov, color, wgt in zip(means, covs, colors, weights):
        try:
            rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
            pdf = rv.pdf(pos)
            pdf = pdf / (np.max(pdf) + 1e-8) * wgt
            pdf = np.expand_dims(pdf, axis=-1)
            canvas += pdf * color
            weight_sum += pdf
        except (np.linalg.LinAlgError, ValueError):
            continue
    render = np.divide(canvas, weight_sum, out=np.zeros_like(canvas), where=weight_sum > 1e-8)
    return np.clip(render, 0, 1)


def _render_from_sklearn_gmm(gmm, covariance_type, n_comp, h, w, start_time, fit_time, verbose):
    """Extract parameters from a fitted sklearn GMM and render."""
    means_pos = gmm.means_[:, :2] * np.array([[w, h]])
    colors = gmm.means_[:, 2:]
    weights = gmm.weights_

    if covariance_type == 'full':
        covs_pos = gmm.covariances_[:, :2, :2] * np.array([[[w, h], [w, h]]]) ** 2
    elif covariance_type == 'tied':
        cov = gmm.covariances_
        covs_pos = np.tile(cov[:2, :2] * np.array([[w, h], [w, h]]) ** 2, (n_comp, 1, 1))
    elif covariance_type == 'diag':
        diag_vars = gmm.covariances_
        covs_pos = np.zeros((n_comp, 2, 2))
        covs_pos[:, 0, 0] = diag_vars[:, 0] * w ** 2
        covs_pos[:, 1, 1] = diag_vars[:, 1] * h ** 2
    elif covariance_type == 'spherical':
        variances = gmm.covariances_
        covs_pos = np.zeros((n_comp, 2, 2))
        covs_pos[:, 0, 0] = variances * w ** 2
        covs_pos[:, 1, 1] = variances * h ** 2
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    if verbose:
        print(f"  Rendering {n_comp} splats...")
    render_start = time.time()
    render_np = render_gaussians(means_pos, covs_pos, colors, weights, (h, w))
    render_time = time.time() - render_start

    return render_np, {
        'fit_time': fit_time,
        'render_time': render_time,
        'total_time': time.time() - start_time,
        'converged': gmm.converged_,
        'n_iter': gmm.n_iter_,
    }


# ---------------------------------------------------------------------------
# Variant 1 — Standard EM  (original, closed-form sklearn GMM)
# ---------------------------------------------------------------------------

def fit_gmm_and_render(img_np, config=None, n_components=None, verbose=False):
    """
    Standard EM via sklearn GaussianMixture (closed-form E and M steps).
    This is the original baseline.
    """
    data, h, w = _prepare_data(img_np)
    n_comp = _resolve_n_comp(config, n_components)
    cfg = config or {}
    covariance_type = cfg.get('covariance_type', 'full')
    max_iter = cfg.get('max_iter', 100)
    init_params = cfg.get('init_params', 'kmeans')

    if verbose:
        print(f"  [Standard EM] Fitting {n_comp} components...")
    start_time = time.time()

    gmm = GaussianMixture(
        n_components=n_comp,
        covariance_type=covariance_type,
        max_iter=max_iter,
        init_params=init_params,
        random_state=42,
        verbose=0,
    )
    gmm.fit(data)
    fit_time = time.time() - start_time

    if verbose:
        print(f"  ✓ fitted in {fit_time:.2f}s "
              f"(converged={gmm.converged_}, iters={gmm.n_iter_})")

    return _render_from_sklearn_gmm(gmm, covariance_type, n_comp, h, w,
                                    start_time, fit_time, verbose)


# ---------------------------------------------------------------------------
# Variant 2 — Hard EM  (k-means limit: snap responsibilities to 0/1)
# ---------------------------------------------------------------------------

def fit_hard_em_and_render(img_np, config=None, n_components=None, verbose=False):
    """
    Hard EM: the E-step assigns each pixel to exactly one component (argmax),
    turning soft responsibilities into binary indicator variables.

    Why it matters for splatting: many rasterisation-based renderers implicitly
    do hard assignment (a pixel is 'owned' by its nearest/highest-alpha splat).
    Hard EM formalises that and lets you compare it against the soft baseline.

    Implementation:
      - Init: KMeans centroids
      - E-step: argmin squared Mahalanobis distance (collapses to Euclidean
                until covariances are estimated)
      - M-step: per-cluster mean and full covariance from hard counts
      - Iterate until labels stabilise or max_iter reached
    """
    data, h, w = _prepare_data(img_np)
    n_comp = _resolve_n_comp(config, n_components)
    cfg = config or {}
    max_iter = cfg.get('max_iter', 100)

    if verbose:
        print(f"  [Hard EM] Fitting {n_comp} components...")
    start_time = time.time()

    # ---- Initialise with KMeans ----
    km = KMeans(n_clusters=n_comp, random_state=42, n_init=3, max_iter=500)
    labels = km.fit_predict(data)
    means = km.cluster_centers_.copy()              # (K, D)
    D = data.shape[1]
    reg = 1e-4 * np.eye(D)

    def compute_covs_and_weights(data, labels, means, K):
        covs = np.zeros((K, D, D))
        weights = np.zeros(K)
        for k in range(K):
            mask = labels == k
            nk = mask.sum()
            weights[k] = nk / len(data)
            if nk > D:
                diff = data[mask] - means[k]
                covs[k] = (diff.T @ diff) / nk + reg
            else:
                covs[k] = reg
        return covs, weights

    n_iter = 0
    for iteration in range(max_iter):
        # ---- M-step ----
        covs, weights = compute_covs_and_weights(data, labels, means, n_comp)
        means = np.array([
            data[labels == k].mean(axis=0) if (labels == k).any() else means[k]
            for k in range(n_comp)
        ])

        # ---- E-step (hard): squared Mahalanobis distance to each component ----
        dists = np.zeros((len(data), n_comp))
        for k in range(n_comp):
            try:
                cov_inv = np.linalg.inv(covs[k])
            except np.linalg.LinAlgError:
                cov_inv = np.eye(D)
            diff = data - means[k]                  # (N, D)
            dists[:, k] = np.einsum('ni,ij,nj->n', diff, cov_inv, diff)

        new_labels = np.argmin(dists, axis=1)
        n_iter = iteration + 1

        if np.all(new_labels == labels):
            if verbose:
                print(f"  Hard EM converged at iteration {n_iter}")
            labels = new_labels
            break
        labels = new_labels

    fit_time = time.time() - start_time

    # Final covariances and weights
    covs, weights = compute_covs_and_weights(data, labels, means, n_comp)

    if verbose:
        print(f"  ✓ fitted in {fit_time:.2f}s (iters={n_iter})")

    # ---- Render ----
    means_pos = means[:, :2] * np.array([[w, h]])
    colors = np.clip(means[:, 2:], 0, 1)
    covs_pos = covs[:, :2, :2] * np.array([[[w, h], [w, h]]]) ** 2

    render_start = time.time()
    render_np = render_gaussians(means_pos, covs_pos, colors, weights, (h, w))
    render_time = time.time() - render_start

    return render_np, {
        'fit_time': fit_time,
        'render_time': render_time,
        'total_time': time.time() - start_time,
        'converged': n_iter < max_iter,
        'n_iter': n_iter,
    }


# ---------------------------------------------------------------------------
# Variant 3 — Mini-batch EM  (subsample pixels, run full EM on subset)
# ---------------------------------------------------------------------------

def fit_minibatch_em_and_render(img_np, config=None, n_components=None, verbose=False):
    """
    Mini-batch EM: fit the GMM on a random subsample of pixels.

    Why it matters: a 768×512 Kodak image has ~400k pixels, making full EM
    slow. Subsampling lets you study the speed/quality trade-off. The learned
    Gaussians still describe the whole image — only the fitting data is reduced.

    Config keys (under 'minibatch'):
        subsample_ratio : fraction of pixels to use (default 0.15)
        min_samples     : floor on subsample size (default 5000)
    """
    data, h, w = _prepare_data(img_np)
    n_comp = _resolve_n_comp(config, n_components)
    cfg = config or {}
    mb_cfg = cfg.get('minibatch', {})
    subsample_ratio = mb_cfg.get('subsample_ratio', 0.15)
    min_samples = mb_cfg.get('min_samples', 5000)
    covariance_type = cfg.get('covariance_type', 'full')
    max_iter = cfg.get('max_iter', 100)

    n_sub = max(int(len(data) * subsample_ratio), min_samples, n_comp * 10)
    n_sub = min(n_sub, len(data))

    if verbose:
        print(f"  [Mini-batch EM] Subsampling {n_sub}/{len(data)} pixels "
              f"({100*n_sub/len(data):.1f}%)...")
    start_time = time.time()

    rng = np.random.default_rng(42)
    idx = rng.choice(len(data), n_sub, replace=False)
    data_sub = data[idx]

    gmm = GaussianMixture(
        n_components=n_comp,
        covariance_type=covariance_type,
        max_iter=max_iter,
        init_params='kmeans',
        random_state=42,
        verbose=0,
    )
    gmm.fit(data_sub)
    fit_time = time.time() - start_time

    if verbose:
        print(f"  ✓ fitted in {fit_time:.2f}s "
              f"(converged={gmm.converged_}, iters={gmm.n_iter_})")

    return _render_from_sklearn_gmm(gmm, covariance_type, n_comp, h, w,
                                    start_time, fit_time, verbose)


# ---------------------------------------------------------------------------
# Variant 4 — Gradient EM  (M-step via Adam; closest to 3DGS optimisation)
# ---------------------------------------------------------------------------

def fit_gradient_em_and_render(img_np, config=None, n_components=None, verbose=False):
    """
    Gradient EM: replace the closed-form M-step with gradient ascent on the
    expected complete-data log-likelihood (equivalently, gradient descent on
    NLL), using the current responsibilities as fixed weights.

    This is the GEM (Generalised EM) variant — any update that increases the
    objective counts as a valid M-step.  In practice the E-step is also
    softened: we compute responsibilities from the current parameters and
    embed them implicitly into the NLL through the log-sum-exp.

    This is structurally identical to what 3DGS does: it optimises means,
    covariances, colours, and opacities of Gaussians with Adam, rather than
    updating them analytically.  The main difference is the rendering model
    (rasterisation vs scipy PDF evaluation here).

    Parameters (under config['gradient']):
        lr          : Adam learning rate (default 0.01)
        n_epochs    : max gradient steps (default 300)
        batch_size  : pixels per mini-batch (default 20000)

    Implementation notes:
        - Covariances are parameterised via Cholesky: Sigma = L L^T
          (lower-triangular L with exp-parameterised positive diagonal).
          This guarantees positive-definiteness throughout training.
        - Mixing weights use a softmax over unconstrained logits.
        - NLL is estimated on a random mini-batch each step, making this
          stochastic gradient EM.
    """
    try:
        import torch
    except ImportError:
        raise ImportError("Gradient EM requires PyTorch. "
                          "Install with: pip install torch")

    data_np, h, w = _prepare_data(img_np)
    n_comp = _resolve_n_comp(config, n_components)
    cfg = config or {}
    grad_cfg = cfg.get('gradient', {})
    lr = grad_cfg.get('lr', 0.01)
    n_epochs = grad_cfg.get('n_epochs', 300)
    batch_size = grad_cfg.get('batch_size', 20_000)
    tol = grad_cfg.get('tol', 1e-5)

    D = 5   # [x, y, r, g, b]
    N = len(data_np)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f"  [Gradient EM] K={n_comp}, epochs={n_epochs}, "
              f"batch={batch_size}, device={device}")
    start_time = time.time()

    data = torch.from_numpy(data_np).to(device)     # (N, D)

    # ---- Initialise from KMeans (same as sklearn default) ----
    km = KMeans(n_clusters=n_comp, random_state=42, n_init=3, max_iter=500)
    km.fit(data_np[::max(1, N // 30_000)])           # subsample KMeans init
    mu_init = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)

    # Parameters
    mu = torch.nn.Parameter(mu_init.clone())         # (K, D)

    # Cholesky factor: store log-diagonal and off-diagonal elements separately
    # Sigma_k = L_k @ L_k^T,  L_k lower-triangular, diag(L_k) > 0
    init_log_scale = np.log(0.05)
    L_log_diag = torch.nn.Parameter(
        torch.full((n_comp, D), init_log_scale, device=device))
    n_offdiag = D * (D - 1) // 2
    L_offdiag = torch.nn.Parameter(
        torch.zeros(n_comp, n_offdiag, device=device))
    log_pi = torch.nn.Parameter(torch.zeros(n_comp, device=device))

    optimizer = torch.optim.Adam([mu, L_log_diag, L_offdiag, log_pi], lr=lr)

    # Indices for filling the lower-triangular off-diagonal entries
    rows, cols = torch.tril_indices(D, D, offset=-1)

    def build_L():
        """Assemble lower-triangular Cholesky factor (K, D, D)."""
        L = torch.zeros(n_comp, D, D, device=device)
        L[:, torch.arange(D), torch.arange(D)] = torch.exp(L_log_diag)
        L[:, rows, cols] = L_offdiag
        return L

    def nll_batch(x):
        """
        Negative log-likelihood on a mini-batch x of shape (B, D).

        log p(x_i) = log sum_k pi_k * N(x_i | mu_k, Sigma_k)

        Computed stably via log-sum-exp.  Mahalanobis distance uses the
        Cholesky solve:  ||L^{-1}(x - mu)||^2  =  (x-mu)^T Sigma^{-1} (x-mu)
        """
        B = x.shape[0]
        L = build_L()                                # (K, D, D)
        log_weights = log_pi - torch.logsumexp(log_pi, dim=0)   # (K,)

        # diff: (K, D, B)  ← broadcast over components
        diff = (x.unsqueeze(0) - mu.unsqueeze(1)).permute(0, 2, 1)

        # Solve L z = diff for z:  z = L^{-1} diff,  shape (K, D, B)
        z = torch.linalg.solve_triangular(L, diff, upper=False)

        mahal = (z ** 2).sum(dim=1)                  # (K, B)

        # log |Sigma_k| = 2 * sum(log diag L_k)
        log_det = 2.0 * L_log_diag.sum(dim=1)        # (K,)

        log_comp = (
            -0.5 * D * np.log(2 * np.pi)
            - 0.5 * log_det.unsqueeze(1)
            - 0.5 * mahal
        )                                             # (K, B)

        log_joint = log_comp + log_weights.unsqueeze(1)   # (K, B)
        return -torch.logsumexp(log_joint, dim=0).mean()   # scalar NLL

    # ---- Gradient descent loop ----
    prev_loss = float('inf')
    n_iter = 0
    for epoch in range(n_epochs):
        idx = torch.randperm(N, device=device)[:batch_size]
        loss = nll_batch(data[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_iter = epoch + 1

        loss_val = loss.item()
        if verbose and epoch % 50 == 0:
            print(f"    epoch {epoch:4d}/{n_epochs}  NLL={loss_val:.5f}")
        if abs(prev_loss - loss_val) < tol and epoch > 30:
            if verbose:
                print(f"  Gradient EM converged at epoch {epoch+1}")
            break
        prev_loss = loss_val

    fit_time = time.time() - start_time

    # ---- Extract final numpy parameters ----
    with torch.no_grad():
        L = build_L()
        Sigma = L @ L.permute(0, 2, 1)              # (K, D, D)
        pi = torch.softmax(log_pi, dim=0)
        mu_np = mu.cpu().numpy()
        Sigma_np = Sigma.cpu().numpy()
        pi_np = pi.cpu().numpy()

    if verbose:
        print(f"  ✓ fitted in {fit_time:.2f}s (iters={n_iter})")

    means_pos = mu_np[:, :2] * np.array([[w, h]])
    colors = np.clip(mu_np[:, 2:], 0, 1)
    covs_pos = Sigma_np[:, :2, :2] * np.array([[[w, h], [w, h]]]) ** 2

    render_start = time.time()
    render_np = render_gaussians(means_pos, covs_pos, colors, pi_np, (h, w))
    render_time = time.time() - render_start

    return render_np, {
        'fit_time': fit_time,
        'render_time': render_time,
        'total_time': time.time() - start_time,
        'converged': n_iter < n_epochs,
        'n_iter': n_iter,
    }


# ---------------------------------------------------------------------------
# Variant 5 — Decoupled EM  (spatial GMM + independent color estimation)
# ---------------------------------------------------------------------------

def fit_decoupled_and_render(img_np, config=None, n_components=None, verbose=False):
    """
    Decoupled EM: fit the GMM on spatial coordinates (x, y) only, then derive
    each splat's color from the responsibility-weighted average of pixel colors
    within that component.

    Motivation
    ----------
    In the joint 5D fit, the 5×5 covariance matrix has off-diagonal blocks that
    couple position to color.  This causes a concrete problem: a large smooth
    region with a color gradient produces a Gaussian that is *spatially smaller*
    than it should be, because color variance leaks into the spatial covariance
    estimate and tightens it.

    Decoupling removes this coupling entirely:
      - Spatial EM  : GMM on (x_norm, y_norm) → 2×2 covariance = true splat shape
      - Color step  : c_k = Σ_n r_{nk} · color_n / N_k  (responsibility-weighted mean)

    This matches the structural assumption of 3DGS far more closely: Gaussians
    define spatial footprints, and color (via spherical harmonics) is a separate
    per-splat attribute optimised independently.

    An optional second stage refines colors by fitting a per-component color GMM
    on the assigned pixels, capturing intra-splat color variation.  Enable via
    config['decoupled']['color_gmm'] = true.

    Config keys (under config['decoupled']):
        color_gmm        : fit a small GMM per component to get better color
                           representation (default False — mean color only)
        color_gmm_k      : components in each per-splat color GMM (default 1,
                           i.e. just the mean; set >1 only for large splat counts)
        spatial_cov_type : covariance type for the spatial GMM (default 'full')
    """
    data_5d, h, w = _prepare_data(img_np)
    n_comp = _resolve_n_comp(config, n_components)
    cfg = config or {}
    dec_cfg = cfg.get('decoupled', {})
    spatial_cov_type = dec_cfg.get('spatial_cov_type', cfg.get('covariance_type', 'full'))
    max_iter = cfg.get('max_iter', 100)
    color_gmm = dec_cfg.get('color_gmm', False)

    # Split into spatial and color arrays
    spatial = data_5d[:, :2]        # (N, 2)  normalised (x, y)
    colors  = data_5d[:, 2:]        # (N, 3)  (r, g, b) in [0, 1]

    if verbose:
        print(f"  [Decoupled EM] Fitting spatial GMM ({n_comp} components, "
              f"cov={spatial_cov_type})...")
    start_time = time.time()

    # ---- Stage 1: spatial GMM on (x, y) ----
    spatial_gmm = GaussianMixture(
        n_components=n_comp,
        covariance_type=spatial_cov_type,
        max_iter=max_iter,
        init_params='kmeans',
        random_state=42,
        verbose=0,
    )
    spatial_gmm.fit(spatial)
    fit_time = time.time() - start_time

    if verbose:
        print(f"  ✓ Spatial GMM fitted in {fit_time:.2f}s "
              f"(converged={spatial_gmm.converged_}, iters={spatial_gmm.n_iter_})")

    # ---- Stage 2: derive per-component colors ----
    #
    # responsibilities: r_{nk} = p(Z_n = k | x_n, theta_spatial)
    # shape: (N, K)
    responsibilities = spatial_gmm.predict_proba(spatial)   # (N, K)
    Nk = responsibilities.sum(axis=0)                       # (K,)

    # Responsibility-weighted mean color per component
    # c_k = (r[:, k] @ colors) / N_k
    mean_colors = (responsibilities.T @ colors) / Nk[:, np.newaxis]  # (K, 3)
    mean_colors = np.clip(mean_colors, 0, 1)

    if color_gmm:
        # Optional: replace mean color with the mode of a per-component color GMM.
        # Useful when a splat covers a region with distinct sub-colors (e.g.
        # sky/cloud boundary).  We pick the dominant component's mean.
        color_gmm_k = dec_cfg.get('color_gmm_k', 3)
        if verbose:
            print(f"  Fitting per-component color GMMs (k={color_gmm_k})...")
        for k in range(n_comp):
            # Weight each pixel by its responsibility for component k
            weights_k = responsibilities[:, k]
            if weights_k.sum() < 1e-6:
                continue
            # Subsample to keep it fast — cap at 5000 pixels per component
            idx = np.where(weights_k > 1e-4)[0]
            if len(idx) < color_gmm_k:
                continue
            if len(idx) > 5000:
                idx = idx[np.random.default_rng(k).choice(
                    len(idx), 5000, replace=False)]
            sub_colors = colors[idx]
            sub_weights = weights_k[idx]
            sub_weights /= sub_weights.sum()
            try:
                cg = GaussianMixture(
                    n_components=min(color_gmm_k, len(idx)),
                    covariance_type='full',
                    max_iter=50,
                    random_state=42,
                    weights_init=None,
                )
                cg.fit(sub_colors, )
                # Use the dominant component's mean as the splat color
                dominant = np.argmax(cg.weights_)
                mean_colors[k] = np.clip(cg.means_[dominant], 0, 1)
            except Exception:
                pass  # fall back to weighted mean if GMM fails

    # ---- Extract spatial parameters ----
    weights = spatial_gmm.weights_
    means_pos = spatial_gmm.means_ * np.array([[w, h]])     # back to pixel space

    if spatial_cov_type == 'full':
        covs_pos = spatial_gmm.covariances_ * np.array([[[w, h], [w, h]]]) ** 2
    elif spatial_cov_type == 'tied':
        cov = spatial_gmm.covariances_
        covs_pos = np.tile(cov * np.array([[w, h], [w, h]]) ** 2, (n_comp, 1, 1))
    elif spatial_cov_type == 'diag':
        diag_vars = spatial_gmm.covariances_
        covs_pos = np.zeros((n_comp, 2, 2))
        covs_pos[:, 0, 0] = diag_vars[:, 0] * w ** 2
        covs_pos[:, 1, 1] = diag_vars[:, 1] * h ** 2
    elif spatial_cov_type == 'spherical':
        variances = spatial_gmm.covariances_
        covs_pos = np.zeros((n_comp, 2, 2))
        covs_pos[:, 0, 0] = variances * w ** 2
        covs_pos[:, 1, 1] = variances * h ** 2
    else:
        raise ValueError(f"Unsupported spatial_cov_type: {spatial_cov_type}")

    # ---- Render ----
    if verbose:
        print(f"  Rendering {n_comp} splats...")
    render_start = time.time()
    render_np = render_gaussians(means_pos, covs_pos, mean_colors, weights, (h, w))
    render_time = time.time() - render_start

    return render_np, {
        'fit_time': fit_time,
        'render_time': render_time,
        'total_time': time.time() - start_time,
        'converged': spatial_gmm.converged_,
        'n_iter': spatial_gmm.n_iter_,
    }


# ---------------------------------------------------------------------------
# Variant 6 — Augmented EM  (engineered feature space before clustering)
# ---------------------------------------------------------------------------

def _compute_augmented_features(img_np, feature_groups, weights):
    """
    Build an enriched feature matrix from img_np.

    The base 5D [x_norm, y_norm, r, g, b] is always included as the first five
    columns.  Each enabled feature group appends further columns.  All groups
    are independently z-score normalised before weighting, so no single group
    dominates the EM distance metric by virtue of scale alone.

    The per-group weight (from config['augmented']['weights']) lets you tune
    how strongly each group pulls on the clustering.  A weight of 0 excludes
    the group; 1.0 matches the scale of the base features after normalisation.

    Parameters
    ----------
    img_np        : (H, W, 3) float32 image in [0, 1]
    feature_groups: list of str — which groups to add beyond the base 5D
    weights       : dict mapping group name -> float scalar weight

    Returns
    -------
    data_aug : (N, D_aug) float32 — augmented feature matrix
    h, w     : image dimensions
    """
    from skimage.color import rgb2lab, rgb2hsv
    from scipy.ndimage import uniform_filter

    h, w, _ = img_np.shape
    N = h * w

    # ---- Base 5D (always present, first five columns) ----
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords_norm = np.stack([xx.ravel(), yy.ravel()], axis=1) / np.array([[w, h]])
    pixels = img_np.reshape(N, 3)
    base = np.concatenate([coords_norm, pixels], axis=1).astype(np.float32)

    groups = [base]

    def _normalise_and_weight(arr, name):
        """Z-score normalise then scale by the configured weight."""
        std = arr.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        normed = (arr - arr.mean(axis=0, keepdims=True)) / std
        return (normed * weights.get(name, 1.0)).astype(np.float32)

    # ---- LAB color (perceptually uniform Euclidean distance) ----
    # L: lightness 0-100, a/b: opponent color axes
    # After normalisation, Euclidean distance in LAB ≈ perceived color difference.
    if 'lab' in feature_groups:
        lab = rgb2lab(img_np).reshape(N, 3).astype(np.float32)
        groups.append(_normalise_and_weight(lab, 'lab'))

    # ---- HSV color (hue as semantic grouping signal) ----
    # Hue separates sky/foliage/skin even when their L or RGB values overlap.
    # Note: hue wraps at 0/1 (red), so distant hues may appear close. For most
    # natural images this is rarely a problem in practice.
    if 'hsv' in feature_groups:
        hsv = rgb2hsv(img_np).reshape(N, 3).astype(np.float32)
        groups.append(_normalise_and_weight(hsv, 'hsv'))

    # ---- Luminance gradient (magnitude + orientation) ----
    # Pixels on opposite sides of an edge should cluster differently even if
    # they have similar colors.  Gradient magnitude and direction encode edge
    # membership, biasing splats to respect image boundaries.
    if 'gradient' in feature_groups:
        lum = 0.2126 * img_np[:, :, 0] + 0.7152 * img_np[:, :, 1] \
            + 0.0722 * img_np[:, :, 2]     # (H, W) luminance
        gy = np.gradient(lum, axis=0)
        gx = np.gradient(lum, axis=1)
        mag   = np.sqrt(gx ** 2 + gy ** 2).reshape(N, 1)
        angle = np.arctan2(gy, gx).reshape(N, 1)   # [-pi, pi]
        grad_feats = np.concatenate([mag, angle], axis=1).astype(np.float32)
        groups.append(_normalise_and_weight(grad_feats, 'gradient'))

    # ---- Local variance (texture vs. flat region) ----
    # High local variance = textured; low = smooth.  This single scalar helps
    # EM separate detailed regions (fur, foliage, fabric) from smooth ones
    # (sky, skin, water), which should get very different splat shapes.
    if 'local_variance' in feature_groups:
        patch = int(weights.get('local_variance_patch', 5))
        lum = 0.2126 * img_np[:, :, 0] + 0.7152 * img_np[:, :, 1] \
            + 0.0722 * img_np[:, :, 2]
        mean_sq = uniform_filter(lum ** 2, size=patch)
        sq_mean = uniform_filter(lum, size=patch) ** 2
        variance = np.maximum(mean_sq - sq_mean, 0).reshape(N, 1).astype(np.float32)
        groups.append(_normalise_and_weight(variance, 'local_variance'))

    # ---- Spatial encoding (polar coordinates) ----
    # Radial distance from image centre + polar angle capture circular/radial
    # structure (vignetting, centred subjects, radial blur) that Cartesian
    # coordinates miss.
    if 'spatial_encoding' in feature_groups:
        cx, cy = 0.5, 0.5
        dx = coords_norm[:, 0] - cx
        dy = coords_norm[:, 1] - cy
        radius = np.sqrt(dx ** 2 + dy ** 2).reshape(N, 1)
        angle  = np.arctan2(dy, dx).reshape(N, 1)
        polar  = np.concatenate([radius, angle], axis=1).astype(np.float32)
        groups.append(_normalise_and_weight(polar, 'spatial_encoding'))

    data_aug = np.concatenate(groups, axis=1)
    return data_aug, h, w


def fit_augmented_and_render(img_np, config=None, n_components=None, verbose=False):
    """
    Augmented EM: engineer a richer feature space before running the GMM,
    giving EM more degrees of freedom to discover semantically coherent clusters.

    Key insight
    -----------
    EM clusters in whatever metric space you hand it.  The base 5D space
    [x, y, r, g, b] conflates perceptually unlike quantities with no shared
    scale.  By appending derived features — LAB color, HSV, luminance gradient,
    local variance, polar spatial encoding — we bias the distance metric toward
    groupings that are invisible to the raw 5D basis.

    The augmented dimensions only affect *which pixels get grouped together*.
    Rendering always uses the first five dimensions of the GMM means
    [x_norm, y_norm, r, g, b], so the extra columns disappear after fitting.
    This is a free lunch in terms of the rendering pipeline.

    Config keys (under config['augmented']):
        feature_groups : list of groups to enable (default: all)
                         choices: lab, hsv, gradient, local_variance,
                                  spatial_encoding
        weights        : per-group float multiplier after z-score normalisation
                         (default: 1.0 for each group)
                         e.g.  gradient: 2.0 doubles gradient's pull on clustering
        covariance_type: GMM covariance type (inherits from em.covariance_type
                         if not set here; default 'full')

    Example config:
        augmented:
          feature_groups: [lab, gradient, local_variance]
          weights:
            lab: 1.5
            gradient: 2.0
            local_variance: 0.5
    """
    cfg = config or {}
    aug_cfg = cfg.get('augmented', {})
    n_comp = _resolve_n_comp(config, n_components)

    all_groups = ['lab', 'hsv', 'gradient', 'local_variance', 'spatial_encoding']
    feature_groups = aug_cfg.get('feature_groups', all_groups)
    feat_weights   = aug_cfg.get('weights', {})
    covariance_type = aug_cfg.get('covariance_type', cfg.get('covariance_type', 'full'))
    max_iter = cfg.get('max_iter', 100)

    if verbose:
        print(f"  [Augmented EM] Feature groups: {feature_groups}")
        print(f"  Building augmented feature matrix...")

    start_time = time.time()
    data_aug, h, w = _compute_augmented_features(img_np, feature_groups, feat_weights)
    D_aug = data_aug.shape[1]

    if verbose:
        print(f"  Feature space: 5D base → {D_aug}D augmented")
        print(f"  Fitting GMM ({n_comp} components, cov={covariance_type})...")

    gmm = GaussianMixture(
        n_components=n_comp,
        covariance_type=covariance_type,
        max_iter=max_iter,
        init_params='kmeans',
        random_state=42,
        verbose=0,
    )
    gmm.fit(data_aug)
    fit_time = time.time() - start_time

    if verbose:
        print(f"  ✓ fitted in {fit_time:.2f}s "
              f"(converged={gmm.converged_}, iters={gmm.n_iter_}, D={D_aug})")

    # ---- Extract only the base 5D slice for rendering ----
    # GMM means are in augmented space; first 5 dims are always [x,y,r,g,b]
    means_5d   = gmm.means_[:, :5]
    means_pos  = means_5d[:, :2] * np.array([[w, h]])
    mean_colors = np.clip(means_5d[:, 2:], 0, 1)
    weights     = gmm.weights_

    # Spatial covariance: slice top-left 2×2 block from the full D_aug covariance
    if covariance_type == 'full':
        covs_pos = gmm.covariances_[:, :2, :2] * np.array([[[w, h], [w, h]]]) ** 2
    elif covariance_type == 'tied':
        cov = gmm.covariances_
        covs_pos = np.tile(cov[:2, :2] * np.array([[w, h], [w, h]]) ** 2, (n_comp, 1, 1))
    elif covariance_type == 'diag':
        diag_vars = gmm.covariances_
        covs_pos = np.zeros((n_comp, 2, 2))
        covs_pos[:, 0, 0] = diag_vars[:, 0] * w ** 2
        covs_pos[:, 1, 1] = diag_vars[:, 1] * h ** 2
    elif covariance_type == 'spherical':
        variances = gmm.covariances_
        covs_pos = np.zeros((n_comp, 2, 2))
        covs_pos[:, 0, 0] = variances * w ** 2
        covs_pos[:, 1, 1] = variances * h ** 2
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    if verbose:
        print(f"  Rendering {n_comp} splats...")
    render_start = time.time()
    render_np = render_gaussians(means_pos, covs_pos, mean_colors, weights, (h, w))
    render_time = time.time() - render_start

    return render_np, {
        'fit_time': fit_time,
        'render_time': render_time,
        'total_time': time.time() - start_time,
        'converged': gmm.converged_,
        'n_iter': gmm.n_iter_,
        'feature_dim': D_aug,
    }


# ---------------------------------------------------------------------------
# Variant 7 — Variational Bayes GMM  (automatic K via Dirichlet process prior)
# ---------------------------------------------------------------------------

def fit_vbgmm_and_render(img_np, config=None, n_components=None, verbose=False):
    """
    Variational Bayes GMM via sklearn's BayesianGaussianMixture.

    The core difference from standard EM
    -------------------------------------
    Standard EM treats K and the parameters (pi_k, mu_k, Sigma_k) as fixed
    unknowns to be point-estimated.  It answers the question:
        "Given that there are exactly K components, what are the best parameters?"

    Variational Bayes GMM places priors over *all* parameters and optimises a
    lower bound on the marginal likelihood (the ELBO), treating everything as
    uncertain:

        pi   ~ Dirichlet(alpha_0 / K, ..., alpha_0 / K)   mixing weights
        mu_k ~ Normal(mu_0, (beta_0 * Lambda_k)^{-1})      component means
        Lambda_k ~ Wishart(W_0, nu_0)                       precisions

    The ELBO optimisation is still coordinate ascent (structurally identical
    to EM), but each "step" updates the *parameters of a distribution* over
    theta rather than a point estimate.  The result:

    1. Automatic sparsity — components with no evidence receive a near-zero
       weight alpha_k → 0.  The model decides how many splats it needs.
       n_components is now an *upper bound*, not a fixed K.

    2. Uncertainty quantification — each splat has a posterior distribution
       over its position and shape, not just a point estimate.  We render
       using the posterior mean (same as EM) but the uncertainty is available.

    3. Better regularisation — the Wishart prior over precision matrices
       prevents covariance collapse, making it more robust than EM at low
       data density regions.

    The weight_concentration_prior parameter
    ----------------------------------------
    This is alpha_0 in the Dirichlet prior, and it is the single most
    important knob:

        alpha_0 >> 1  →  prior pulls weights toward uniform → more active components
        alpha_0 << 1  →  prior encourages sparsity → fewer active components
        alpha_0 = 1/K →  (sklearn default) neutral; let data decide

    For splatting, lower alpha_0 is usually better — you want the model to use
    as few splats as needed for each region rather than spreading weight evenly.

    Config keys (under config['vbgmm']):
        weight_concentration_prior : float  (default 1/K — sklearn default)
        weight_concentration_prior_type : 'dirichlet_process' | 'dirichlet_distribution'
            'dirichlet_process'     → stick-breaking; strongest sparsity
            'dirichlet_distribution' → symmetric Dirichlet; softer sparsity
        active_threshold : float  weight below this is considered inactive (default 0.01/K)
        mean_precision_prior : float  beta_0 (default None = sklearn default)
    """
    from sklearn.mixture import BayesianGaussianMixture

    data, h, w = _prepare_data(img_np)
    n_comp = _resolve_n_comp(config, n_components)
    cfg    = config or {}
    vb_cfg = cfg.get('vbgmm', {})

    covariance_type  = cfg.get('covariance_type', 'full')
    max_iter         = cfg.get('max_iter', 100)
    wcp_type         = vb_cfg.get('weight_concentration_prior_type',
                                   'dirichlet_process')
    wcp              = vb_cfg.get('weight_concentration_prior', 1.0 / n_comp)
    active_threshold = vb_cfg.get('active_threshold', 0.01 / n_comp)
    mean_prec_prior  = vb_cfg.get('mean_precision_prior', None)

    if verbose:
        print(f"  [VB-GMM] Upper bound K={n_comp}, "
              f"prior_type={wcp_type}, alpha_0={wcp:.5f}")
    start_time = time.time()

    kwargs = dict(
        n_components=n_comp,
        covariance_type=covariance_type,
        max_iter=max_iter,
        weight_concentration_prior_type=wcp_type,
        weight_concentration_prior=wcp,
        init_params='kmeans',
        random_state=42,
        verbose=0,
    )
    if mean_prec_prior is not None:
        kwargs['mean_precision_prior'] = mean_prec_prior

    vbgmm = BayesianGaussianMixture(**kwargs)
    vbgmm.fit(data)
    fit_time = time.time() - start_time

    # ---- Count active components ----
    active_mask = vbgmm.weights_ > active_threshold
    n_active    = active_mask.sum()

    if verbose:
        print(f"  ✓ fitted in {fit_time:.2f}s "
              f"(converged={vbgmm.converged_}, iters={vbgmm.n_iter_})")
        print(f"  Active components: {n_active} / {n_comp}  "
              f"(threshold={active_threshold:.5f})")
        inactive_weights = vbgmm.weights_[~active_mask]
        if len(inactive_weights):
            print(f"  Pruned weights: min={inactive_weights.min():.6f}  "
                  f"max={inactive_weights.max():.6f}")

    # ---- Render active components only ----
    # Inactive splats contribute negligibly to the render but waste time in
    # render_gaussians.  Subsetting here makes the comparison fair: VB-GMM
    # with K_active << K_upper is not penalised for having small K.
    means_all  = vbgmm.means_
    weights_all = vbgmm.weights_

    means_pos_all  = means_all[:, :2] * np.array([[w, h]])
    colors_all     = np.clip(means_all[:, 2:], 0, 1)

    if covariance_type == 'full':
        covs_pos_all = vbgmm.covariances_[:, :2, :2] * np.array([[[w, h], [w, h]]]) ** 2
    elif covariance_type == 'tied':
        cov = vbgmm.covariances_
        covs_pos_all = np.tile(cov[:2, :2] * np.array([[w, h], [w, h]]) ** 2,
                                (n_comp, 1, 1))
    elif covariance_type == 'diag':
        covs_pos_all = np.zeros((n_comp, 2, 2))
        covs_pos_all[:, 0, 0] = vbgmm.covariances_[:, 0] * w ** 2
        covs_pos_all[:, 1, 1] = vbgmm.covariances_[:, 1] * h ** 2
    elif covariance_type == 'spherical':
        covs_pos_all = np.zeros((n_comp, 2, 2))
        covs_pos_all[:, 0, 0] = vbgmm.covariances_ * w ** 2
        covs_pos_all[:, 1, 1] = vbgmm.covariances_ * h ** 2
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    # Render active splats only
    means_pos  = means_pos_all[active_mask]
    covs_pos   = covs_pos_all[active_mask]
    colors     = colors_all[active_mask]
    weights    = weights_all[active_mask]
    weights    = weights / weights.sum()   # renormalise after pruning

    if verbose:
        print(f"  Rendering {n_active} active splats...")
    render_start = time.time()
    render_np    = render_gaussians(means_pos, covs_pos, colors, weights, (h, w))
    render_time  = time.time() - render_start

    return render_np, {
        'fit_time':       fit_time,
        'render_time':    render_time,
        'total_time':     time.time() - start_time,
        'converged':      vbgmm.converged_,
        'n_iter':         vbgmm.n_iter_,
        'n_active':       int(n_active),       # key new metric
        'n_upper':        n_comp,
        'utilisation':    float(n_active / n_comp),
    }


# ---------------------------------------------------------------------------
# Dispatch registry — maps variant name -> fit function
# ---------------------------------------------------------------------------

FIT_REGISTRY = {
    'standard':  fit_gmm_and_render,
    'hard':      fit_hard_em_and_render,
    'minibatch': fit_minibatch_em_and_render,
    'gradient':  fit_gradient_em_and_render,
    'decoupled': fit_decoupled_and_render,
    'augmented': fit_augmented_and_render,
    'vbgmm':     fit_vbgmm_and_render,
}