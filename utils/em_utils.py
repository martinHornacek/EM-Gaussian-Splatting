"""
em_utils.py - Shared utilities for EM fitting and rendering

Contains: load_config, download_kodak_dataset, _prepare_data, render_gaussians.
"""

import os
import yaml
import numpy as np
from pathlib import Path
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


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
        print(f"Kodak dataset present ({len(existing)} images)")
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
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")
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


def render_gaussians(means, covs, colors, weights, image_size):
    """
    Alpha-composite Gaussian splats onto an (H, W, 3) canvas.

    Each Gaussian k contributes to the canvas proportionally to its mixture
    weight ``weights[k]``.  The PDF of each Gaussian is peak-normalised to 1
    before scaling by the weight, so that ``weights`` carry the relative
    importance of each component (not the absolute intensity scale).  The final
    pixel value is the weighted average of all Gaussian colours at that pixel —
    equivalent to alpha-compositing with per-pixel alpha equal to the joint
    mixture probability.

    This renderer is used for:
      * the initial EM approximation of the full image, and
      * rendering positive/negative residual corrections (where colours carry
        the actual residual magnitude learned by the GMM).

    Args:
        means      : (K, 2)    Gaussian centres in pixel space (x, y).
        covs       : (K, 2, 2) Spatial covariance matrices in pixel space.
        colors     : (K, 3)    Per-Gaussian RGB colour.
        weights    : (K,)      Mixture weights (sum to ~1).
        image_size : (H, W)    Canvas dimensions.

    Returns:
        (H, W, 3) float32 image clipped to [0, 1].
    """
    h, w = image_size
    canvas     = np.zeros((h, w, 3), dtype=np.float32)
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)
    # Build pixel-coordinate grid; pos[i, j] = [x=j, y=i]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    pos = np.dstack((X, Y))
    for mean, cov, color, wgt in zip(means, covs, colors, weights):
        try:
            rv  = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
            pdf = rv.pdf(pos)  # shape (H, W)
        except (np.linalg.LinAlgError, ValueError):
            continue
        # Skip any Gaussian whose PDF contains non-finite values (can arise
        # from near-singular covariance matrices that pass allow_singular).
        if not np.isfinite(pdf).all():
            continue
        # Peak-normalise then scale by mixture weight so weights govern
        # the relative contribution of each component.
        pdf = pdf / (np.max(pdf) + 1e-8) * wgt
        pdf = np.expand_dims(pdf, axis=-1)   # (H, W, 1) for broadcast
        canvas     += pdf * color
        weight_sum += pdf
    # Weighted-average composite; pixels with zero total weight stay black.
    render = np.divide(
        canvas, weight_sum,
        out=np.zeros_like(canvas),
        where=weight_sum > 1e-8,
    )
    return np.clip(render, 0.0, 1.0)


def render_gaussians_signed(means, covs, colors, amplitudes, image_size):
    """
    Pure additive Gaussian splatting with signed amplitudes.

    Each Gaussian k contributes::

        amplitudes[k] * (pdf_k(x,y) / max(pdf_k)) * colors[k]

    where the spatial profile is peak-normalised to 1 so that ``amplitudes``
    carries the absolute scale of the contribution.  Negative amplitudes
    produce subtractive corrections.

    Unlike ``render_gaussians``, this function does **not** clip the output —
    the caller is responsible for clipping when compositing with a base image.

    Args:
        means      : (K, 2)   Gaussian centres in pixel coordinates.
        covs       : (K, 2, 2) Spatial covariance matrices in pixel coordinates.
        colors     : (K, 3)   Per-Gaussian RGB colour (the value at the peak).
        amplitudes : (K,)     Signed scale factors; negative values subtract.
        image_size : (H, W)   Output canvas dimensions.

    Returns:
        (H, W, 3) float32 array — signed additive canvas (not clipped).
    """
    h, w = image_size
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    pos = np.dstack((X, Y))
    for mean, cov, color, amp in zip(means, covs, colors, amplitudes):
        try:
            rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
            pdf = rv.pdf(pos)
            peak = np.max(pdf)
            if peak < 1e-10:
                continue
            pdf_norm = pdf / peak                          # peak-normalised to 1
            canvas += amp * pdf_norm[:, :, np.newaxis] * color[np.newaxis, np.newaxis, :]
        except (np.linalg.LinAlgError, ValueError):
            continue
    return canvas


def render_residual_correction(residual_np, means, covs, weights, image_size):
    """
    Render a residual correction using Gaussian SPATIAL COVERAGE to weight
    the ACTUAL residual values.  Unlike ``render_gaussians`` (which assigns
    GMM cluster-centre colours to all covered pixels), this function:

      correction[p] = clip(sum_k(pdf_k(p)/max(pdf_k)*wgt_k), 0, 1) * residual[p]

    Key properties:
      * The correction at each pixel is bounded by the actual residual —
        no oracle (gt_np) is needed to prevent overshoot.
      * The Gaussians guide WHICH pixels are corrected (coverage map),
        not HOW MUCH they are corrected (that comes from the residual itself).
      * Pixels fully inside Gaussian blobs (coverage ≈ 1) are corrected
        proportionally to their full residual; pixels outside are untouched.

    This is the honest replacement for the oracle-clamped update.  It makes
    the algorithm equivalent to "Gaussian-guided residual coverage correction"
    rather than "oracle-bounded density estimation".

    Args:
        residual_np : (H, W, 3) float32  non-negative residual channel
                      (either I_res_pos or I_res_neg from the main loop).
        means       : (K, 2)    Gaussian centres in pixel space.
        covs        : (K, 2, 2) Spatial covariance matrices.
        weights     : (K,)      GMM mixture weights.
        image_size  : (H, W).

    Returns:
        (H, W, 3) float32  correction image, bounded by *residual_np*.
    """
    h, w = image_size
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    pos  = np.dstack((X, Y))

    coverage = np.zeros((h, w), dtype=np.float32)
    for mean, cov, wgt in zip(means, covs, weights):
        try:
            rv  = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
            pdf = rv.pdf(pos)
            peak = np.max(pdf)
            if peak < 1e-10:
                continue
            if not np.isfinite(pdf).all():
                continue
            coverage += (pdf / peak * wgt).astype(np.float32)
        except (np.linalg.LinAlgError, ValueError):
            continue

    # Clip coverage to [0, 1] so it acts as a per-pixel blending weight.
    coverage = np.clip(coverage, 0.0, 1.0)
    return coverage[:, :, np.newaxis] * residual_np   # (H, W, 3), bounded by residual


def fit_em_to_distribution(img_np, n_components, em_config, use_minibatch=True):
    """
    Fit a 5D GMM [x_norm, y_norm, r, g, b] to a non-negative image, weighting
    each pixel proportionally to its summed channel intensity so that brighter
    pixels attract more Gaussian components.

    Intensity-weighted sampling is used (sampling with replacement according to
    pixel brightness) rather than sklearn's unweighted ``fit``.

    Args:
        img_np        : (H, W, 3) float32, non-negative values.
        n_components  : Number of Gaussian components to fit.
        em_config     : Dict — same EM config as elsewhere
                        (``covariance_type``, ``max_iter``, ``minibatch``).
        use_minibatch : bool (default True).
                        If True  : cap the sample pool at
                                   max(n_components * 20, 20 000) pixels.
                        If False : use ALL pixels (no cap) — slower but gives
                                   the GMM full coverage of the residual image.
                        In both cases the draw is intensity-weighted with
                        replacement so high-brightness regions attract more
                        Gaussian components.

    Returns:
        ``(means_px, covs_px, colors, weights)`` — pixel-space GMM parameters,
        or ``None`` if the image has no signal or fitting fails.
    """
    h, w, _ = img_np.shape
    intensity = img_np.sum(axis=2).ravel()          # (H*W,)
    total_signal = float(intensity.sum())
    if total_signal < 1e-10:
        return None

    # Build 5D feature matrix [x_norm, y_norm, r, g, b]
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords_norm = np.stack([xx.ravel() / w, yy.ravel() / h], axis=1)
    data_5d = np.concatenate(
        [coords_norm, img_np.reshape(-1, 3)], axis=1
    ).astype(np.float32)

    # Normalised intensity weights for sampling
    probs = intensity / total_signal

    # Sample pool size
    n_pixels = len(data_5d)
    if use_minibatch:
        max_mb = em_config.get('minibatch', {}).get('max_minibatch_samples', 20_000)
        n_sub = max(n_components * 20, min(max_mb, n_pixels))
    else:
        n_sub = n_pixels   # full-batch: draw from all pixels

    rng = np.random.default_rng(42)
    chosen = rng.choice(n_pixels, size=n_sub, replace=True, p=probs)
    data_fit = data_5d[chosen]

    # Cap n_components to what the sampled data can support
    n_comp = min(n_components, max(1, len(data_fit) // 5))

    cov_type = em_config.get('covariance_type', 'full')
    gmm = GaussianMixture(
        n_components=n_comp,
        covariance_type=cov_type,
        max_iter=em_config.get('max_iter', 100),
        init_params='kmeans',
        random_state=42,
        reg_covar=1e-5,
    )
    try:
        gmm.fit(data_fit)
    except Exception:
        return None

    means_px  = gmm.means_[:, :2] * np.array([[w, h]])
    # Correct Jacobian transform (same as in _initial_em_fit):
    #   Σ_px = J @ Σ_norm @ J^T  where J = diag(w, h)
    cov_scale = np.array([[[w**2, w * h], [w * h, h**2]]])  # (1, 2, 2)
    covs_px   = gmm.covariances_[:, :2, :2] * cov_scale
    # Residual colour values live in [0, 1]; clip GMM means to the same range.
    colors    = np.clip(gmm.means_[:, 2:5], 0.0, 1.0)
    return means_px, covs_px, colors, gmm.weights_
