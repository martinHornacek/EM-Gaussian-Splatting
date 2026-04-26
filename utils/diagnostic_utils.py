"""
diagnostic_utils.py — Splat boundary diagnostics for EM variants

Entry point
-----------
    run_splat_diagnostics(gt_np, fitted_variants, output_dir, image_name)

Where fitted_variants is a dict:
    {
        'standard':  SplatFit(...),
        'augmented': SplatFit(...),
    }

Each SplatFit carries the GMM parameters needed to recompute responsibilities
and draw ellipses — see extract_splat_fit() below.

Panels produced
---------------
  1. Responsibility segmentation  — argmax(r) coloured per component
  2. Boundary overlay             — segmentation edges on original image
  3. Ellipse overlay              — 2σ spatial ellipses drawn on original image
  4. Entropy map                  — per-pixel responsibility entropy
  5. Coverage map                 — fraction of canvas each splat's 2σ ellipse covers
  6. Side-by-side render          — reconstruction vs ground truth

All panels are written to a single multi-page figure per image.
A compact per-variant summary strip is also written for quick comparison
across many images.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from dataclasses import dataclass, field
from pathlib import Path
from scipy.stats import multivariate_normal
from skimage.filters import sobel


# ---------------------------------------------------------------------------
# Data structure carrying everything needed for diagnostics
# ---------------------------------------------------------------------------

@dataclass
class SplatFit:
    """
    All parameters extracted from a fitted GMM, in pixel space.

    Attributes
    ----------
    means_pos   : (K, 2)   splat centres in pixel coordinates
    covs_pos    : (K, 2, 2) spatial covariance in pixel coordinates
    colors      : (K, 3)   RGB color per splat
    weights     : (K,)     mixing coefficients
    responsibilities : (N, K) soft assignments (may be None if unavailable)
    n_iter      : int      EM iterations used
    converged   : bool
    feature_dim : int      dimensionality of the space EM was fitted in
    variant     : str      human-readable name
    """
    means_pos:        np.ndarray
    covs_pos:         np.ndarray
    colors:           np.ndarray
    weights:          np.ndarray
    responsibilities: np.ndarray | None
    n_iter:           int
    converged:        bool
    feature_dim:      int
    variant:          str


def extract_splat_fit(gmm, covariance_type, h, w,
                      data_for_responsibilities=None,
                      feature_dim=5,
                      variant='unknown',
                      timing=None):
    """
    Build a SplatFit from a fitted sklearn GaussianMixture.

    Parameters
    ----------
    gmm                    : fitted GaussianMixture
    covariance_type        : str  ('full', 'tied', 'diag', 'spherical')
    h, w                   : image dimensions
    data_for_responsibilities : (N, D) array to recompute r from; pass the
                               same data used to fit the GMM
    feature_dim            : D (dimensionality of fitting space)
    variant                : name string
    timing                 : dict returned by fit_*_and_render (optional)
    """
    K = gmm.n_components
    means_pos  = gmm.means_[:, :2] * np.array([[w, h]])
    colors     = np.clip(gmm.means_[:, 2:5], 0, 1)
    weights    = gmm.weights_

    # Spatial covariance in pixel space
    if covariance_type == 'full':
        covs_pos = gmm.covariances_[:, :2, :2] * np.array([[[w, h], [w, h]]]) ** 2
    elif covariance_type == 'tied':
        cov = gmm.covariances_
        covs_pos = np.tile(cov[:2, :2] * np.array([[w, h], [w, h]]) ** 2, (K, 1, 1))
    elif covariance_type == 'diag':
        covs_pos = np.zeros((K, 2, 2))
        covs_pos[:, 0, 0] = gmm.covariances_[:, 0] * w ** 2
        covs_pos[:, 1, 1] = gmm.covariances_[:, 1] * h ** 2
    elif covariance_type == 'spherical':
        covs_pos = np.zeros((K, 2, 2))
        covs_pos[:, 0, 0] = gmm.covariances_ * w ** 2
        covs_pos[:, 1, 1] = gmm.covariances_ * h ** 2
    else:
        raise ValueError(f"Unsupported covariance_type: {covariance_type}")

    # Responsibilities
    r = None
    if data_for_responsibilities is not None:
        r = gmm.predict_proba(data_for_responsibilities)   # (N, K)

    t = timing or {}
    return SplatFit(
        means_pos=means_pos,
        covs_pos=covs_pos,
        colors=colors,
        weights=weights,
        responsibilities=r,
        n_iter=getattr(gmm, 'n_iter_', -1),
        converged=getattr(gmm, 'converged_', False),
        feature_dim=feature_dim,
        variant=variant,
    )


# ---------------------------------------------------------------------------
# Individual diagnostic computations
# ---------------------------------------------------------------------------

def _responsibility_segmentation(fit: SplatFit, h: int, w: int):
    """
    Return an (H, W, 3) RGB image where each pixel is coloured by its
    dominant component (argmax responsibility).

    If responsibilities are not stored, fall back to nearest-centroid in
    pixel space — a coarse but always-available approximation.
    """
    K = len(fit.weights)
    palette = cm.tab20(np.linspace(0, 1, K))[:, :3]   # (K, 3)

    if fit.responsibilities is not None:
        labels = np.argmax(fit.responsibilities, axis=1)   # (N,)
    else:
        # Nearest centroid fallback
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
        dists = np.linalg.norm(
            coords[:, None, :] - fit.means_pos[None, :, :], axis=2)  # (N, K)
        labels = np.argmin(dists, axis=1)

    seg_img = palette[labels].reshape(h, w, 3)
    return seg_img.astype(np.float32)


def _boundary_overlay(seg_img, gt_np, edge_color=(1.0, 1.0, 0.0), thickness=1):
    """
    Detect boundaries between responsibility regions and overlay them
    on the original image.

    Uses Sobel on each channel of the segmentation (R, G, B independently)
    and ORs the result to catch all colour transitions.
    """
    from skimage.morphology import binary_dilation, disk

    # Edge detect on each colour channel of the segmentation map
    edge_mask = np.zeros(seg_img.shape[:2], dtype=bool)
    for c in range(3):
        edge_mask |= sobel(seg_img[:, :, c]) > 0.05

    if thickness > 1:
        edge_mask = binary_dilation(edge_mask, disk(thickness - 1))

    overlay = gt_np.copy()
    for c, val in enumerate(edge_color):
        overlay[:, :, c][edge_mask] = val

    return overlay


def _entropy_map(fit: SplatFit, h: int, w: int):
    """
    Per-pixel Shannon entropy of the responsibility distribution.
    H(x) = -Σ_k r_k log(r_k)

    High entropy → pixel is contested between components (poorly separated).
    Low entropy  → pixel is cleanly owned by one component.
    """
    if fit.responsibilities is not None:
        r = fit.responsibilities
    else:
        # Approx from spatial Gaussian PDFs only
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
        K = len(fit.weights)
        W = np.zeros((h * w, K))
        for k in range(K):
            try:
                rv = multivariate_normal(mean=fit.means_pos[k],
                                         cov=fit.covs_pos[k],
                                         allow_singular=True)
                W[:, k] = fit.weights[k] * rv.pdf(coords)
            except Exception:
                pass
        W_sum = W.sum(axis=1, keepdims=True)
        r = np.where(W_sum > 1e-12, W / W_sum, 1.0 / K)

    eps = 1e-12
    entropy = -(r * np.log(r + eps)).sum(axis=1)   # (N,)
    return entropy.reshape(h, w)


def _coverage_stats(fit: SplatFit, h: int, w: int, n_std: float = 2.0):
    """
    For each splat, compute what fraction of the canvas falls within its
    n_std ellipse.  Returns (K,) array of coverage fractions.

    Uses the Mahalanobis radius: pixel p is inside splat k if
        (p - mu_k)^T Sigma_k^{-1} (p - mu_k) <= n_std^2
    """
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    K = len(fit.weights)
    coverages = np.zeros(K)

    for k in range(K):
        try:
            cov_inv = np.linalg.inv(fit.covs_pos[k])
        except np.linalg.LinAlgError:
            continue
        diff = coords - fit.means_pos[k]                          # (N, 2)
        mahal_sq = np.einsum('ni,ij,nj->n', diff, cov_inv, diff)  # (N,)
        coverages[k] = (mahal_sq <= n_std ** 2).mean()

    return coverages


def _draw_ellipses(ax, fit: SplatFit, n_std: float = 2.0,
                   max_ellipses: int = 200, alpha: float = 0.6):
    """
    Draw 2σ covariance ellipses on an existing axes.

    Draws at most max_ellipses splats (sorted by weight descending) to avoid
    visual clutter on large K.
    """
    K = len(fit.weights)
    palette = cm.tab20(np.linspace(0, 1, K))

    order = np.argsort(fit.weights)[::-1][:max_ellipses]

    for k in order:
        cov = fit.covs_pos[k]
        try:
            vals, vecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        # Clamp negative eigenvalues from numerical noise
        vals = np.maximum(vals, 1e-6)
        order_ev = vals.argsort()[::-1]
        vals, vecs = vals[order_ev], vecs[:, order_ev]
        angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width  = 2 * n_std * np.sqrt(vals[0])
        height = 2 * n_std * np.sqrt(vals[1])
        ell = Ellipse(
            xy=(fit.means_pos[k, 0], fit.means_pos[k, 1]),
            width=width, height=height, angle=angle,
            edgecolor=palette[k % len(palette)],
            facecolor='none', linewidth=0.8, alpha=alpha,
        )
        ax.add_patch(ell)

    # Draw centroids
    ax.scatter(fit.means_pos[order, 0], fit.means_pos[order, 1],
               s=8, c='white', zorder=5, linewidths=0.5,
               edgecolors='black')


# ---------------------------------------------------------------------------
# Full diagnostic figure for one image, multiple variants
# ---------------------------------------------------------------------------

def make_diagnostic_figure(gt_np, fits: dict[str, SplatFit],
                            n_std: float = 2.0):
    """
    Build a full diagnostic figure.

    Layout: rows = variants, columns = diagnostic panels.

    Columns
    -------
    0  Original image (repeated for reference)
    1  Responsibility segmentation
    2  Boundary overlay
    3  Ellipse overlay
    4  Entropy map
    5  Coverage histogram

    Parameters
    ----------
    gt_np : (H, W, 3) ground truth image
    fits  : dict  variant_name -> SplatFit
    n_std : float  sigma level for ellipses and coverage computation

    Returns
    -------
    fig : matplotlib Figure
    """
    h, w = gt_np.shape[:2]
    variant_names = list(fits.keys())
    n_variants = len(variant_names)

    col_titles = [
        'Original',
        'Responsibility\nsegmentation',
        'Boundary\noverlay',
        f'Ellipse overlay\n({n_std:.0f}σ)',
        'Entropy map\n(higher = more contested)',
        'Coverage\ndistribution',
    ]
    n_cols = len(col_titles)

    fig, axes = plt.subplots(
        n_variants, n_cols,
        figsize=(4 * n_cols, 3.8 * n_variants),
        squeeze=False,
    )

    for row, name in enumerate(variant_names):
        fit = fits[name]
        axes[row, 0].set_ylabel(name.upper(), fontsize=12, fontweight='bold',
                                rotation=90, labelpad=8)

        # ---- Col 0: original ----
        axes[row, 0].imshow(gt_np)
        axes[row, 0].axis('off')

        # ---- Col 1: segmentation ----
        seg = _responsibility_segmentation(fit, h, w)
        axes[row, 1].imshow(seg)
        axes[row, 1].axis('off')

        # ---- Col 2: boundary overlay ----
        boundary = _boundary_overlay(seg, gt_np)
        axes[row, 2].imshow(boundary)
        axes[row, 2].axis('off')

        # ---- Col 3: ellipse overlay ----
        ax = axes[row, 3]
        ax.imshow(gt_np)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        _draw_ellipses(ax, fit, n_std=n_std)
        ax.axis('off')

        # ---- Col 4: entropy map ----
        ent = _entropy_map(fit, h, w)
        im = axes[row, 4].imshow(ent, cmap='inferno',
                                  vmin=0, vmax=np.log(len(fit.weights)))
        axes[row, 4].axis('off')
        plt.colorbar(im, ax=axes[row, 4], fraction=0.046, pad=0.04)

        # ---- Col 5: coverage distribution ----
        coverages = _coverage_stats(fit, h, w, n_std=n_std)
        ax = axes[row, 5]
        ax.hist(coverages * 100, bins=30, color='steelblue', edgecolor='white',
                linewidth=0.4)
        ax.axvline(np.median(coverages) * 100, color='orange', linewidth=1.5,
                   label=f'median {np.median(coverages)*100:.2f}%')
        ax.set_xlabel('Canvas covered per splat (%)', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)

        # Subtitle: key stats
        K = len(fit.weights)
        median_cov = np.median(coverages) * 100
        mean_ent   = ent.mean()
        subtitle = (f"K={K}  |  D={fit.feature_dim}D  |  "
                    f"iters={fit.n_iter}  |  "
                    f"median coverage={median_cov:.3f}%  |  "
                    f"mean entropy={mean_ent:.3f}")
        axes[row, 1].set_title(subtitle, fontsize=7, pad=3)

    # Column headers (top row only)
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight='bold', pad=6)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Compact summary strip (one row per variant, one column per image)
# ---------------------------------------------------------------------------

def make_summary_strip(image_names, all_fits, gt_images):
    """
    A compact grid: rows = variants, columns = images.
    Each cell shows the boundary overlay at a glance.
    Useful for scanning many images quickly.

    Parameters
    ----------
    image_names : list of str
    all_fits    : list of dicts — one dict per image, variant_name -> SplatFit
    gt_images   : list of (H, W, 3) ground truth arrays

    Returns
    -------
    fig : matplotlib Figure
    """
    variant_names = list(all_fits[0].keys())
    n_images   = len(image_names)
    n_variants = len(variant_names)

    fig, axes = plt.subplots(
        n_variants, n_images,
        figsize=(3 * n_images, 2.5 * n_variants),
        squeeze=False,
    )

    for col, (img_name, fits, gt) in enumerate(
            zip(image_names, all_fits, gt_images)):
        h, w = gt.shape[:2]
        axes[0, col].set_title(img_name, fontsize=8, fontweight='bold')

        for row, vname in enumerate(variant_names):
            ax = axes[row, col]
            if vname in fits:
                fit = fits[vname]
                seg      = _responsibility_segmentation(fit, h, w)
                boundary = _boundary_overlay(seg, gt)
                ax.imshow(boundary)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(vname.upper(), fontsize=8,
                              fontweight='bold', rotation=90)

    plt.suptitle('Boundary overlay — all variants × all images',
                 fontsize=11, y=1.01)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: per-variant scalar diagnostic summary
# ---------------------------------------------------------------------------

def compute_diagnostic_scalars(fit: SplatFit, h: int, w: int,
                                n_std: float = 2.0) -> dict:
    """
    Return a dict of scalar diagnostics for a single variant+image.

    Metrics
    -------
    median_coverage_pct : median fraction of canvas inside each splat's 2σ ellipse
    mean_coverage_pct   : mean of the same
    coverage_gini       : Gini coefficient of coverage — 0 = perfectly uniform,
                          1 = one splat covers everything.  High Gini means
                          a few large splats dominate.
    mean_entropy        : mean per-pixel responsibility entropy
    max_entropy         : maximum (= log K for a uniform distribution)
    pct_high_entropy    : fraction of pixels with entropy > 0.5 * log(K)
                          (proxy for "poorly assigned" pixels)
    """
    coverages = _coverage_stats(fit, h, w, n_std=n_std)
    entropy   = _entropy_map(fit, h, w).ravel()
    K         = len(fit.weights)
    max_ent   = np.log(K)

    # Gini coefficient
    sorted_cov = np.sort(coverages)
    n = len(sorted_cov)
    cumsum = np.cumsum(sorted_cov)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_cov) /
            (n * cumsum[-1]) - (n + 1) / n) if cumsum[-1] > 0 else 0.0

    return {
        'median_coverage_pct': float(np.median(coverages) * 100),
        'mean_coverage_pct':   float(np.mean(coverages) * 100),
        'coverage_gini':       float(gini),
        'mean_entropy':        float(entropy.mean()),
        'max_entropy':         float(max_ent),
        'pct_high_entropy':    float((entropy > 0.5 * max_ent).mean() * 100),
    }


# ---------------------------------------------------------------------------
# Top-level entry point called from run_em_algorithm.py
# ---------------------------------------------------------------------------

def run_splat_diagnostics(gt_np, fits: dict[str, SplatFit],
                           output_dir: Path, image_name: str,
                           n_std: float = 2.0):
    """
    Generate and save all diagnostic figures for one image.

    Parameters
    ----------
    gt_np      : (H, W, 3) ground truth
    fits       : dict  variant_name -> SplatFit
    output_dir : Path  where to save figures
    image_name : str   used as filename stem
    n_std      : float sigma level for ellipses

    Returns
    -------
    scalars : dict  variant_name -> diagnostic scalar dict
    """
    h, w = gt_np.shape[:2]

    print(f"  Running splat diagnostics for {image_name}...")

    # Full diagnostic grid
    fig = make_diagnostic_figure(gt_np, fits, n_std=n_std)
    fig.savefig(output_dir / f'{image_name}_diagnostics.png',
                dpi=130, bbox_inches='tight')
    plt.close(fig)

    # Per-variant scalar summary
    scalars = {}
    for name, fit in fits.items():
        scalars[name] = compute_diagnostic_scalars(fit, h, w, n_std=n_std)

    # Print scalar table
    print(f"\n  {'Variant':<12} {'Med.cov%':>9} {'Mean.cov%':>10} "
          f"{'Gini':>7} {'MeanEnt':>9} {'%HighEnt':>9}")
    print(f"  {'-'*60}")
    for name, s in scalars.items():
        print(f"  {name:<12} {s['median_coverage_pct']:>9.4f} "
              f"{s['mean_coverage_pct']:>10.4f} "
              f"{s['coverage_gini']:>7.4f} "
              f"{s['mean_entropy']:>9.4f} "
              f"{s['pct_high_entropy']:>9.2f}%")

    return scalars