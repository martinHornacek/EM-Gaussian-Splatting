"""
tiling_utils.py — Helpers for tile-based image processing.

Responsibilities
----------------
* Compute tile bounding boxes that fully cover an image (border tiles are
  clipped to image edges so that no pixel is ever duplicated or missed).
* Extract and merge tile arrays.
* Transform Gaussian means from tile-local pixel coords to global pixel
  coords (the only transform needed — covariances are already in pixel units
  and are image-size-agnostic).
* Generate visual diagnostics: grid overlay, boundary mask, side-by-side
  comparison panels.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tile geometry
# ---------------------------------------------------------------------------

def compute_tile_bboxes(h: int, w: int, tile_h: int, tile_w: int) -> list[tuple[int, int, int, int]]:
    """
    Partition a (H, W) image into non-overlapping axis-aligned tiles.

    Returns a list of (y0, x0, y1, x1) tuples.  The last tile in each
    dimension may be smaller than (tile_h, tile_w) if the image dimensions
    are not exact multiples — tiles are clipped to image boundaries so that
    every pixel is covered exactly once.

    Args:
        h, w     : Image height and width in pixels.
        tile_h   : Requested tile height (will be clipped at bottom edge).
        tile_w   : Requested tile width  (will be clipped at right edge).

    Returns:
        List of (y0, x0, y1, x1), top-left inclusive, bottom-right exclusive.
    """
    bboxes: list[tuple[int, int, int, int]] = []
    y0 = 0
    while y0 < h:
        y1 = min(y0 + tile_h, h)
        x0 = 0
        while x0 < w:
            x1 = min(x0 + tile_w, w)
            bboxes.append((y0, x0, y1, x1))
            x0 = x1
        y0 = y1
    return bboxes


def n_tiles(h: int, w: int, tile_h: int, tile_w: int) -> int:
    """Number of tiles for the given image and tile dimensions."""
    import math
    return math.ceil(h / tile_h) * math.ceil(w / tile_w)


# ---------------------------------------------------------------------------
# Extract / merge
# ---------------------------------------------------------------------------

def extract_tiles(
    img_np: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
) -> list[np.ndarray]:
    """
    Slice tile sub-images from *img_np*.

    Args:
        img_np : (H, W, C) float32 or uint8.
        bboxes : List of (y0, x0, y1, x1) from ``compute_tile_bboxes``.

    Returns:
        List of (tile_h, tile_w, C) arrays — views into *img_np*.
    """
    return [img_np[y0:y1, x0:x1] for (y0, x0, y1, x1) in bboxes]


def merge_tiles(
    rendered_tiles: list[np.ndarray],
    bboxes: list[tuple[int, int, int, int]],
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Paste axis-aligned tile renders back into a full-image canvas.

    Each tile occupies its designated bounding box exactly — no blending is
    applied.  Visible seams at tile boundaries are expected at this stage and
    are addressed by the subsequent gradient-descent refinement step.

    Args:
        rendered_tiles : List of (tile_h_i, tile_w_i, C) float32 arrays,
                         one per bbox in *bboxes*.
        bboxes         : List of (y0, x0, y1, x1) tile bboxes.
        image_shape    : (H, W, C) of the target canvas.

    Returns:
        (H, W, C) float32 composite image.
    """
    canvas = np.zeros(image_shape, dtype=np.float32)
    for tile, (y0, x0, y1, x1) in zip(rendered_tiles, bboxes):
        canvas[y0:y1, x0:x1] = tile
    return canvas


# ---------------------------------------------------------------------------
# Coordinate transforms (tile-local ↔ global pixel space)
# ---------------------------------------------------------------------------

def local_to_global_means(
    means_local_px: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Shift Gaussian centres from tile-local to global pixel coordinates.

    Tile-local pixel coords have their origin at the tile's top-left corner.
    Adding (x0, y0) converts them to global coordinates.

    Covariance matrices are **already in pixel units** (computed via
    Jacobian transform J = diag(tile_w, tile_h) in ``_prepare_data``),
    so pixels are the same size globally and no covariance transform is
    needed.

    Args:
        means_local_px : (K, 2) float32 — (x, y) in tile-local pixel space.
        bbox           : (y0, x0, y1, x1) tile bounding box.

    Returns:
        (K, 2) float32 — (x, y) in global pixel space.
    """
    y0, x0, _, _ = bbox
    offset = np.array([[x0, y0]], dtype=np.float32)
    return means_local_px + offset


def global_to_local_means(
    means_global_px: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """Inverse of ``local_to_global_means``."""
    y0, x0, _, _ = bbox
    offset = np.array([[x0, y0]], dtype=np.float32)
    return means_global_px - offset


# ---------------------------------------------------------------------------
# Boundary mask
# ---------------------------------------------------------------------------

def boundary_mask(
    image_shape: tuple[int, int],
    bboxes: list[tuple[int, int, int, int]],
    border_px: int = 3,
) -> np.ndarray:
    """
    Boolean mask that is True for pixels within *border_px* of any tile edge.

    Used to focus the gradient-descent loss on the seam regions most affected
    by blocking artefacts.

    Args:
        image_shape : (H, W)
        bboxes      : Tile bounding boxes.
        border_px   : Width of the border band around each tile edge.

    Returns:
        (H, W) bool array.
    """
    h, w = image_shape
    mask = np.zeros((h, w), dtype=bool)
    for y0, x0, y1, x1 in bboxes:
        # Top/bottom edges of the tile
        mask[y0:min(y0 + border_px, y1), x0:x1] = True
        mask[max(y1 - border_px, y0):y1,  x0:x1] = True
        # Left/right edges of the tile
        mask[y0:y1, x0:min(x0 + border_px, x1)] = True
        mask[y0:y1, max(x1 - border_px, x0):x1] = True
    return mask


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def tile_grid_overlay(
    img_np: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    line_width: int = 1,
) -> np.ndarray:
    """
    Draw tile boundary lines on a copy of *img_np*.

    Args:
        img_np     : (H, W, 3) float32 image.
        bboxes     : Tile bounding boxes.
        color      : RGB line colour, default red.
        line_width : Width of boundary lines in pixels.

    Returns:
        (H, W, 3) float32 image with coloured boundary lines.
    """
    overlay = img_np.copy()
    h, w = img_np.shape[:2]
    c = np.array(color, dtype=np.float32)
    for y0, x0, y1, x1 in bboxes:
        # Horizontal edges
        for dy in range(line_width):
            r = y0 + dy
            if 0 <= r < h:
                overlay[r, x0:x1] = c
            r = y1 - 1 - dy
            if 0 <= r < h:
                overlay[r, x0:x1] = c
        # Vertical edges
        for dx in range(line_width):
            col = x0 + dx
            if 0 <= col < w:
                overlay[y0:y1, col] = c
            col = x1 - 1 - dx
            if 0 <= col < w:
                overlay[y0:y1, col] = c
    return overlay


def save_tile_diagnostic(
    out_path: Path,
    gt_np: np.ndarray,
    composite: np.ndarray,
    refined: np.ndarray | None,
    bboxes: list[tuple[int, int, int, int]],
    metrics_composite: dict,
    metrics_refined:   dict | None,
) -> None:
    """
    Save a diagnostic figure showing:
      - Ground truth with tile grid
      - Tile composite (no GD)
      - [Optional] After gradient-descent refinement
      - Residual maps before and after GD

    Args:
        out_path          : Output PNG file path.
        gt_np             : (H, W, 3) float32 ground truth.
        composite         : (H, W, 3) float32 tiled composite before GD.
        refined           : (H, W, 3) float32 after GD, or None.
        bboxes            : Tile bounding boxes for overlay.
        metrics_composite : Metrics dict for composite image.
        metrics_refined   : Metrics dict for refined image, or None.
    """
    if not _MPL_AVAILABLE:
        return

    n_panels = 4 if refined is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    fig.suptitle('Tiled EM — Diagnostic', fontsize=12, fontweight='bold')

    gt_grid = tile_grid_overlay(gt_np, bboxes)

    def _show(ax, img, title):
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    _show(axes[0], gt_grid,   f'Ground truth + {len(bboxes)} tile grid')

    psnr_c = metrics_composite.get('psnr', float('nan'))
    ssim_c = metrics_composite.get('ssim', float('nan'))
    _show(axes[1], composite,
          f'Tiled composite\nPSNR {psnr_c:.2f} dB  SSIM {ssim_c:.4f}')

    res_c = np.mean(np.abs(gt_np - composite), axis=2)
    axes[2].imshow(res_c, cmap='hot', vmin=0, vmax=0.2)
    axes[2].set_title('Residual (composite)', fontsize=9)
    axes[2].axis('off')

    if refined is not None and metrics_refined is not None:
        psnr_r = metrics_refined.get('psnr', float('nan'))
        ssim_r = metrics_refined.get('ssim', float('nan'))
        delta_p = psnr_r - psnr_c
        sign    = '+' if delta_p >= 0 else ''
        _show(axes[3], refined,
              f'After GD refinement\nPSNR {psnr_r:.2f} dB ({sign}{delta_p:.2f})')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_boundary_comparison(
    out_path: Path,
    gt_np: np.ndarray,
    composite: np.ndarray,
    refined: np.ndarray | None,
    bboxes: list[tuple[int, int, int, int]],
    border_px: int = 8,
) -> None:
    """
    Zoom in on the worst-affected tile-boundary region and show a before/after
    comparison of the gradient-descent refinement.
    """
    if not _MPL_AVAILABLE:
        return

    bmask = boundary_mask(gt_np.shape[:2], bboxes, border_px=border_px)
    # Find the row and column of highest mean residual along tile boundaries
    res_boundary = np.mean(np.abs(gt_np - composite), axis=2) * bmask
    if res_boundary.max() < 1e-8:
        return  # Nothing interesting to zoom into

    flat_idx = int(np.argmax(res_boundary))
    cy, cx   = divmod(flat_idx, gt_np.shape[1])
    pad      = max(32, border_px * 4)
    h, w     = gt_np.shape[:2]
    y0_z = max(0, cy - pad);  y1_z = min(h, cy + pad)
    x0_z = max(0, cx - pad);  x1_z = min(w, cx + pad)

    panels = [('GT', gt_np), ('Tiled (no GD)', composite)]
    if refined is not None:
        panels.append(('After GD', refined))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    fig.suptitle(f'Boundary zoom (centre y={cy} x={cx})', fontsize=11)

    for ax, (title, img) in zip(axes, panels):
        crop = img[y0_z:y1_z, x0_z:x1_z]
        crop_grid = tile_grid_overlay(crop,
            [(max(0, y0 - y0_z), max(0, x0 - x0_z),
              min(y1_z - y0_z, y1 - y0_z), min(x1_z - x0_z, x1 - x0_z))
             for (y0, x0, y1, x1) in bboxes
             if y0 < y1_z and y1 > y0_z and x0 < x1_z and x1 > x0_z])
        ax.imshow(np.clip(crop_grid, 0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
