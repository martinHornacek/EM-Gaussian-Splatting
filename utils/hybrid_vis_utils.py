"""
hybrid_vis_utils.py - Visualization helpers for hybrid residual refinement.

Per-iteration frame
-------------------
save_render_frame(frame_dir, iteration, gt, render, residual_signed, metrics, n, mae)
    3-panel PNG  Ground truth | Current render | Signed residual (RdBu_r)
    These frames are assembled into the evolution video.

Post-run diagnostic summary plots (one figure each, focused and interpretable)
------------------------------------------------------------------------------
save_diag_residuals(output_path, residual_snapshots, iter_indices)
    Grid of |I_gt - I_render| heatmaps, one cell per recorded iteration.

save_diag_gaussians(output_path, all_means, all_covs, all_amplitudes, image_size)
    Scatter of Gaussian centres: blue = additive, red = subtractive.
    Marker size scales with the geometric mean of the Gaussian spatial std devs.

save_diag_metrics(output_path, iter_records, baselines=None)
    Four subplots: RMSE, PSNR, SSIM, Loss vs Gaussian budget (x-axis).
    Optional ``baselines`` list overlays Init EM / Pure EM reference lines and
    markers so all methods are compared on the same budget axis.

3-panel comparison helper
--------------------------
save_comparison_3panel(output_path, gt, render, label, metrics)
    Ground truth | Render | Signed residual.

Animation assemblers
--------------------
assemble_mp4(frame_paths, output_path, fps)   requires imageio + imageio-ffmpeg
assemble_gif(frame_paths, output_path, fps)   Pillow only, always available
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image


def save_render_frame(
    frame_dir,
    iteration,
    gt_np,
    I_current,
    I_res_signed,
    metrics,
    n_total_gaussians,
    loss_scalar,
):
    """
    Save a 3-panel PNG for one algorithm step.

    Panels: Ground truth | Current render | Signed residual (RdBu_r)

    Returns the Path of the saved PNG.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"Iter {iteration}  |  Gaussians: {n_total_gaussians}  |  "
        f"PSNR: {metrics.get('psnr', 0.0):.2f} dB  |  "
        f"SSIM: {metrics.get('ssim', 0.0):.4f}  |  "
        f"RMSE: {metrics.get('rmse', 0.0):.4f}  |  "
        f"Loss: {metrics.get('loss', loss_scalar):.4f}",
        fontsize=9,
    )

    axes[0].imshow(np.clip(gt_np, 0, 1))
    axes[0].set_title('Ground Truth', fontsize=9)
    axes[0].axis('off')

    axes[1].imshow(np.clip(I_current, 0, 1))
    axes[1].set_title('Current Render', fontsize=9)
    axes[1].axis('off')

    res_2d = I_res_signed.mean(axis=2)
    vmax = max(float(np.abs(res_2d).max()), 1e-6)
    im = axes[2].imshow(res_2d, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02)
    axes[2].set_title(
        'Residual  I_gt - I_render\n(red = under-exposed, blue = over-exposed)',
        fontsize=8,
    )
    axes[2].axis('off')

    plt.tight_layout()
    out = Path(frame_dir) / f'frame_{iteration:04d}.png'
    plt.savefig(out, dpi=90, bbox_inches='tight')
    plt.close(fig)
    return out


def save_diag_residuals(output_path, residual_snapshots, iter_indices, max_cols=5):
    """
    Save a grid of mean-absolute-residual heatmaps, one cell per recorded
    iteration.  The colour scale is shared across all cells.

    Parameters
    ----------
    output_path        : Path
    residual_snapshots : list of (H,W) float32 arrays  mean |residual|
    iter_indices       : list of int   iteration label for each snapshot
    max_cols           : int   maximum columns in the grid
    """
    n = len(residual_snapshots)
    if n == 0:
        return

    n_cols = min(n, max_cols)
    n_rows = (n + n_cols - 1) // n_cols
    vmax = max(float(max(s.max() for s in residual_snapshots)), 1e-6)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )
    im_last = None
    for idx, (snap, it_idx) in enumerate(zip(residual_snapshots, iter_indices)):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        im_last = ax.imshow(snap, cmap='inferno', vmin=0, vmax=vmax)
        ax.set_title(f'Iter {it_idx}\nMAE={snap.mean():.4f}', fontsize=8)
        ax.axis('off')

    if im_last is not None:
        fig.colorbar(
            im_last, ax=axes, fraction=0.02, pad=0.02,
            label='mean |I_gt - I_render|',
        )

    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis('off')

    fig.suptitle(
        'Residual Heatmap Evolution  |I_gt - I_render|',
        fontsize=11, fontweight='bold',
    )
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return output_path


def save_diag_gaussians(output_path, all_means, all_covs, all_amplitudes, image_size):
    """
    Scatter of all Gaussian centres coloured by amplitude sign.

    Blue  = positive (additive) Gaussians
    Red   = negative (subtractive) Gaussians

    Marker area scales with sqrt(det(cov))^0.5 so larger blobs appear bigger.
    """
    h, w = image_size
    means = np.array(all_means)
    covs  = np.array(all_covs)
    amps  = np.array(all_amplitudes)

    scales = []
    for cov in covs:
        try:
            vals = np.linalg.eigvalsh(cov)
            vals = np.maximum(vals, 1e-6)
            scales.append(float(np.sqrt(np.sqrt(vals[0] * vals[1]))))
        except np.linalg.LinAlgError:
            scales.append(1.0)
    scales = np.array(scales, dtype=np.float32)
    s_max = scales.max() + 1e-8
    marker_sizes = np.clip(scales / s_max * 200 + 10, 10, 250)

    pos_mask = amps >= 0
    neg_mask = ~pos_mask

    fig, ax = plt.subplots(figsize=(max(5, 5 * w / max(h, 1)), 5))
    ax.set_facecolor('#12121f')

    if pos_mask.any():
        ax.scatter(
            means[pos_mask, 0], means[pos_mask, 1],
            s=marker_sizes[pos_mask],
            c='#4cc9f0', alpha=0.65, linewidths=0.3, edgecolors='white',
            label=f'Additive (+)  {pos_mask.sum()}',
        )
    if neg_mask.any():
        ax.scatter(
            means[neg_mask, 0], means[neg_mask, 1],
            s=marker_sizes[neg_mask],
            c='#f72585', alpha=0.65, linewidths=0.3, edgecolors='white',
            label=f'Subtractive (-)  {neg_mask.sum()}',
        )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel('x  (pixels)', color='white')
    ax.set_ylabel('y  (pixels)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.set_title(
        f'Gaussian Distribution  ({len(means)} total)\n'
        'Marker size proportional to geometric mean std dev',
        fontsize=10, fontweight='bold', color='white',
    )
    ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')
    ax.grid(alpha=0.12, color='white')
    fig.patch.set_facecolor('#12121f')

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def save_diag_metrics(output_path, iter_records, baselines=None):
    """
    Four subplots comparing method quality as a function of Gaussian budget.

    The x-axis is the cumulative number of Gaussians used (from
    ``iter_records['n_gaussians']``), so every method is compared on the same
    budget axis — the fairest apples-to-apples view.

    Hybrid residual refinement is drawn as a continuous curve.  Each entry in
    ``baselines`` is drawn as a single labelled marker plus a horizontal dashed
    reference line at its quality level, allowing side-by-side comparison even
    though those methods don't have an iteration trace.

    Parameters
    ----------
    output_path  : Path
    iter_records : list of dicts with keys including ``n_gaussians``,
                   ``rmse``, ``psnr``, ``ssim``, ``loss``.
    baselines    : list of dicts, optional.  Each dict must have::

                       {
                         'label':       str,          # legend entry
                         'n_gaussians': int,          # x-position
                         'rmse':        float,
                         'psnr':        float,
                         'ssim':        float,
                         'loss':        float,
                         'style':       dict,         # matplotlib kwargs (color, marker, …)
                       }

                   Typical entries: initial-EM state, pure-EM result.
    """
    if not iter_records:
        return

    import pandas as pd
    df = pd.DataFrame(iter_records)

    # Use Gaussians as x-axis when available for apples-to-apples comparison.
    x_col   = 'n_gaussians' if 'n_gaussians' in df.columns else 'iteration'
    x_label = 'Gaussians' if x_col == 'n_gaussians' else 'Iteration'
    x_vals  = df[x_col].values

    panel_cfg = [
        ('rmse', 'RMSE',      '#4895ef', 'lower is better'),
        ('psnr', 'PSNR (dB)', '#f4a261', 'higher is better'),
        ('ssim', 'SSIM',      '#2a9d8f', 'higher is better'),
        ('loss', 'Loss',      '#e76f51', 'lower is better'),
    ]
    avail = [(col, lbl, col_c, note)
             for col, lbl, col_c, note in panel_cfg if col in df.columns]
    n_panels = len(avail)
    if n_panels == 0:
        return

    n_cols  = min(n_panels, 2)
    n_rows  = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6.5 * n_cols, 4.5 * n_rows), squeeze=False
    )
    axes_flat = axes.ravel()

    for panel_idx, (col, label, colour, note) in enumerate(avail):
        ax   = axes_flat[panel_idx]
        vals = df[col].values

        # ---- Hybrid curve -------------------------------------------------- #
        ax.plot(
            x_vals, vals,
            '-o', color=colour, markersize=4, lw=2.0,
            label='Hybrid residual', zorder=3,
        )
        ax.fill_between(x_vals, vals, alpha=0.10, color=colour)

        # Annotate start and end values
        ax.annotate(
            f'{vals[0]:.4f}', xy=(x_vals[0], vals[0]),
            xytext=(5, 5), textcoords='offset points',
            fontsize=7, color=colour,
        )
        ax.annotate(
            f'{vals[-1]:.4f}', xy=(x_vals[-1], vals[-1]),
            xytext=(5, -11), textcoords='offset points',
            fontsize=7, color=colour,
        )

        # ---- Baseline markers + reference lines ---------------------------- #
        for bl in (baselines or []):
            if col not in bl:
                continue
            sty   = bl.get('style', {})
            bval  = bl[col]
            bx    = bl.get('n_gaussians', x_vals[-1])
            bcol  = sty.get('color', '#888')
            bmark = sty.get('marker', 'D')
            # Horizontal dashed reference line
            ax.axhline(
                bval, color=bcol, lw=1.2, ls='--', alpha=0.55, zorder=1,
            )
            # Single marker at budget position
            ax.scatter(
                [bx], [bval],
                marker=bmark, s=70, color=bcol, zorder=5,
                edgecolors='white', linewidths=0.6,
                label=f"{bl['label']}  ({bval:.4f})",
            )

        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f'{label}  ({note})', fontsize=9, fontweight='bold')
        ax.legend(fontsize=7.5, loc='best')
        ax.grid(alpha=0.3)

    for idx in range(len(avail), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        'Metrics vs Gaussian Budget — Method Comparison',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return output_path


def save_comparison_3panel(output_path, gt_np, render_np, label, metrics):
    """
    Save a focused 3-panel comparison figure.

    Panels: Ground Truth | Render | Signed residual I_gt - I_render (RdBu_r)
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"{label}  |  PSNR: {metrics.get('psnr', 0.0):.2f} dB  |  "
        f"SSIM: {metrics.get('ssim', 0.0):.4f}  |  RMSE: {metrics.get('rmse', 0.0):.4f}",
        fontsize=10, fontweight='bold',
    )

    axes[0].imshow(np.clip(gt_np, 0, 1))
    axes[0].set_title('Ground Truth', fontsize=9)
    axes[0].axis('off')

    axes[1].imshow(np.clip(render_np, 0, 1))
    axes[1].set_title('Render', fontsize=9)
    axes[1].axis('off')

    res = (gt_np - render_np).mean(axis=2)
    vmax = max(float(np.abs(res).max()), 1e-6)
    im = axes[2].imshow(res, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02)
    axes[2].set_title(
        'Residual  I_gt - I_render\n(red = under-exposed, blue = over-exposed)',
        fontsize=8,
    )
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def assemble_mp4(frame_paths, output_path, fps=2):
    """
    Assemble PNG frames into an MP4 using imageio + imageio-ffmpeg.

    Falls back gracefully with an install hint if imageio is unavailable.
    """
    if not frame_paths:
        print(f"  No frames to assemble for {Path(output_path).name}")
        return
    try:
        import imageio
    except ImportError:
        print(
            f"  MP4 skipped ({Path(output_path).name}): imageio not installed.  "
            "Run:  pip install imageio imageio-ffmpeg"
        )
        return

    try:
        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            quality=8,
            macro_block_size=1,
        )
        # Determine the target size from the first valid frame (even H and W
        # required by yuv420p).
        target_h, target_w = None, None
        for p in frame_paths:
            ref = np.array(Image.open(p).convert('RGB'))
            rh, rw = ref.shape[:2]
            target_h = rh if rh % 2 == 0 else rh - 1
            target_w = rw if rw % 2 == 0 else rw - 1
            break

        for p in frame_paths:
            frame = np.array(Image.open(p).convert('RGB').resize(
                (target_w, target_h), Image.LANCZOS,
            ))
            writer.append_data(frame)
        writer.close()
        print(
            f"  MP4 saved  ({len(frame_paths)} frames, {fps} fps): "
            f"{Path(output_path).name}"
        )
    except Exception as exc:
        print(f"  MP4 export failed for {Path(output_path).name}: {exc}")
        print("  Hint: pip install imageio imageio-ffmpeg")


def assemble_gif(frame_paths, output_path, fps=2):
    """
    Assemble PNG frames into an animated GIF using Pillow.

    GIFs have lower quality than MP4 but require no extra dependencies.
    Use as a fallback when imageio-ffmpeg is unavailable.
    """
    if not frame_paths:
        print(f"  No frames to assemble for {Path(output_path).name}")
        return

    frames = []
    for p in frame_paths:
        try:
            frames.append(Image.open(p).convert('RGBA'))
        except Exception as exc:
            print(f"  Warning: skipping frame {Path(p).name} ({exc})")

    if not frames:
        return

    duration_ms = max(20, int(1000 / fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    print(
        f"  GIF saved  ({len(frames)} frames, {fps} fps): "
        f"{Path(output_path).name}"
    )