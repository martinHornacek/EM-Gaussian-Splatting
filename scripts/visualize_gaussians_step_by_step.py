"""
visualize_gaussians_step_by_step.py
====================================
Shows exactly WHAT the hybrid residual method does, Gaussian by Gaussian.

For each iteration step the script produces a 4-panel figure:

  Panel A  GT + all Gaussian ellipses so far
           * initial-EM Gaussians  ── white, dashed outline, semi-transparent
           * previously-added (+)  ── cyan  solid
           * previously-added (-)  ── magenta  solid
           * NEW this step (+)     ── bright yellow, thick border + star marker
           * NEW this step (-)     ── bright red,    thick border + star marker

  Panel B  Current image approximation (loaded from saved frame PNGs)

  Panel C  Signed residual  I_gt − I_current
           red  = under-exposed (positive residual → where + Gaussians go)
           blue = over-exposed  (negative residual → where − Gaussians go)

  Panel D  PSNR progress curve with the current iteration highlighted

These frames are assembled into an MP4 so you can watch the algorithm work.

A second "zoom" video zooms into a region of interest (default: bounding box
of the first 8 Gaussians, roughly the most active area at the start).

Usage
-----
  python scripts/visualize_gaussians_step_by_step.py
  python scripts/visualize_gaussians_step_by_step.py --run-dir outputs/hybrid_residual_20260429_195213
  python scripts/visualize_gaussians_step_by_step.py --image kodim03 --n-sigma 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def latest_run_dir(outputs: Path) -> Path:
    dirs = sorted(
        [d for d in outputs.iterdir()
         if d.is_dir() and d.name.startswith("hybrid_residual_2")
         and (d / "results.csv").exists()],
        key=lambda d: d.name,
    )
    if not dirs:
        raise FileNotFoundError(f"No run directories found in {outputs}")
    return dirs[-1]


def load_run(run_dir: Path, stem: str):
    """Load all artefacts for one image from a run directory."""
    log  = pd.read_csv(run_dir / f"{stem}_gaussian_log.csv")
    itr  = pd.read_csv(run_dir / f"{stem}_iterations.csv")
    frames_dir = run_dir / f"{stem}_frames"
    frame_files = sorted(frames_dir.glob("frame_*.png")) if frames_dir.exists() else []
    return log, itr, frame_files


def draw_gaussian_ellipse(ax, row, color, lw, alpha, zorder, marker=None, n_sigma=2):
    """Draw one n_sigma-ellipse for a Gaussian row from the log CSV."""
    e = Ellipse(
        xy=(row['x'], row['y']),
        width=2 * n_sigma * row['scale_x'],
        height=2 * n_sigma * row['scale_y'],
        angle=row['rotation_deg'],
        edgecolor=color,
        facecolor='none',
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(e)
    if marker:
        ax.plot(row['x'], row['y'], marker, color=color,
                markersize=6, zorder=zorder + 1, markeredgewidth=0.8,
                markeredgecolor='white')


def make_step_frame(
    gt_np, frame_img, res_np, log_df, itr_df,
    step, n_sigma, output_path
):
    """
    Render the 4-panel figure for one algorithm step.

    ``step`` is the iteration index (0 = initial EM, 1+ = residual iters).
    """
    h, w = gt_np.shape[:2]
    current_log = log_df[log_df['iteration'] <= step]
    new_log     = log_df[log_df['iteration'] == step]

    # Current metrics
    row_m = itr_df[itr_df['iteration'] == step]
    if row_m.empty:
        # step 0 is the init state — use first iter row as placeholder
        psnr_now = itr_df['psnr'].iloc[0] if len(itr_df) else float('nan')
        n_g_now  = current_log.shape[0]
    else:
        psnr_now = float(row_m['psnr'].iloc[0])
        n_g_now  = int(row_m['n_gaussians'].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#0e0e1a')

    title_color = '#e0e0ff'
    for ax in axes.ravel():
        ax.set_facecolor('#0e0e1a')
        ax.tick_params(colors=title_color)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    step_label = "Initial EM" if step == 0 else f"Iter {step}"
    fig.suptitle(
        f"Hybrid Residual Refinement  ·  {step_label}  "
        f"·  Gaussians: {n_g_now}  ·  PSNR: {psnr_now:.2f} dB",
        color=title_color, fontsize=13, fontweight='bold',
    )

    # ── Panel A: GT + Gaussian ellipses ──────────────────────────────────────
    ax_a = axes[0, 0]
    ax_a.imshow(np.clip(gt_np, 0, 1))
    ax_a.set_title(
        'Ground truth  +  Gaussian coverage\n'
        'white-dash=init  cyan=+add  magenta=−sub  '
        'yellow★=new + this step  red★=new − this step',
        fontsize=7.5, color=title_color,
    )
    ax_a.axis('off')

    init_log = current_log[current_log['iteration'] == 0]
    prev_pos  = current_log[(current_log['iteration'] >  0) &
                             (current_log['iteration'] <  step) &
                             (current_log['amplitude'] >= 0)]
    prev_neg  = current_log[(current_log['iteration'] >  0) &
                             (current_log['iteration'] <  step) &
                             (current_log['amplitude'] <  0)]
    new_pos   = new_log[new_log['amplitude'] >= 0]
    new_neg   = new_log[new_log['amplitude'] <  0]

    for _, r in init_log.iterrows():
        draw_gaussian_ellipse(ax_a, r, '#ffffff', lw=0.8, alpha=0.35,
                              zorder=2, n_sigma=n_sigma)
    for _, r in prev_pos.iterrows():
        draw_gaussian_ellipse(ax_a, r, '#00d4ff', lw=1.0, alpha=0.55,
                              zorder=3, marker='o', n_sigma=n_sigma)
    for _, r in prev_neg.iterrows():
        draw_gaussian_ellipse(ax_a, r, '#ff55cc', lw=1.0, alpha=0.55,
                              zorder=3, marker='o', n_sigma=n_sigma)
    for _, r in new_pos.iterrows():
        draw_gaussian_ellipse(ax_a, r, '#ffe600', lw=2.5, alpha=0.95,
                              zorder=5, marker='*', n_sigma=n_sigma)
    for _, r in new_neg.iterrows():
        draw_gaussian_ellipse(ax_a, r, '#ff2200', lw=2.5, alpha=0.95,
                              zorder=5, marker='*', n_sigma=n_sigma)

    ax_a.set_xlim(0, w)
    ax_a.set_ylim(h, 0)

    # ── Panel B: Current approximation ───────────────────────────────────────
    ax_b = axes[0, 1]
    ax_b.imshow(np.clip(frame_img, 0, 1))
    ax_b.set_title('Current image approximation', fontsize=9, color=title_color)
    ax_b.axis('off')

    # ── Panel C: Signed residual ──────────────────────────────────────────────
    ax_c = axes[1, 0]
    res_2d  = res_np.mean(axis=2)
    vmax    = max(float(np.abs(res_2d).max()), 1e-6)
    im      = ax_c.imshow(res_2d, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax_c, fraction=0.03, pad=0.02,
                 label='I_gt − I_approx', shrink=0.9)
    ax_c.set_title(
        'Signed residual  (red=under-exposed → + Gaussian here next)\n'
        '                 (blue=over-exposed  → − Gaussian here next)',
        fontsize=8, color=title_color,
    )
    ax_c.tick_params(bottom=False, left=False,
                     labelbottom=False, labelleft=False)

    # ── Panel D: PSNR progress ────────────────────────────────────────────────
    ax_d = axes[1, 1]
    ax_d.set_facecolor('#0e0e1a')
    xs = itr_df['n_gaussians'].values
    ys = itr_df['psnr'].values
    ax_d.plot(xs, ys, '-o', color='#f4a261', lw=2, markersize=4, zorder=3)
    ax_d.fill_between(xs, ys, alpha=0.12, color='#f4a261')
    if n_g_now > 0:
        ax_d.axvline(n_g_now, color='#ffe600', lw=1.5, ls='--', alpha=0.7,
                     label=f'← Now ({n_g_now} G, {psnr_now:.2f} dB)')
        closest = itr_df.iloc[(itr_df['n_gaussians'] - n_g_now).abs().argsort()[:1]]
        if not closest.empty:
            ax_d.scatter([n_g_now], [psnr_now],
                         color='#ffe600', s=80, zorder=5)
    ax_d.set_xlabel('Gaussians', color=title_color, fontsize=9)
    ax_d.set_ylabel('PSNR (dB)', color=title_color, fontsize=9)
    ax_d.set_title('PSNR vs Gaussian budget', color=title_color, fontsize=9)
    ax_d.tick_params(colors=title_color)
    ax_d.grid(alpha=0.2, color='#aaaaaa')
    ax_d.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
                framealpha=0.7)
    for spine in ax_d.spines.values():
        spine.set_edgecolor('#333355')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=110, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def make_zoom_frame(
    gt_np, frame_img, res_np, log_df, step, n_sigma, bbox, output_path
):
    """
    Zoomed 3-panel frame (GT | Approx | Residual) clipped to bbox,
    with Gaussian ellipses shown at native pixel scale.
    bbox = (x0, y0, x1, y1)  in pixel coordinates
    """
    x0, y0, x1, y1 = bbox
    h, w = gt_np.shape[:2]

    current_log = log_df[log_df['iteration'] <= step]
    new_log     = log_df[log_df['iteration'] == step]
    new_pos     = new_log[new_log['amplitude'] >= 0]
    new_neg     = new_log[new_log['amplitude'] <  0]

    crops = [
        np.clip(gt_np[y0:y1, x0:x1], 0, 1),
        np.clip(frame_img[y0:y1, x0:x1], 0, 1),
        (gt_np - frame_img).mean(axis=2)[y0:y1, x0:x1],
    ]
    titles = ['Ground truth (zoom)', 'Approximation (zoom)',
              'Residual (zoom)']
    cmaps  = [None, None, 'RdBu_r']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0e0e1a')
    step_label = "Initial EM" if step == 0 else f"Iter {step}"
    fig.suptitle(
        f'{step_label}  ·  zoomed region  ({x0}-{x1}, {y0}-{y1})',
        color='#e0e0ff', fontsize=11, fontweight='bold',
    )

    for ax, crop, title, cmap in zip(axes, crops, titles, cmaps):
        ax.set_facecolor('#0e0e1a')
        ax.axis('off')
        if cmap:
            vmax = max(float(np.abs(crop).max()), 1e-6)
            im = ax.imshow(crop, cmap=cmap, vmin=-vmax, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.01, shrink=0.8)
        else:
            ax.imshow(crop)
        ax.set_title(title, color='#e0e0ff', fontsize=9)

    # Draw Gaussians on first two panels, offset to crop coords
    for ax in axes[:2]:
        for _, r in log_df[(log_df['iteration'] == 0)].iterrows():
            rx, ry = r['x'] - x0, r['y'] - y0
            r2 = r.copy(); r2['x'] = rx; r2['y'] = ry
            draw_gaussian_ellipse(ax, r2, '#ffffff', lw=0.8, alpha=0.4,
                                  zorder=2, n_sigma=n_sigma)
        for _, r in current_log[(current_log['iteration'] > 0) &
                                (current_log['iteration'] < step) &
                                (current_log['amplitude'] >= 0)].iterrows():
            rx, ry = r['x'] - x0, r['y'] - y0
            r2 = r.copy(); r2['x'] = rx; r2['y'] = ry
            draw_gaussian_ellipse(ax, r2, '#00d4ff', lw=1.0, alpha=0.55,
                                  zorder=3, n_sigma=n_sigma)
        for _, r in current_log[(current_log['iteration'] > 0) &
                                (current_log['iteration'] < step) &
                                (current_log['amplitude'] < 0)].iterrows():
            rx, ry = r['x'] - x0, r['y'] - y0
            r2 = r.copy(); r2['x'] = rx; r2['y'] = ry
            draw_gaussian_ellipse(ax, r2, '#ff55cc', lw=1.0, alpha=0.55,
                                  zorder=3, n_sigma=n_sigma)
        for _, r in new_pos.iterrows():
            rx, ry = r['x'] - x0, r['y'] - y0
            r2 = r.copy(); r2['x'] = rx; r2['y'] = ry
            draw_gaussian_ellipse(ax, r2, '#ffe600', lw=2.5, alpha=0.95,
                                  zorder=5, marker='*', n_sigma=n_sigma)
        for _, r in new_neg.iterrows():
            rx, ry = r['x'] - x0, r['y'] - y0
            r2 = r.copy(); r2['x'] = rx; r2['y'] = ry
            draw_gaussian_ellipse(ax, r2, '#ff2200', lw=2.5, alpha=0.95,
                                  zorder=5, marker='*', n_sigma=n_sigma)
        ax.set_xlim(0, x1 - x0)
        ax.set_ylim(y1 - y0, 0)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=110, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def assemble_mp4(frame_paths, output_path, fps=1.5):
    try:
        import imageio
    except ImportError:
        print("  MP4 skipped: pip install imageio imageio-ffmpeg")
        return
    try:
        target_h, target_w = None, None
        for p in frame_paths:
            ref = np.array(Image.open(p).convert('RGB'))
            rh, rw = ref.shape[:2]
            target_h = rh if rh % 2 == 0 else rh - 1
            target_w = rw if rw % 2 == 0 else rw - 1
            break
        writer = imageio.get_writer(
            str(output_path), fps=fps, codec='libx264',
            pixelformat='yuv420p', quality=8, macro_block_size=1,
        )
        for p in frame_paths:
            fr = np.array(
                Image.open(p).convert('RGB').resize(
                    (target_w, target_h), Image.LANCZOS))
            writer.append_data(fr)
        writer.close()
        print(f"  MP4  -> {output_path.name}  ({len(frame_paths)} frames, {fps} fps)")
    except Exception as exc:
        print(f"  MP4 failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise hybrid residual Gaussians step-by-step with ellipses")
    parser.add_argument('--run-dir', default=None,
        help="Output directory of a run (default: latest in ./outputs)")
    parser.add_argument('--image', default='kodim03',
        help="Image stem, e.g. kodim03 (default)")
    parser.add_argument('--n-sigma', type=float, default=2.0,
        help="Ellipse radius in standard deviations (default: 2)")
    parser.add_argument('--fps', type=float, default=1.5,
        help="Video frame rate (default: 1.5)")
    parser.add_argument('--zoom-x0', type=int, default=None)
    parser.add_argument('--zoom-y0', type=int, default=None)
    parser.add_argument('--zoom-x1', type=int, default=None)
    parser.add_argument('--zoom-y1', type=int, default=None)
    args = parser.parse_args()

    outputs = ROOT / "outputs"
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(outputs)
    stem    = args.image
    print(f"Run dir : {run_dir}")
    print(f"Image   : {stem}")

    # ── Load artefacts ────────────────────────────────────────────────────────
    log_df, itr_df, frame_files = load_run(run_dir, stem)
    gt_np = (np.array(
        Image.open(ROOT / 'kodak' / f'{stem}.png').convert('RGB'))
        .astype(np.float32) / 255.0)
    h, w = gt_np.shape[:2]

    max_step = int(log_df['iteration'].max())
    steps    = list(range(0, max_step + 1))

    # ── Output directory ──────────────────────────────────────────────────────
    vis_dir = run_dir / f'{stem}_gaussian_vis'
    vis_dir.mkdir(exist_ok=True)
    zoom_dir = run_dir / f'{stem}_gaussian_vis_zoom'
    zoom_dir.mkdir(exist_ok=True)

    # ── Determine zoom region ─────────────────────────────────────────────────
    # Use user-supplied bbox if given, else auto-detect: pick the 80th-percentile
    # x/y bounding box of the Gaussians that appear earliest (small iteration)
    # — this tends to find the most "interesting" region of the image.
    if args.zoom_x0 is not None:
        bbox = (args.zoom_x0, args.zoom_y0, args.zoom_x1, args.zoom_y1)
    else:
        # Early Gaussians show where the main structure is
        early = log_df[log_df['iteration'] <= max(1, max_step // 4)]
        cx = np.clip(int(early['x'].median()), 50, w - 51)
        cy = np.clip(int(early['y'].median()), 50, h - 51)
        # 40% width/height crop centred on that point
        half_x = int(w * 0.22)
        half_y = int(h * 0.22)
        bbox = (
            max(0,     cx - half_x),
            max(0,     cy - half_y),
            min(w - 1, cx + half_x),
            min(h - 1, cy + half_y),
        )
    print(f"Zoom box: x={bbox[0]}-{bbox[2]}  y={bbox[1]}-{bbox[3]}")

    # ── Print Gaussian summary ────────────────────────────────────────────────
    print("\n--- Gaussian summary ---")
    print(f"{'Step':>5}  {'N_add':>5}  {'New+ (x,y)':>40}  {'New- (x,y)':>40}")
    for step in steps:
        new = log_df[log_df['iteration'] == step]
        n = len(new)
        pos_str = "  ".join(
            f"({r.x:.0f},{r.y:.0f})" for _, r in new[new.amplitude >= 0].iterrows()
        )
        neg_str = "  ".join(
            f"({r.x:.0f},{r.y:.0f})" for _, r in new[new.amplitude < 0].iterrows()
        )
        step_lbl = "init" if step == 0 else f"iter{step}"
        print(f"{step_lbl:>5}  {n:>5}  {pos_str[:40]:>40}  {neg_str[:40]:>40}")

    # ── Generate frames ───────────────────────────────────────────────────────
    overview_frames = []
    zoom_frames     = []

    for step in steps:
        frame_idx = min(step, len(frame_files) - 1)
        frame_img = (np.array(
            Image.open(frame_files[frame_idx]).convert('RGB'))
            .astype(np.float32) / 255.0) if frame_files else gt_np.copy()

        # The frame PNG contains more than just the render (it also has GT and
        # residual panels). Reload just the raw render if possible; otherwise
        # use the full frame.  The frames are saved as (H, ~3W) 3-panel images.
        # We crop the middle third.
        fh, fw = frame_img.shape[:2]
        if fw > fh * 1.5:
            # 3-panel image: GT | Render | Residual — take the middle third
            third = fw // 3
            frame_img = frame_img[:, third : 2 * third, :]

        # Resize to match gt_np if necessary
        if frame_img.shape[:2] != (h, w):
            frame_img = np.array(
                Image.fromarray((frame_img * 255).astype(np.uint8))
                .resize((w, h), Image.LANCZOS)
            ).astype(np.float32) / 255.0

        res_np  = gt_np - frame_img

        # ---- metrics for this step (init state uses iter 1 as proxy) --------
        step_itr_df = itr_df.copy()
        # add a row 0 = initial EM
        if 0 not in itr_df['iteration'].values:
            init_psnr = float(itr_df['psnr'].iloc[0]) - 2.0  # approx
            init_ng   = int(log_df[log_df['iteration'] == 0].shape[0])
            step_itr_df = pd.concat([
                pd.DataFrame([{'iteration': 0, 'psnr': init_psnr, 'n_gaussians': init_ng}]),
                itr_df
            ], ignore_index=True)

        out_ov  = vis_dir  / f'frame_{step:04d}.png'
        out_zm  = zoom_dir / f'frame_{step:04d}.png'

        print(f"  Rendering step {step} ... ", end='', flush=True)
        make_step_frame(gt_np, frame_img, res_np,
                        log_df, step_itr_df,
                        step, args.n_sigma, out_ov)
        make_zoom_frame(gt_np, frame_img, res_np,
                        log_df, step, args.n_sigma, bbox, out_zm)
        print("done")

        overview_frames.append(out_ov)
        zoom_frames.append(out_zm)

    # ── Assemble videos ───────────────────────────────────────────────────────
    assemble_mp4(overview_frames, run_dir / f'{stem}_gaussians_overview.mp4', fps=args.fps)
    assemble_mp4(zoom_frames,     run_dir / f'{stem}_gaussians_zoom.mp4',     fps=args.fps)

    # also print top-5 largest Gaussians (most area)
    log_df['area'] = log_df['scale_x'] * log_df['scale_y']
    top5 = log_df.nlargest(5, 'area')[
        ['iteration', 'x', 'y', 'scale_x', 'scale_y', 'amplitude', 'r', 'g', 'b', 'area']
    ]
    print(f"\nTop-5 largest Gaussians (by area = scale_x * scale_y):\n{top5.to_string()}")

    # Top-5 closest to a region of interest (bbox centre)
    cx_zoom = (bbox[0] + bbox[2]) // 2
    cy_zoom = (bbox[1] + bbox[3]) // 2
    log_df['dist_to_zoom'] = np.sqrt(
        (log_df['x'] - cx_zoom) ** 2 + (log_df['y'] - cy_zoom) ** 2)
    top5_zoom = log_df.nsmallest(5, 'dist_to_zoom')[
        ['iteration', 'x', 'y', 'scale_x', 'scale_y', 'amplitude', 'dist_to_zoom']
    ]
    print(f"\nTop-5 Gaussians nearest zoom-centre ({cx_zoom},{cy_zoom}):\n{top5_zoom.to_string()}")

    print(f"\nAll frames saved to:\n  {vis_dir}\n  {zoom_dir}")


if __name__ == '__main__':
    main()
