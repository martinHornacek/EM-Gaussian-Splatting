"""Plot 2D-GS metric curves (vs epoch) and overlay EM points; compare runtimes.

Usage:
    python scripts/plot_gs_vs_em_overlay.py --gs-results-dir results/gs_full_kodak_2500_gaussians \
        --em-results-csv results/em_full_kodak_2500_gaussians/results.csv \
        --outdir results/comparison/gs_em_overlays

For each image present in both GS history files and EM results, the script:
 - loads the GS per-epoch history (kodimXX_history.csv)
 - plots PSNR vs epoch (and SSIM vs epoch when available) for GS
 - overlays EM's final metric values (horizontal lines at EM PSNR/SSIM) and annotates EM time vs GS time to reach the closest PSNR epoch (approx)

Saves per-image PNGs and a combined summary figure and CSV with time comparisons.
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_em_results(path):
    return pd.read_csv(path)


def find_history_path(gs_dir, image_name):
    # image_name like kodim01.png -> history file kodim01_history.csv
    base = os.path.splitext(image_name)[0]
    hist = os.path.join(gs_dir, f"{base}_history.csv")
    if os.path.exists(hist):
        return hist
    return None


def compute_epoch_metrics_and_time(history_df, gs_meta_row):
    # Only compute per-epoch PSNR/SSIM and an approximate per-epoch time.
    total_time = float(gs_meta_row['total_time'])

    epochs = history_df['epoch'].values
    psnr = history_df['psnr'].values
    ssim = history_df['ssim'].values if 'ssim' in history_df.columns else None

    max_epoch = epochs.max()
    # approximate cumulative time proportional to epoch
    epoch_time = (epochs / float(max_epoch)) * total_time

    return epochs, psnr, ssim, epoch_time


def plot_per_image(image, history_df, gs_row, em_row, outdir):
    epochs, psnr, ssim, gs_time = compute_epoch_metrics_and_time(history_df, gs_row)

    psnr_em = float(em_row['psnr'])
    time_em = float(em_row['total_time'])

    # Find GS epoch closest in PSNR
    idx_psnr = np.argmin(np.abs(psnr - psnr_em))
    gs_time_psnr = gs_time[idx_psnr]

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PSNR vs epoch
    axes[0].plot(epochs, psnr, marker='o', label='2D-GS')
    axes[0].axhline(psnr_em, color='red', linestyle='--', label=f'EM PSNR {psnr_em:.2f}')
    axes[0].scatter([epochs[idx_psnr]], [psnr[idx_psnr]], color='orange', zorder=5, label='GS (match PSNR)')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title(f'{image} — PSNR vs epoch')
    axes[0].legend()

    # SSIM vs epoch if available
    if ssim is not None:
        axes[1].plot(epochs, ssim, marker='o', label='2D-GS')
        ssim_em = em_row.get('ssim')
        if ssim_em is not None and not pd.isna(ssim_em):
            axes[1].axhline(ssim_em, color='red', linestyle='--', label=f'EM SSIM {ssim_em:.3f}')
        axes[1].set_xlabel('epoch')
        axes[1].set_ylabel('SSIM')
        axes[1].set_title(f'{image} — SSIM vs epoch')
        axes[1].legend()
    else:
        # If SSIM not in history, plot nothing and hide
        axes[1].text(0.5, 0.5, 'SSIM history not available', ha='center', va='center')
        axes[1].axis('off')

    # Annotate time comparisons on PSNR axis
    text = (
        f"EM: psnr={psnr_em:.2f}, time={time_em:.1f}s\n"
        f"GS match-PSNR: epoch={int(epochs[idx_psnr])}, time={gs_time_psnr:.1f}s"
    )
    axes[0].text(0.02, 0.02, text, transform=axes[0].transAxes, fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.8))

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{os.path.splitext(image)[0]}_gs_em_overlay.png")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

    return dict(image=image, psnr_em=psnr_em, time_em=time_em,
                gs_time_psnr=float(gs_time_psnr), epoch_psnr=int(epochs[idx_psnr]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gs-results-dir', required=True, help='Directory with GS results and history CSVs')
    parser.add_argument('--em-results-csv', required=True, help='EM results CSV path')
    parser.add_argument('--outdir', default='results/comparison/gs_em_overlays', help='Output directory for plots and CSV')
    parser.add_argument('--images', default=None, help='Comma-separated list of images to process (e.g. kodim01.png,kodim02.png)')
    args = parser.parse_args()

    em_df = load_em_results(args.em_results_csv)
    gs_results_csv = os.path.join(args.gs_results_dir, 'results.csv')
    if not os.path.exists(gs_results_csv):
        raise FileNotFoundError(f'GS results.csv not found in {args.gs_results_dir}')
    gs_df = pd.read_csv(gs_results_csv)

    # find common images
    em_images = set(em_df['image'].values)
    gs_images = set(gs_df['image'].values)
    common = sorted(list(em_images & gs_images))
    if args.images:
        requested = [x.strip() for x in args.images.split(',') if x.strip()]
        common = [x for x in common if x in requested]

    if not common:
        print('No common images found between EM and GS results.')
        return

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for img in common:
        hist_path = find_history_path(args.gs_results_dir, img)
        if hist_path is None:
            print(f'History for {img} not found, skipping')
            continue
        history_df = pd.read_csv(hist_path)
        gs_row = gs_df[gs_df['image'] == img].iloc[0]
        em_row = em_df[em_df['image'] == img].iloc[0]
        row = plot_per_image(img, history_df, gs_row, em_row, args.outdir)
        rows.append(row)
        print(f'Plotted {img} -> {row}')

    out_csv = os.path.join(args.outdir, 'gs_em_time_comparison.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f'Saved summary CSV to {out_csv}')

    # Summary bar plot for time ratios (EM time vs GS time to match PSNR)
    df = pd.DataFrame(rows)
    if not df.empty:
        fig, ax = plt.subplots(figsize=(8, max(4, len(df)*0.3)))
        y = np.arange(len(df))
        width = 0.35
        ax.barh(y - width/2, df['time_em'], height=width, label='EM time (s)')
        ax.barh(y + width/2, df['gs_time_psnr'], height=width, label='GS time to match PSNR (s)')
        ax.set_yticks(y)
        ax.set_yticklabels(df['image'])
        ax.set_xlabel('time (s)')
        ax.set_title('Time comparison: EM vs GS (to match PSNR)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, 'time_comparison_em_vs_gs_match_psnr.png'))
        plt.close(fig)
        print('Saved summary time comparison plot')


if __name__ == '__main__':
    main()
