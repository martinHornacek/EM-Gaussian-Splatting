"""
analyze_hybrid_kodak.py — Evaluate hybrid residual refinement on the full Kodak set.

Loads results from one or two hybrid runs (minibatch and/or fullbatch), computes
aggregate statistics, prints a LaTeX-ready table, and saves summary plots.

Usage
-----
# Single run
python scripts/analyze_hybrid_kodak.py \
    --minibatch outputs/hybrid_residual_<ts_mb>/results.csv

# Both variants side by side
python scripts/analyze_hybrid_kodak.py \
    --minibatch   outputs/hybrid_residual_<ts_mb>/results.csv \
    --fullbatch   outputs/hybrid_residual_<ts_fb>/results.csv \
    --out         results/hybrid_kodak_eval

Outputs
-------
<out>/aggregate_stats.csv          — mean / std / min / max per metric per variant
<out>/per_image_table.csv          — per-image metrics for all variants
<out>/psnr_per_image.png           — bar chart
<out>/ssim_per_image.png
<out>/rmse_per_image.png
<out>/lpips_per_image.png
<out>/metrics_distribution.png     — violin / box plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Literature reference values (PSNR dB, SSIM on Kodak 768×512 full-res)
# These are collected from published papers at various bpp levels that roughly
# correspond to lossless or near-lossless compression of a Gaussian mixture.
# They serve as rough orientation — direct comparison requires matched bpp.
# ---------------------------------------------------------------------------
LITERATURE = {
    # -- Classical codecs (vary by quality setting, values at ~0.5–1.0 bpp) --
    'JPEG (q=90)':      {'psnr': 37.5,  'ssim': 0.948, 'source': 'Kodak benchmark'},
    'JPEG 2000':        {'psnr': 38.8,  'ssim': 0.958, 'source': 'Kodak benchmark'},
    'BPG (HEVC intra)': {'psnr': 40.5,  'ssim': 0.970, 'source': 'Kodak benchmark'},
    'WebP (q=90)':      {'psnr': 38.2,  'ssim': 0.952, 'source': 'Kodak benchmark'},
    # -- Neural / learned image compression --
    'Ballé et al. 2018 (hyperprior)':
                        {'psnr': 36.9,  'ssim': 0.951, 'source': 'NeurIPS 2018'},
    'Minnen et al. 2018 (joint AR+HP)':
                        {'psnr': 38.7,  'ssim': 0.965, 'source': 'NeurIPS 2018'},
    'Cheng et al. 2020 (attn)':
                        {'psnr': 39.5,  'ssim': 0.970, 'source': 'CVPR 2020'},
    # -- NeRF / Gaussian representation (not direct Kodak compression, context only) --
    '3D-GS (Kerbl 2023, novel view)':
                        {'psnr': 27.2,  'ssim': None,  'source': 'SIGGRAPH 2023'},
}

METRICS = ['psnr', 'ssim', 'rmse', 'lpips']
METRIC_LABELS = {'psnr': 'PSNR (dB)', 'ssim': 'SSIM', 'rmse': 'RMSE', 'lpips': 'LPIPS'}
HIGHER_BETTER = {'psnr': True, 'ssim': True, 'rmse': False, 'lpips': False}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # The hybrid results.csv prefixes final metrics with 'hybrid_'
    rename = {}
    for m in METRICS:
        if f'hybrid_{m}' in df.columns and m not in df.columns:
            rename[f'hybrid_{m}'] = m
    if rename:
        df = df.rename(columns=rename)
    n_col = 'n_gaussians_final' if 'n_gaussians_final' in df.columns else 'n_gaussians'
    df['image_stem'] = df['image'].apply(lambda x: Path(x).stem)
    keep = ['image_stem'] + [m for m in METRICS if m in df.columns]
    if n_col in df.columns:
        keep += [n_col]
    if 'total_time' in df.columns:
        keep += ['total_time']
    df = df[keep].copy()
    df['variant'] = label
    return df


def _agg(df: pd.DataFrame) -> pd.DataFrame:
    cols = [m for m in METRICS if m in df.columns]
    rows = []
    for stat in ['mean', 'std', 'min', 'max']:
        row = {'stat': stat}
        for m in cols:
            row[m] = getattr(df[m], stat)()
        rows.append(row)
    return pd.DataFrame(rows)


def _bar(df_list: list, metric: str, out: Path):
    images = sorted(df_list[0]['image_stem'].unique())
    x      = np.arange(len(images))
    w      = 0.8 / max(len(df_list), 1)
    colours = ['#4895ef', '#f4a261', '#2a9d8f', '#e76f51']

    fig, ax = plt.subplots(figsize=(max(9, len(images) * 0.8), 5))
    for i, df in enumerate(df_list):
        vals = [df.loc[df['image_stem'] == im, metric].values[0]
                if im in df['image_stem'].values else np.nan
                for im in images]
        offset = (i - (len(df_list) - 1) / 2) * w
        bars = ax.bar(
            x + offset, vals, w,
            label=df['variant'].iloc[0], color=colours[i % len(colours)],
            edgecolor='white', linewidth=0.5, alpha=0.88,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(images, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=10)
    direction = 'higher is better' if HIGHER_BETTER[metric] else 'lower is better'
    ax.set_title(
        f'{METRIC_LABELS[metric]} per Kodak image — Hybrid Residual Refinement\n'
        f'({direction})',
        fontsize=11, fontweight='bold',
    )
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)


def _violin(df_combined: pd.DataFrame, out: Path):
    metrics_avail = [m for m in METRICS if m in df_combined.columns]
    n = len(metrics_avail)
    variants = df_combined['variant'].unique().tolist()
    colours   = ['#4895ef', '#f4a261', '#2a9d8f', '#e76f51']

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)
    for idx, m in enumerate(metrics_avail):
        ax = axes[0][idx]
        data_per_variant = [
            df_combined.loc[df_combined['variant'] == v, m].dropna().values
            for v in variants
        ]
        parts = ax.violinplot(
            data_per_variant, positions=range(len(variants)),
            showmedians=True, showextrema=True,
        )
        for pc, col in zip(parts['bodies'], colours):
            pc.set_facecolor(col)
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, fontsize=8, rotation=15, ha='right')
        ax.set_ylabel(METRIC_LABELS[m], fontsize=9)
        ax.set_title(METRIC_LABELS[m], fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(
        'Metric distributions across 24 Kodak images\n(Hybrid Residual Refinement)',
        fontsize=11, fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)


def _print_table(agg_dfs: dict):
    """Print a LaTeX-compatible summary table to stdout."""
    metrics_avail = [m for m in METRICS if any(m in df.columns for df in agg_dfs.values())]

    print('\n' + '=' * 72)
    print('  AGGREGATE METRICS — Hybrid Residual Refinement on Kodak (24 images)')
    print('=' * 72)
    header = f"{'Variant':<22}" + ''.join(f'{METRIC_LABELS[m]:>14}' for m in metrics_avail)
    print(header)
    print('-' * len(header))
    for label, agg in agg_dfs.items():
        mean_row = agg.loc[agg['stat'] == 'mean'].iloc[0]
        std_row  = agg.loc[agg['stat'] == 'std'].iloc[0]
        vals = ''.join(
            f'{mean_row[m]:>8.4f}±{std_row[m]:>5.4f}'
            for m in metrics_avail if m in mean_row
        )
        print(f'{label:<22}{vals}')
    print('=' * 72)

    print('\n  LITERATURE REFERENCE  (Kodak full-res 768×512, various bpp)')
    print('-' * 72)
    for name, ref in LITERATURE.items():
        p = f"{ref['psnr']:.1f}" if ref.get('psnr') else '  —  '
        s = f"{ref['ssim']:.3f}" if ref.get('ssim') else '  —  '
        print(f"  {name:<42}  PSNR {p:>5} dB   SSIM {s}")
    print()
    print('  NOTE: Literature values are at various bitrates and on full-res')
    print('  Kodak (768×512).  Our runs use 256×256 resized images with a fixed')
    print('  Gaussian budget — direct numerical comparison is indicative only.')
    print('=' * 72 + '\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate hybrid residual refinement on full Kodak dataset'
    )
    parser.add_argument('--minibatch', default=None,
                        help='Path to minibatch run results.csv')
    parser.add_argument('--fullbatch', default=None,
                        help='Path to fullbatch run results.csv')
    parser.add_argument('--out', default='results/hybrid_kodak_eval',
                        help='Output directory (default: results/hybrid_kodak_eval)')
    args = parser.parse_args()

    if not args.minibatch and not args.fullbatch:
        parser.error('Provide at least one of --minibatch or --fullbatch')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    agg_dfs = {}
    if args.minibatch:
        df_mb = _load(args.minibatch, 'Hybrid (minibatch)')
        dfs.append(df_mb)
        agg_dfs['Hybrid (minibatch)'] = _agg(df_mb)
    if args.fullbatch:
        df_fb = _load(args.fullbatch, 'Hybrid (fullbatch)')
        dfs.append(df_fb)
        agg_dfs['Hybrid (fullbatch)'] = _agg(df_fb)

    _print_table(agg_dfs)

    # ---- Save tables ------------------------------------------------------- #
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(out_dir / 'per_image_table.csv', index=False)

    agg_all = pd.concat(
        [df.assign(variant=label) for label, df in agg_dfs.items()],
        ignore_index=True,
    )
    agg_all.to_csv(out_dir / 'aggregate_stats.csv', index=False)
    print(f'  Tables saved to {out_dir}')

    # ---- Plots ------------------------------------------------------------- #
    for m in METRICS:
        _bar(dfs, m, out_dir / f'{m}_per_image.png')
        print(f'  {m}_per_image.png')

    _violin(combined, out_dir / 'metrics_distribution.png')
    print(f'  metrics_distribution.png')

    print(f'\nAll outputs in: {out_dir}\n')


if __name__ == '__main__':
    main()
