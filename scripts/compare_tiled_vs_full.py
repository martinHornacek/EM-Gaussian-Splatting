"""
scripts/compare_tiled_vs_full.py
---------------------------------
Compare the latest tiled-EM run against the latest full-image hybrid run
(or a specified pair of output directories).

Outputs:
  - Console table: PSNR / SSIM / RMSE / time for each image and each method
  - outputs/comparison_tiled_vs_full.png  — bar charts + scatter plot

Usage:
  python scripts/compare_tiled_vs_full.py
  python scripts/compare_tiled_vs_full.py \\
      --tiled   outputs/tiled_em_YYYYMMDD_HHMMSS \\
      --full    outputs/hybrid_residual_YYYYMMDD_HHMMSS
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


# ---------------------------------------------------------------------------
# Path discovery helpers
# ---------------------------------------------------------------------------

def _latest_subdir(outputs_root: Path, prefix: str) -> Path | None:
    """Return the most-recently-created sub-directory matching *prefix*."""
    candidates = sorted(
        [d for d in outputs_root.iterdir()
         if d.is_dir() and d.name.startswith(prefix)],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _load_results(run_dir: Path, csv_name: str = 'results.csv') -> pd.DataFrame | None:
    p = run_dir / csv_name
    if not p.exists():
        print(f"  WARNING: {p} not found")
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    """
    Return a DataFrame with standardised metric columns:
      psnr, ssim, rmse, lpips, total_time

    Tries common column naming patterns used by both run_hybrid_residual.py
    and run_tiled_em.py.
    """
    rename_map = {}

    # PSNR
    for cand in [f'{prefix}psnr', 'hybrid_psnr', 'refined_psnr',
                 'composite_psnr', 'psnr']:
        if cand in df.columns:
            rename_map[cand] = 'psnr'
            break

    # SSIM
    for cand in [f'{prefix}ssim', 'hybrid_ssim', 'refined_ssim',
                 'composite_ssim', 'ssim']:
        if cand in df.columns:
            rename_map[cand] = 'ssim'
            break

    # RMSE
    for cand in [f'{prefix}rmse', 'hybrid_rmse', 'refined_rmse',
                 'composite_rmse', 'rmse']:
        if cand in df.columns:
            rename_map[cand] = 'rmse'
            break

    # LPIPS
    for cand in [f'{prefix}lpips', 'hybrid_lpips', 'refined_lpips',
                 'composite_lpips', 'lpips']:
        if cand in df.columns:
            rename_map[cand] = 'lpips'
            break

    # Time
    for cand in ['total_time', 'tile_time']:
        if cand in df.columns:
            rename_map[cand] = 'total_time'
            break

    # image → image
    if 'image' not in rename_map:
        rename_map['image'] = 'image'

    return df.rename(columns=rename_map)


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def compare(
    tiled_dir : Path,
    full_dir  : Path,
    output_dir: Path,
) -> None:
    print(f"\nTiled run   : {tiled_dir}")
    print(f"Full run    : {full_dir}\n")

    df_tiled_raw = _load_results(tiled_dir)
    df_full_raw  = _load_results(full_dir)

    if df_tiled_raw is None or df_full_raw is None:
        print("Cannot compare — results missing.")
        return

    # Try composite columns first, then refined (for tiled), then hybrid (for full)
    def _best_prefix(df):
        for p in ('refined_', 'composite_', 'hybrid_', ''):
            if f'{p}psnr' in df.columns:
                return p
        return ''

    tiled_prefix = _best_prefix(df_tiled_raw)
    full_prefix  = _best_prefix(df_full_raw)

    df_tiled = _normalise_columns(df_tiled_raw.copy(), tiled_prefix)
    df_full  = _normalise_columns(df_full_raw.copy(),  full_prefix)

    # Merge on image name
    merged = df_full[['image', 'psnr', 'ssim', 'rmse', 'total_time']].merge(
        df_tiled[['image', 'psnr', 'ssim', 'rmse', 'total_time',
                  *([c for c in df_tiled.columns if c == 'composite_psnr'] or [])]],
        on='image', suffixes=('_full', '_tiled'),
    )
    # Also add composite PSNR if present (to see gain from GD)
    if tiled_prefix == 'refined_' and 'composite_psnr' in df_tiled_raw.columns:
        merged = merged.merge(
            df_tiled_raw[['image', 'composite_psnr']].rename(
                columns={'composite_psnr': 'psnr_tiled_noGD'}),
            on='image', how='left',
        )

    # ---- Console table ------------------------------------------------------ #
    print("=" * 90)
    print(f"{'Image':>14}  {'Full PSNR':>10}  {'Tiled PSNR':>10}  "
          f"{'Full SSIM':>10}  {'Tiled SSIM':>10}  "
          f"{'Full t(s)':>9}  {'Tiled t(s)':>10}")
    print("-" * 90)

    for _, row in merged.iterrows():
        full_p = row.get('psnr_full', float('nan'))
        tile_p = row.get('psnr_tiled', float('nan'))
        full_s = row.get('ssim_full', float('nan'))
        tile_s = row.get('ssim_tiled', float('nan'))
        full_t = row.get('total_time_full', float('nan'))
        tile_t = row.get('total_time_tiled', float('nan'))
        delta  = tile_p - full_p
        sign   = '+' if delta >= 0 else ''
        print(f"{row['image']:>14}  {full_p:>10.2f}  {tile_p:>10.2f} ({sign}{delta:.2f})  "
              f"{full_s:>10.4f}  {tile_s:>10.4f}  {full_t:>9.1f}  {tile_t:>10.1f}")

    print("-" * 90)

    def _col_mean(col):
        return merged[col].mean() if col in merged.columns else float('nan')

    avg_psnr_full  = _col_mean('psnr_full')
    avg_psnr_tiled = _col_mean('psnr_tiled')
    avg_ssim_full  = _col_mean('ssim_full')
    avg_ssim_tiled = _col_mean('ssim_tiled')
    avg_t_full     = _col_mean('total_time_full')
    avg_t_tiled    = _col_mean('total_time_tiled')

    print(f"{'AVERAGE':>14}  {avg_psnr_full:>10.2f}  {avg_psnr_tiled:>10.2f} "
          f"({avg_psnr_tiled - avg_psnr_full:+.2f})  "
          f"{avg_ssim_full:>10.4f}  {avg_ssim_tiled:>10.4f}  "
          f"{avg_t_full:>9.1f}  {avg_t_tiled:>10.1f}")
    print("=" * 90)

    speedup = avg_t_full / avg_t_tiled if avg_t_tiled > 0 else float('nan')
    print(f"\nSpeedup (full → tiled): {speedup:.2f}×")
    print(f"PSNR delta (tiled − full): {avg_psnr_tiled - avg_psnr_full:+.2f} dB")
    print(f"SSIM delta (tiled − full): {avg_ssim_tiled - avg_ssim_full:+.4f}")

    # ---- GD gain (if available) -------------------------------------------- #
    if 'psnr_tiled_noGD' in merged.columns:
        avg_noGD = merged['psnr_tiled_noGD'].mean()
        print(f"\nGD refinement gain (composite → refined): "
              f"{avg_psnr_tiled - avg_noGD:+.2f} dB")
        if 'gd_time' in df_tiled_raw.columns:
            avg_gd_t = df_tiled_raw['gd_time'].mean()
            print(f"Mean GD time: {avg_gd_t:.1f} s")

    # ---- Save results CSV -------------------------------------------------- #
    out_csv = output_dir / 'comparison_tiled_vs_full.csv'
    merged.to_csv(out_csv, index=False)
    print(f"\nComparison CSV saved: {out_csv}")

    # ---- Plots ------------------------------------------------------------- #
    if not _MPL:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Tiled EM vs Full-Image Hybrid\n'
                 f'({tiled_dir.name}  vs  {full_dir.name})',
                 fontsize=11, fontweight='bold')

    images = merged['image'].tolist()
    x      = np.arange(len(images))
    w      = 0.35

    # PSNR bars
    ax = axes[0]
    bars_f = ax.bar(x - w/2, merged['psnr_full'].values,  w, label='Full-image',
                    color='steelblue', alpha=0.85)
    bars_t = ax.bar(x + w/2, merged['psnr_tiled'].values, w, label='Tiled',
                    color='tomato', alpha=0.85)
    if 'psnr_tiled_noGD' in merged.columns:
        ax.bar(x + w/2, merged['psnr_tiled_noGD'].values, w,
               label='Tiled (no GD)', color='salmon', alpha=0.45)
    ax.set_xticks(x); ax.set_xticklabels(images, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('PSNR (dB)'); ax.set_title('PSNR comparison'); ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # SSIM bars
    ax = axes[1]
    ax.bar(x - w/2, merged['ssim_full'].values,  w, label='Full-image',
           color='steelblue', alpha=0.85)
    ax.bar(x + w/2, merged['ssim_tiled'].values, w, label='Tiled',
           color='tomato', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(images, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('SSIM'); ax.set_title('SSIM comparison'); ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Timing scatter
    ax = axes[2]
    ax.scatter(merged['total_time_full'].values,
               merged['total_time_tiled'].values,
               c='darkorange', s=60, zorder=3)
    for _, row in merged.iterrows():
        ax.annotate(row['image'].replace('.png', ''),
                    (row.get('total_time_full', 0), row.get('total_time_tiled', 0)),
                    textcoords='offset points', xytext=(4, 3), fontsize=7)
    lim = max(merged.get('total_time_full', pd.Series([1])).max(),
              merged.get('total_time_tiled', pd.Series([1])).max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.4, label='y=x (equal time)')
    ax.set_xlabel('Full-image time (s)'); ax.set_ylabel('Tiled time (s)')
    ax.set_title('Runtime comparison'); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_png = output_dir / 'comparison_tiled_vs_full.png'
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparison plot: {out_png}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare tiled-EM vs full-image hybrid EM run results')
    parser.add_argument('--tiled',   default='',  help='Path to tiled_em_* output dir')
    parser.add_argument('--full',    default='',  help='Path to hybrid_residual_* output dir')
    parser.add_argument('--outputs', default='outputs', help='Root outputs directory')
    args = parser.parse_args()

    outputs     = Path(args.outputs)
    tiled_dir   = Path(args.tiled)  if args.tiled  else _latest_subdir(outputs, 'tiled_em_')
    full_dir    = Path(args.full)   if args.full   else _latest_subdir(outputs, 'hybrid_residual_')

    if tiled_dir is None:
        print(f"No tiled_em_* directory found under {outputs}. "
              "Run  python run_tiled_em.py  first.")
        return
    if full_dir is None:
        print(f"No hybrid_residual_* directory found under {outputs}. "
              "Run  python run_hybrid_residual.py  first.")
        return

    compare(tiled_dir, full_dir, outputs)


if __name__ == '__main__':
    main()
