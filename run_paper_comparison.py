"""
run_paper_comparison.py — Paper comparison experiment runner.

Runs three method variants at K in {128, 256, 512, 1024, 2048} on the full 24-image
Kodak dataset:
  1. EM one-shot  full-batch   — all 65536 pixels, single GMM fit
  2. Hybrid EM   full-batch   — residual refinement, all pixels every iteration
  3. Hybrid EM   mini-batch   — residual refinement, sub-sampled pixels (fast)

For each run, run_hybrid_residual.py is called with compare_pure_em=True so the
EM baseline and the hybrid result share an identical setup (same seed, preprocessing,
EM settings).  Renders are saved for every image; a visual comparison strip is
assembled at the end.

Results are written to:
  results/paper_comparison/
    summary.csv                       — aggregate table across all methods/budgets
    per_run/<variant>_k<K>/
      results.csv                     — per-image metrics from run_hybrid_residual
      renders/                        — GT / EM / Hybrid renders + residual maps
      comparison_strips/              — side-by-side comparison images per kodim

Usage:
  python run_paper_comparison.py

Estimated runtime on CPU (24 images):
  full-batch  K=128..2048 : ~4–6 h  (≈1 h per K level, 2 variants per level)
  mini-batch  K=128..2048 : ~1–2 h  (≈25 min per K level)
"""

import sys
import subprocess
import yaml
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Budget splits: (n_total, n_init, n_per_iter, max_iter)
# Invariant: n_init + max_iter * n_per_iter == n_total
# ---------------------------------------------------------------------------
BUDGETS = [
    (128,   16,   8, 14),
    (256,   32,  16, 14),
    (512,   64,  32, 14),
    (1024, 128,  64, 14),
    # (2048, 256, 128, 14),  # skipped — too slow on CPU
]

KODAK_DIR = Path('./kodak')
IMAGE_SIZE = [256, 256]

BASE_EM = {
    'enabled': True,
    'covariance_type': 'full',
    'max_iter': 100,
    'minibatch': {
        'subsample_ratio': 0.15,
        'min_samples': 5000,
    },
}


def make_config(n_total, n_init, n_per_iter, max_iter, use_minibatch, run_output_dir):
    em_variant = 'minibatch' if use_minibatch else 'standard'
    return {
        'dataset': {
            'path': './kodak',
            'image_size': IMAGE_SIZE,
            'selection_mode': 'full',
            'random_count': 5,
            'image_list': [15],     # ignored when selection_mode = full
        },
        'experiment': {
            'output_dir': str(run_output_dir),
            'save_plots': False,
            'save_renders': True,   # saves *_pure_em_render.png and *_hybrid_render.png
            'save_raw_data': False,
            'run_diagnostics': False,
        },
        'em': BASE_EM,
        'gaussian_splatting': {'enabled': False, 'lambda_dssim': 0.2},
        'hybrid_residual': {
            'enabled': True,
            'n_total': n_total,
            'n_init': n_init,
            'n_per_iter': n_per_iter,
            'max_iter': max_iter,
            'residual_threshold': 0.0001,
            'step_size': 0.5,
            'oracle_clamp': False,
            'em_variant': em_variant,
            'use_minibatch': use_minibatch,
            'compare_pure_em': True,   # run one-shot EM at n_total alongside hybrid
            'gd': {'enabled': False},
            'visualize': {'enabled': False, 'fps': 2, 'make_mp4': False, 'make_gif': False},
        },
        'metrics': {'compute_lpips': False},
        'device': 'cpu',
    }


def run_one(cfg_path):
    result = subprocess.run(
        [sys.executable, 'run_hybrid_residual.py', '--config', str(cfg_path)],
        capture_output=False,
        text=True,
    )
    return result.returncode == 0


def find_run_dir(run_output_dir: Path):
    """Return the single hybrid_residual_* subdir created inside run_output_dir."""
    dirs = sorted(run_output_dir.glob('hybrid_residual_*'))
    return dirs[-1] if dirs else None


# ---------------------------------------------------------------------------
# Visual output helpers
# ---------------------------------------------------------------------------

def _load_img(path):
    return np.asarray(Image.open(path).convert('RGB')).astype(np.float32) / 255.0


def make_comparison_strips(run_dir: Path, strip_dir: Path, n_total: int):
    """
    For every kodim* in run_dir that has both a pure_em_render and a
    hybrid_render, assemble a 4-panel strip:
      GT | EM one-shot | Hybrid EM | |GT - Hybrid| residual map
    Save as strip_dir/<stem>_strip.png.
    """
    strip_dir.mkdir(parents=True, exist_ok=True)
    em_renders  = sorted(run_dir.glob('*_pure_em_render.png'))
    made = 0
    for em_path in em_renders:
        stem = em_path.name.replace('_pure_em_render.png', '')
        hyb_path = run_dir / (stem + '_hybrid_render.png')
        gt_candidates = sorted(KODAK_DIR.glob(stem + '.png'))
        if not hyb_path.exists() or not gt_candidates:
            continue
        gt  = _load_img(gt_candidates[0])
        # resize to match renders (which are 256x256)
        gt_img = Image.fromarray((gt * 255).astype(np.uint8))
        render_ref = Image.open(em_path)
        if gt_img.size != render_ref.size:
            gt_img = gt_img.resize(render_ref.size, Image.LANCZOS)
        gt  = np.asarray(gt_img).astype(np.float32) / 255.0
        em  = _load_img(em_path)
        hyb = _load_img(hyb_path)
        res = np.abs(gt - hyb).mean(axis=2)   # mean-colour residual map

        em_psnr  = float(10 * np.log10(1.0 / max(np.mean((gt - em)  ** 2), 1e-10)))
        hyb_psnr = float(10 * np.log10(1.0 / max(np.mean((gt - hyb) ** 2), 1e-10)))

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        for ax, panel, title in zip(
            axes,
            [gt, em, hyb, res],
            [
                'Ground truth',
                f'EM one-shot  K={n_total}\nPSNR={em_psnr:.2f} dB',
                f'Hybrid EM (full-batch)  K={n_total}\nPSNR={hyb_psnr:.2f} dB',
                f'|GT − Hybrid|',
            ],
        ):
            if panel.ndim == 2:
                ax.imshow(panel, cmap='hot', vmin=0, vmax=0.15)
            else:
                ax.imshow(panel)
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        fig.suptitle(stem, fontsize=9, y=1.01)
        plt.tight_layout()
        out_path = strip_dir / (stem + '_strip.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        made += 1

    print(f'  Saved {made} comparison strip(s) to {strip_dir}')


# ---------------------------------------------------------------------------
def main(only_minibatch=False):
    out_root = Path('results/paper_comparison')
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []

    variants = [True] if only_minibatch else [False, True]
    for use_minibatch in variants:
        variant_tag = 'minibatch' if use_minibatch else 'fullbatch'
        print()
        print('=' * 70)
        print(f'VARIANT: {variant_tag.upper()}')
        print('=' * 70)

        for n_total, n_init, n_per_iter, max_iter in BUDGETS:
            run_label = f'{variant_tag}_k{n_total}'
            print(f'\n--- {run_label}  (n_init={n_init}  n_per_iter={n_per_iter}) ---')

            # keep each run's raw output in its own sub-directory so renders
            # don't overwrite each other across K values
            run_output_dir = out_root / 'per_run' / run_label
            run_output_dir.mkdir(parents=True, exist_ok=True)

            cfg = make_config(n_total, n_init, n_per_iter, max_iter,
                              use_minibatch, run_output_dir)
            cfg_path = run_output_dir / 'config.yml'
            with open(cfg_path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False)

            ok = run_one(cfg_path)
            if not ok:
                print(f'  ERROR: run failed for {run_label}')
                continue

            run_dir = find_run_dir(run_output_dir)
            if run_dir is None:
                print(f'  WARNING: no output dir found under {run_output_dir}')
                continue

            csv_path = run_dir / 'results.csv'
            if not csv_path.exists():
                print(f'  WARNING: results.csv missing in {run_dir}')
                continue

            df = pd.read_csv(csv_path)
            for _, r in df.iterrows():
                all_rows.append({
                    'variant':      variant_tag,
                    'n_total':      n_total,
                    'image':        r['image'],
                    'em_psnr':      r.get('pure_em_psnr', float('nan')),
                    'em_ssim':      r.get('pure_em_ssim', float('nan')),
                    'hybrid_psnr':  r['hybrid_psnr'],
                    'hybrid_ssim':  r['hybrid_ssim'],
                    'hybrid_rmse':  r['hybrid_rmse'],
                    'init_psnr':    r['init_psnr'],
                    'total_time':   r['total_time'],
                })
            print(f'  Collected {len(df)} image rows')

            # build visual comparison strips only for full-batch runs
            if not use_minibatch:
                make_comparison_strips(
                    run_dir,
                    run_output_dir / 'comparison_strips',
                    n_total,
                )

    # ---------------------------------------------------------------------- #
    # Save aggregate summary                                                  #
    # ---------------------------------------------------------------------- #
    if not all_rows:
        print('\nNo results collected.')
        return

    summary = pd.DataFrame(all_rows)
    summary_path = out_root / 'summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f'\nSaved summary → {summary_path}')

    # ---------------------------------------------------------------------- #
    # Print pivot tables                                                       #
    # ---------------------------------------------------------------------- #
    print()
    print('=' * 80)
    print('MEAN PSNR (dB) — full Kodak dataset (24 images)')
    print('=' * 80)
    pivot = summary.pivot_table(
        index='n_total',
        columns='variant',
        values=['em_psnr', 'hybrid_psnr'],
        aggfunc='mean',
    ).round(2)
    print(pivot.to_string())

    print()
    print('DELTA  hybrid − EM (positive = hybrid wins)')
    for tag in ['fullbatch', 'minibatch']:
        sub = summary[summary['variant'] == tag].groupby('n_total')[['em_psnr', 'hybrid_psnr']].mean()
        sub['delta'] = sub['hybrid_psnr'] - sub['em_psnr']
        print(f'\n  {tag}:')
        print(sub.round(2).to_string())

    # ---------------------------------------------------------------------- #
    # Tiled EM reference numbers (already computed)                           #
    # ---------------------------------------------------------------------- #
    print()
    print('TILED EM reference (full Kodak, 24 images, 64x64 tiles):')
    for k in [128, 256, 512, 1024, 2048]:
        dirs = sorted(Path('results/tiled_fullbatch_' + str(k)).glob('tiled_em_*'))
        if dirs:
            df_t = pd.read_csv(dirs[-1] / 'results.csv')
            col = 'composite_psnr'
            m = df_t[col].mean()
            print(f'  K={k:5d}: {m:.2f} dB  (dir={dirs[-1].name})')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--only-minibatch', action='store_true',
        help='Skip fullbatch runs (already done) and run only minibatch variants.',
    )
    args = parser.parse_args()
    main(only_minibatch=args.only_minibatch)
