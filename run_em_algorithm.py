"""
run_em_algorithm.py - EM Algorithm for Gaussian Splatting
Fits standard EM and mini-batch EM on image pixels (5D: [x,y,r,g,b]),
computes reconstruction metrics, and produces diagnostic plots.

Configuration via config.yml:
    em:
        n_components: 256              # Number of Gaussian components
        covariance_type: full          # 'full', 'diag', 'spherical', 'tied'
        max_iter: 500                  # Maximum EM iterations
        minibatch:
            subsample_ratio: 0.15      # Fraction of pixels to subsample
            min_samples: 5000          # Minimum pixels to use
    
    experiment:
        run_diagnostics: true          # Generate diagnostic plots
        diagnostic_n_std: 2.0          # Ellipse size for diagnostics
        save_plots: true               # Save comparison plots
        save_renders: true             # Save rendered images
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from PIL import Image
from sklearn.mixture import GaussianMixture

from utils.metrics_utils import evaluate_metrics, print_metrics
from utils.dataset_utils import select_images
from utils.em_utils import (load_config, download_kodak_dataset,
                             _prepare_data, render_gaussians)
from utils.diagnostic_utils import (SplatFit, run_splat_diagnostics, make_summary_strip)



# ---------------------------------------------------------------------------
# EM fitting functions
# ---------------------------------------------------------------------------

def _fit_em_standard(img_np, config, n_comp):
    """Fit standard EM on full 5D pixel data."""
    data_5d, h, w = _prepare_data(img_np)
    cov_type = config.get('covariance_type', 'full')
    gmm = GaussianMixture(
        n_components=n_comp, covariance_type=cov_type,
        max_iter=config.get('max_iter', 100),
        init_params='kmeans', random_state=42)
    gmm.fit(data_5d)
    render_np = render_gaussians(
        gmm.means_[:, :2] * np.array([[w, h]]),
        gmm.covariances_[:, :2, :2] * np.array([[[w, h], [w, h]]]) ** 2,
        np.clip(gmm.means_[:, 2:5], 0, 1),
        gmm.weights_, (h, w))
    return render_np, gmm, data_5d, h, w


def _fit_em_minibatch(img_np, config, n_comp):
    """Fit EM on subsampled pixels for speed."""
    data_5d, h, w = _prepare_data(img_np)
    mb_cfg = config.get('minibatch', {})
    ratio = mb_cfg.get('subsample_ratio', 0.15)
    min_s = mb_cfg.get('min_samples', 5000)
    n_sub = max(int(len(data_5d) * ratio), min_s, n_comp * 10)
    n_sub = min(n_sub, len(data_5d))
    rng = np.random.default_rng(42)
    data_fit = data_5d[rng.choice(len(data_5d), n_sub, replace=False)]
    
    cov_type = config.get('covariance_type', 'full')
    gmm = GaussianMixture(
        n_components=n_comp, covariance_type=cov_type,
        max_iter=config.get('max_iter', 100),
        init_params='kmeans', random_state=42)
    gmm.fit(data_fit)
    render_np = render_gaussians(
        gmm.means_[:, :2] * np.array([[w, h]]),
        gmm.covariances_[:, :2, :2] * np.array([[[w, h], [w, h]]]) ** 2,
        np.clip(gmm.means_[:, 2:5], 0, 1),
        gmm.weights_, (h, w))
    return render_np, gmm, data_5d, h, w


def _build_splat_fit(gmm, cov_type, h, w, data_5d, variant):
    """Build SplatFit from fitted GMM."""
    K = gmm.n_components
    covs_pos = gmm.covariances_[:, :2, :2] * np.array([[[w, h], [w, h]]]) ** 2
    r = gmm.predict_proba(data_5d)
    return SplatFit(
        means_pos        = gmm.means_[:, :2] * np.array([[w, h]]),
        covs_pos         = covs_pos,
        colors           = np.clip(gmm.means_[:, 2:5], 0, 1),
        weights          = gmm.weights_,
        responsibilities = r,
        n_iter           = gmm.n_iter_,
        converged        = gmm.converged_,
        feature_dim      = 5,
        variant          = variant,
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def create_variant_comparison(gt_np, renders, metrics_per_variant,
                               output_dir, image_name):
    variant_names = list(renders.keys())
    n_cols = 1 + len(variant_names)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), squeeze=False)
    axes = axes[0]
    axes[0].imshow(gt_np)
    axes[0].set_title('Ground Truth', fontsize=13, fontweight='bold')
    axes[0].axis('off')
    for ax, name in zip(axes[1:], variant_names):
        ax.imshow(renders[name])
        m = metrics_per_variant[name]
        ax.set_title(
            f"{name.upper()}\n"
            f"PSNR {m['psnr']:.2f} dB  SSIM {m['ssim']:.4f}\n"
            f"fit {m.get('fit_time', 0):.1f}s",
            fontsize=11,
        )
        ax.axis('off')
    plt.suptitle(image_name, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / f'{image_name}_variant_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_plot(results_df, output_dir):
    variants = results_df['variant'].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    images = results_df['image'].unique()
    x = np.arange(len(images))
    width = 0.8 / len(variants)
    for i, v in enumerate(variants):
        sub = results_df[results_df['variant'] == v]
        psnrs = [sub[sub['image'] == img]['psnr'].values[0]
                 if img in sub['image'].values else 0
                 for img in images]
        axes[0].bar(x + i * width, psnrs, width, label=v)
    axes[0].set_xticks(x + width * (len(variants) - 1) / 2)
    axes[0].set_xticklabels([i.replace('.png', '') for i in images],
                             rotation=30, ha='right', fontsize=8)
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR by Variant and Image')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    colors = plt.cm.tab10(np.linspace(0, 1, len(variants)))
    for color, v in zip(colors, variants):
        sub = results_df[results_df['variant'] == v]
        axes[1].scatter(sub['fit_time'], sub['psnr'],
                        label=v, s=60, alpha=0.8, color=color)
        for _, row in sub.iterrows():
            axes[1].annotate(
                row['image'].replace('kodim', '').replace('.png', ''),
                (row['fit_time'], row['psnr']), fontsize=7, alpha=0.6)
    axes[1].set_xlabel('Fit time (s)')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('Quality vs Speed Trade-off')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_em_algorithm(config_path='config.yml'):
    config = load_config(config_path)

    if not config['em'].get('enabled', True):
        print("EM algorithm disabled in config")
        return

    variants = ['standard', 'minibatch']
    exp_cfg = config.get('experiment', {})
    run_diag = exp_cfg.get('run_diagnostics', True)
    diag_n_std = exp_cfg.get('diagnostic_n_std', 2.0)

    print("\n" + "=" * 70)
    print("EM ALGORITHM")
    print(f"Variants: {variants}")
    if run_diag:
        print(f"Diagnostics enabled (n_std={diag_n_std})")
    print("=" * 70)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(exp_cfg['output_dir']) / f"em_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}\n")

    with open(output_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    kodak_dir   = download_kodak_dataset(config['dataset']['path'])
    image_paths = select_images(kodak_dir, config['dataset'])
    n_comp      = config['em']['n_components']

    print(f"Images: {len(image_paths)}  |  Components: {n_comp}")
    print("-" * 70)

    all_results      = []
    all_gt           = []
    all_fits_list    = []
    all_diag_scalars = []

    for img_idx, img_path in enumerate(image_paths, 1):
        print(f"\n[{img_idx}/{len(image_paths)}] {img_path.name}")

        img   = Image.open(img_path).convert('RGB')
        img   = img.resize(tuple(config['dataset']['image_size']), Image.LANCZOS)
        gt_np = np.array(img).astype(np.float32) / 255.0

        renders             = {}
        metrics_per_variant = {}
        fits_this_image     = {}

        for variant in variants:
            print(f"  → {variant}")
            t0 = time.time()
            try:
                if variant == 'standard':
                    render_np, gmm, data_5d, h, w = _fit_em_standard(
                        gt_np, config['em'], n_comp)
                elif variant == 'minibatch':
                    render_np, gmm, data_5d, h, w = _fit_em_minibatch(
                        gt_np, config['em'], n_comp)
                else:
                    raise ValueError(f"Unknown variant: {variant}")
                fit_time = time.time() - t0
            except Exception as exc:
                print(f"    ✗ {variant} failed: {exc}")
                continue

            metrics = evaluate_metrics(
                render_np, gt_np, n_components=n_comp,
                compute_lpips_flag=config['metrics'].get('compute_lpips', False))
            print_metrics(metrics, method_name=variant.upper())
            print(f"    fit_time={fit_time:.2f}s")

            renders[variant] = render_np
            metrics_per_variant[variant] = metrics

            if exp_cfg.get('save_renders', True):
                out_img = (render_np * 255).astype(np.uint8)
                Image.fromarray(out_img).save(
                    output_dir / f'{img_path.stem}_{variant}_render.png')

            all_results.append({
                'image': img_path.name, 'variant': variant,
                'fit_time': fit_time, **metrics})

            # Build SplatFit for diagnostics
            if run_diag:
                try:
                    cov_type = config['em'].get('covariance_type', 'full')
                    sf = _build_splat_fit(gmm, cov_type, h, w, data_5d, variant)
                    fits_this_image[variant] = sf
                except Exception as exc:
                    print(f"    ⚠ SplatFit build failed ({variant}): {exc}")

        if exp_cfg.get('save_plots', True) and renders:
            create_variant_comparison(
                gt_np, renders, metrics_per_variant, output_dir, img_path.stem)

        if run_diag and fits_this_image:
            scalars = run_splat_diagnostics(
                gt_np, fits_this_image, output_dir,
                img_path.stem, n_std=diag_n_std)
            all_diag_scalars.append({
                'image': img_path.name,
                **{f'{v}_{k}': val
                   for v, s in scalars.items()
                   for k, val in s.items()}
            })

        all_gt.append(gt_np)
        all_fits_list.append(fits_this_image)

    # ---- Aggregate ----
    if not all_results:
        print("No results to summarise.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'results.csv', index=False)

    print("\n" + "=" * 70)
    print("SUMMARY (mean across images)")
    print("=" * 70)
    summary_cols = ['psnr', 'ssim', 'rmse', 'fit_time',
                    'bits_per_pixel', 'compression_ratio']
    summary_cols = [c for c in summary_cols if c in results_df.columns]
    summary = results_df.groupby('variant')[summary_cols].mean().round(4)
    print(summary.to_string())
    summary.to_csv(output_dir / 'summary.csv')

    if exp_cfg.get('save_plots', True):
        create_summary_plot(results_df, output_dir)

    if run_diag and all_diag_scalars:
        pd.DataFrame(all_diag_scalars).to_csv(
            output_dir / 'diagnostics_scalars.csv', index=False)

    if run_diag and len(all_fits_list) > 1:
        valid_fits  = [f for f in all_fits_list if f]
        valid_gt    = all_gt[:len(valid_fits)]
        valid_names = [p.name for p in image_paths[:len(valid_fits)]]
        if valid_fits:
            fig = make_summary_strip(valid_names, valid_fits, valid_gt)
            fig.savefig(output_dir / 'diagnostic_summary_strip.png',
                        dpi=120, bbox_inches='tight')
            plt.close(fig)

    print(f"\n✓ Done. Results saved to: {output_dir}")
    print("=" * 70 + "\n")
    return output_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml')
    args = parser.parse_args()
    run_em_algorithm(args.config)