"""
run_hybrid_residual.py — Hybrid Gaussian Splatting with Iterative Residual Refinement

Algorithm
---------
Step 1  Fit N_init Gaussians to I_gt using EM -> render I_init.
Step 2  Compute residual:  I_res = I_gt - I_current.
Step 3  Split:  I_pos = max(I_res, 0),  I_neg = max(-I_res, 0).
Step 4  Fit EM to I_pos (weighted by intensity) -> positive correction render.
        Fit EM to I_neg (weighted by intensity) -> negative correction render.
        Gaussians are allocated adaptively (more to the larger residual mass).
Step 5  Apply: I_current = clip(I_current + step * (I_pos_corr − I_neg_corr), 0, 1)
Step 6  Repeat steps 2–5 until total Gaussians ≥ N_total or residual < threshold.

Positive and negative correction Gaussians are stored with +1 / −1 amplitudes
respectively and can be retrieved as a unified signed Gaussian set at the end.

Configuration (hybrid_residual section of config.yml)
------------------------------------------------------
n_total            — total Gaussians across all stages          [512]
n_init             — Gaussians in the initial EM fit             [64]
n_per_iter         — Gaussians added per refinement iteration    [32]
max_iter           — maximum refinement iterations               [14]
residual_threshold — stop when mean |residual| is below this    [0.001]
step_size          — correction damping factor ∈ (0, 1]          [1.0]
em_variant         — 'standard' or 'minibatch' for initial fit  ['minibatch']
compare_pure_em    — also run pure EM with n_total Gaussians     [true]
"""

import yaml
import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from PIL import Image
from sklearn.mixture import GaussianMixture

from utils.metrics_utils import evaluate_metrics
from utils.dataset_utils import select_images
from utils.em_utils import (
    load_config, download_kodak_dataset, _prepare_data,
    render_gaussians, fit_em_to_distribution,
)
from utils.hybrid_vis_utils import (
    save_render_frame,
    save_diag_residuals,
    save_diag_gaussians,
    save_diag_metrics,
    save_comparison_3panel,
    assemble_mp4,
    assemble_gif,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cov_to_gaussian_row(iteration, gaussian_id, mean, cov, color, amplitude):
    """Return a flat log dict for a single Gaussian added at *iteration*."""
    try:
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-6)
        order = vals.argsort()[::-1]
        vals  = vals[order]
        vecs  = vecs[:, order]
        scale_x      = float(np.sqrt(vals[0]))
        scale_y      = float(np.sqrt(vals[1]))
        rotation_deg = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    except (np.linalg.LinAlgError, ValueError):
        scale_x = scale_y = rotation_deg = float('nan')
    return {
        'iteration':    iteration,
        'gaussian_id':  gaussian_id,
        'x':            float(mean[0]),
        'y':            float(mean[1]),
        'scale_x':      scale_x,
        'scale_y':      scale_y,
        'rotation_deg': rotation_deg,
        'amplitude':    float(amplitude),
        'r':            float(color[0]),
        'g':            float(color[1]),
        'b':            float(color[2]),
    }


def _initial_em_fit(gt_np, n_init, em_cfg, em_variant):
    """
    Fit an initial GMM with *n_init* components to *gt_np*.

    Returns the rendered image and the GMM pixel-space parameters
    ``(means_px, covs_px, colors, weights)``.
    """
    h, w = gt_np.shape[:2]
    data_5d, _, _ = _prepare_data(gt_np)

    if em_variant == 'minibatch':
        mb_cfg = em_cfg.get('minibatch', {})
        n_sub = max(
            int(len(data_5d) * mb_cfg.get('subsample_ratio', 0.15)),
            mb_cfg.get('min_samples', 5000),
            n_init * 10,
        )
        n_sub = min(n_sub, len(data_5d))
        rng = np.random.default_rng(42)
        data_fit = data_5d[rng.choice(len(data_5d), n_sub, replace=False)]
    else:
        data_fit = data_5d

    gmm = GaussianMixture(
        n_components=n_init,
        covariance_type=em_cfg.get('covariance_type', 'full'),
        max_iter=em_cfg.get('max_iter', 100),
        init_params='kmeans',
        random_state=42,
        reg_covar=1e-5,
    )
    try:
        gmm.fit(data_fit)
    except Exception as exc:
        raise RuntimeError(
            f"Initial EM fit failed ({n_init} components, "
            f"{'minibatch' if em_variant == 'minibatch' else 'full'} mode, "
            f"{len(data_fit)} samples): {exc}"
        ) from exc

    means_px  = gmm.means_[:, :2] * np.array([[w, h]])
    # Correct Jacobian transform Σ_px = J Σ_norm J^T where J = diag(w, h).
    cov_scale = np.array([[[w**2, w * h], [w * h, h**2]]])  # (1, 2, 2)
    covs_px   = gmm.covariances_[:, :2, :2] * cov_scale
    colors    = np.clip(gmm.means_[:, 2:5], 0.0, 1.0)
    weights   = gmm.weights_

    render = render_gaussians(means_px, covs_px, colors, weights, (h, w))
    return render, means_px, covs_px, colors, weights


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_hybrid_residual_refinement(config_path='config.yml'):
    """
    Run hybrid Gaussian splatting with iterative residual refinement.

    Reads all settings from the ``hybrid_residual`` section of *config_path*.
    """
    config = load_config(config_path)
    hr_cfg = config.get('hybrid_residual', {})

    if not hr_cfg.get('enabled', True):
        print("Hybrid residual refinement disabled in config.")
        return

    n_total         = hr_cfg.get('n_total', 512)
    n_init          = hr_cfg.get('n_init', 64)
    n_per_iter      = hr_cfg.get('n_per_iter', 32)
    max_iter        = hr_cfg.get('max_iter', 14)
    res_threshold   = hr_cfg.get('residual_threshold', 0.001)
    step_size       = float(hr_cfg.get('step_size', 1.0))
    em_variant      = hr_cfg.get('em_variant', 'minibatch')
    compare_pure_em = hr_cfg.get('compare_pure_em', True)
    use_minibatch   = hr_cfg.get('use_minibatch', True)

    # When use_minibatch=False, always use the full dataset for the initial fit too
    em_variant_actual = em_variant if use_minibatch else 'standard'

    vis_cfg        = hr_cfg.get('visualize', {})
    vis_enabled    = vis_cfg.get('enabled', True)
    vis_fps        = vis_cfg.get('fps', 2)
    vis_make_mp4   = vis_cfg.get('make_mp4', True)
    vis_make_gif   = vis_cfg.get('make_gif', False)
    # vis_n_std was consumed by the old ellipse-overlay diagnostic; kept in
    # config for forward-compatibility but is not used by current vis code.

    lambda_dssim = config.get('gaussian_splatting', {}).get('lambda_dssim', 0.2)
    exp_cfg = config.get('experiment', {})
    em_cfg  = config['em']

    # ---- Validate budget parameters ---------------------------------------- #
    if n_init < 1:
        raise ValueError(f"n_init must be >= 1, got {n_init}")
    if n_init >= n_total:
        raise ValueError(
            f"n_init ({n_init}) must be strictly less than n_total ({n_total}); "
            "at least one residual refinement iteration requires unused budget."
        )
    if n_per_iter < 1:
        raise ValueError(f"n_per_iter must be >= 1, got {n_per_iter}")

    print("\n" + "=" * 70)
    print("HYBRID RESIDUAL REFINEMENT")
    print(f"  N_total={n_total}  N_init={n_init}  N_per_iter={n_per_iter}"
          f"  Max_iter={max_iter}")
    print(f"  step_size={step_size}  residual_threshold={res_threshold}")
    print(f"  em_variant={em_variant_actual}  use_minibatch={use_minibatch}"
          f"  compare_pure_em={compare_pure_em}")
    print("=" * 70)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(exp_cfg['output_dir']) / f"hybrid_residual_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}\n")

    with open(output_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    kodak_dir   = download_kodak_dataset(config['dataset']['path'])
    image_paths = select_images(kodak_dir, config['dataset'])

    print(f"Images: {len(image_paths)}")
    print("-" * 70)

    all_results = []

    for img_idx, img_path in enumerate(image_paths, 1):
        print(f"\n[{img_idx}/{len(image_paths)}] {img_path.name}")

        img   = Image.open(img_path).convert('RGB')
        img   = img.resize(tuple(config['dataset']['image_size']), Image.LANCZOS)
        gt_np = np.array(img).astype(np.float32) / 255.0
        h, w  = gt_np.shape[:2]
        isize = (h, w)

        t_total  = time.time()
        row      = {'image': img_path.name}

        # ------------------------------------------------------------------ #
        # Optional baseline: pure EM with n_total Gaussians                  #
        # ------------------------------------------------------------------ #
        I_pure          = None
        pure_em_metrics = None

        if compare_pure_em:
            print(f"  Baseline pure EM  ({n_total} Gaussians) …")
            t0 = time.time()
            I_pure, *_ = _initial_em_fit(gt_np, n_total, em_cfg, em_variant_actual)
            pure_em_metrics = evaluate_metrics(
                I_pure, gt_np, n_components=n_total,
                compute_lpips_flag=config['metrics'].get('compute_lpips', False))
            pure_em_metrics['l1']   = float(np.mean(np.abs(I_pure - gt_np)))
            pure_em_metrics['loss'] = ((1.0 - lambda_dssim) * pure_em_metrics['l1']
                                       + lambda_dssim * pure_em_metrics['dssim'])
            row['pure_em_time'] = time.time() - t0
            row['pure_em_psnr'] = pure_em_metrics['psnr']
            row['pure_em_ssim'] = pure_em_metrics['ssim']
            print(f"    PSNR={pure_em_metrics['psnr']:.2f} dB  "
                  f"SSIM={pure_em_metrics['ssim']:.4f}  "
                  f"time={row['pure_em_time']:.1f}s")
            if exp_cfg.get('save_renders', True):
                Image.fromarray((I_pure * 255).astype(np.uint8)).save(
                    output_dir / f'{img_path.stem}_pure_em_render.png')

        # ------------------------------------------------------------------ #
        # Step 1 — Initial EM approximation                                  #
        # ------------------------------------------------------------------ #
        print(f"  Step 1: initial EM ({n_init} Gaussians) …")
        t0 = time.time()
        I_current, init_means, init_covs, init_colors, init_weights = \
            _initial_em_fit(gt_np, n_init, em_cfg, em_variant_actual)
        I_init = I_current.copy()   # preserved for comparison plot

        init_metrics = evaluate_metrics(
            I_current, gt_np, n_components=n_init,
            compute_lpips_flag=config['metrics'].get('compute_lpips', False))
        init_metrics['l1']   = float(np.mean(np.abs(I_current - gt_np)))
        init_metrics['loss'] = ((1.0 - lambda_dssim) * init_metrics['l1']
                                + lambda_dssim * init_metrics['dssim'])
        print(f"    PSNR={init_metrics['psnr']:.2f} dB  "
              f"SSIM={init_metrics['ssim']:.4f}  "
              f"Loss={init_metrics['loss']:.4f}  "
              f"time={time.time()-t0:.1f}s")

        if exp_cfg.get('save_renders', True):
            Image.fromarray((I_current * 255).astype(np.uint8)).save(
                output_dir / f'{img_path.stem}_init_render.png')

        # ------------------------------------------------------------------ #
        # Steps 2–5 — Iterative residual refinement                          #
        #                                                                     #
        # We track the full signed Gaussian set for introspection:           #
        #   all_means / all_covs / all_colors / all_amplitudes               #
        # ------------------------------------------------------------------ #
        n_gaussians_total = n_init
        iter_records      = []

        # Seed unified signed Gaussian list with initial Gaussians (amp = +1)
        all_means      = list(init_means)
        all_covs       = list(init_covs)
        all_colors     = list(init_colors)
        all_amplitudes = [1.0] * n_init

        # Gaussian log — one row per Gaussian recording when it was added
        gaussian_log_rows = []
        cum_gid = 0
        for m, cov, col in zip(init_means, init_covs, init_colors):
            gaussian_log_rows.append(
                _cov_to_gaussian_row(0, cum_gid, m, cov, col, 1.0)
            )
            cum_gid += 1

        # Residual snapshots for the post-run diagnostic grid
        residual_snapshots = [np.mean(np.abs(gt_np - I_current), axis=2)]
        snapshot_iters     = [0]

        # Per-image frame directory for video frames
        frame_dir_render = output_dir / f'{img_path.stem}_frames'
        render_frame_paths = []
        if vis_enabled:
            frame_dir_render.mkdir(exist_ok=True)
            # Frame 0 – initial state
            render_frame_paths.append(
                save_render_frame(
                    frame_dir_render, 0,
                    gt_np, I_current, gt_np - I_current,
                    init_metrics, n_gaussians_total,
                    init_metrics['loss'],
                )
            )

        for it in range(max_iter):
            if n_gaussians_total >= n_total:
                break

            I_res        = gt_np - I_current
            mean_abs_res = float(np.mean(np.abs(I_res)))

            if mean_abs_res < res_threshold:
                print(f"    Converged at iter {it+1} "
                      f"(mean|residual|={mean_abs_res:.6f})")
                break

            I_res_pos = np.maximum( I_res, 0.0).astype(np.float32)
            I_res_neg = np.maximum(-I_res, 0.0).astype(np.float32)

            # Adaptive allocation: proportional to residual mass
            n_available = min(n_per_iter, n_total - n_gaussians_total)
            pos_mass    = float(I_res_pos.sum())
            neg_mass    = float(I_res_neg.sum())
            if n_available < 2:
                # Can only add 1 Gaussian — allocate to the larger residual
                n_pos = n_available if pos_mass >= neg_mass else 0
                n_neg = n_available - n_pos
            else:
                pos_frac = pos_mass / (pos_mass + neg_mass + 1e-10)
                n_pos = max(1, round(n_available * pos_frac))
                n_neg = n_available - n_pos
                n_neg = max(1, n_neg)
                n_pos = n_available - n_neg   # guarantee n_pos + n_neg == n_available

            print(f"  Iter {it+1:3d}: mean|res|={mean_abs_res:.4f}  "
                  f"+{n_pos} positive / -{n_neg} negative Gaussians  "
                  f"(running total: {n_gaussians_total})")

            # ---- Fit positive residual -------------------------------------- #
            I_pos_corr    = np.zeros_like(gt_np)
            new_means_pos = None
            new_covs_pos  = None
            n_added_pos   = 0   # actual Gaussians returned by fitter (≤ n_pos)
            if pos_mass > 1e-8 and n_pos >= 1:
                pos_params = fit_em_to_distribution(
                    I_res_pos, n_pos, em_cfg, use_minibatch=use_minibatch)
                if pos_params is not None:
                    m, c, col, wt = pos_params
                    n_added_pos    = len(m)   # may be < n_pos when data is sparse
                    I_pos_corr = render_gaussians(m, c, col, wt, isize)
                    all_means.extend(list(m))
                    all_covs.extend(list(c))
                    all_colors.extend(list(col))
                    all_amplitudes.extend([1.0] * len(m))
                    new_means_pos = m
                    new_covs_pos  = c
                    for mi, ci, coli in zip(m, c, col):
                        gaussian_log_rows.append(
                            _cov_to_gaussian_row(it + 1, cum_gid, mi, ci, coli, 1.0)
                        )
                        cum_gid += 1

            # ---- Fit negative residual -------------------------------------- #
            I_neg_corr    = np.zeros_like(gt_np)
            new_means_neg = None
            new_covs_neg  = None
            n_added_neg   = 0   # actual Gaussians returned by fitter (≤ n_neg)
            if neg_mass > 1e-8 and n_neg >= 1:
                neg_params = fit_em_to_distribution(
                    I_res_neg, n_neg, em_cfg, use_minibatch=use_minibatch)
                if neg_params is not None:
                    m, c, col, wt = neg_params
                    n_added_neg    = len(m)   # may be < n_neg when data is sparse
                    I_neg_corr = render_gaussians(m, c, col, wt, isize)
                    all_means.extend(list(m))
                    all_covs.extend(list(c))
                    all_colors.extend(list(col))
                    all_amplitudes.extend([-1.0] * len(m))
                    new_means_neg = m
                    new_covs_neg  = c
                    for mi, ci, coli in zip(m, c, col):
                        gaussian_log_rows.append(
                            _cov_to_gaussian_row(it + 1, cum_gid, mi, ci, coli, -1.0)
                        )
                        cum_gid += 1

            # ---- Apply correction ------------------------------------------ #
            # Clamp each pixel's correction to its own residual bounds so the
            # correction can never overshoot I_gt — guaranteeing that the
            # per-pixel residual magnitude is non-increasing every iteration.
            I_delta = np.clip(
                step_size * (I_pos_corr - I_neg_corr),
                -I_res_neg,   # can't darken beyond the negative residual here
                I_res_pos,    # can't brighten beyond the positive residual here
            )
            I_current = np.clip(
                I_current + I_delta,
                0.0, 1.0,
            ).astype(np.float32)
            # Use actual count from fitters, not the budget request (n_available),
            # since fit_em_to_distribution can return None or cap n_components.
            n_gaussians_total += n_added_pos + n_added_neg

            it_metrics = evaluate_metrics(
                I_current, gt_np, n_components=n_gaussians_total,
                compute_lpips_flag=config['metrics'].get('compute_lpips', False))
            it_metrics['l1']   = float(np.mean(np.abs(I_current - gt_np)))
            it_metrics['loss'] = ((1.0 - lambda_dssim) * it_metrics['l1']
                                  + lambda_dssim * it_metrics['dssim'])
            iter_records.append({
                'iteration':     it + 1,
                'n_pos':         n_pos,
                'n_neg':         n_neg,
                'mean_residual': mean_abs_res,
                **it_metrics,
                # Override the 'n_gaussians' key from compression metrics so the
                # authoritative running total takes precedence in this record.
                'n_gaussians':   n_gaussians_total,
            })
            print(f"       PSNR={it_metrics['psnr']:.2f} dB  "
                  f"SSIM={it_metrics['ssim']:.4f}  "
                  f"RMSE={it_metrics['rmse']:.4f}  "
                  f"Loss={it_metrics['loss']:.4f}")

            # Residual snapshot for post-run diagnostic grid
            residual_snapshots.append(np.mean(np.abs(gt_np - I_current), axis=2))
            snapshot_iters.append(it + 1)

            # ---- Save video frame ------------------------------------------ #
            if vis_enabled:
                render_frame_paths.append(
                    save_render_frame(
                        frame_dir_render, it + 1,
                        gt_np, I_current, gt_np - I_current,
                        it_metrics, n_gaussians_total, it_metrics['loss'],
                    )
                )

        # ------------------------------------------------------------------ #
        # Final evaluation & output                                           #
        # ------------------------------------------------------------------ #
        total_time    = time.time() - t_total
        final_metrics = evaluate_metrics(
            I_current, gt_np, n_components=n_gaussians_total,
            compute_lpips_flag=config['metrics'].get('compute_lpips', False))
        final_metrics['l1']   = float(np.mean(np.abs(I_current - gt_np)))
        final_metrics['loss'] = ((1.0 - lambda_dssim) * final_metrics['l1']
                                 + lambda_dssim * final_metrics['dssim'])

        print(f"\n  Final: PSNR={final_metrics['psnr']:.2f} dB  "
              f"SSIM={final_metrics['ssim']:.4f}  "
              f"Loss={final_metrics['loss']:.4f}  "
              f"Gaussians={n_gaussians_total}  time={total_time:.1f}s")

        if exp_cfg.get('save_renders', True):
            Image.fromarray((I_current * 255).astype(np.uint8)).save(
                output_dir / f'{img_path.stem}_hybrid_render.png')

        if exp_cfg.get('save_plots', True):
            stem = img_path.stem
            # 3-panel comparison for each render variant
            save_comparison_3panel(
                output_dir / f'{stem}_hybrid_comparison.png',
                gt_np, I_current,
                f'Hybrid Residual ({n_gaussians_total} Gaussians)',
                final_metrics,
            )
            save_comparison_3panel(
                output_dir / f'{stem}_init_comparison.png',
                gt_np, I_init,
                f'Initial EM ({n_init} Gaussians)',
                init_metrics,
            )
            if compare_pure_em and I_pure is not None:
                save_comparison_3panel(
                    output_dir / f'{stem}_pure_em_comparison.png',
                    gt_np, I_pure,
                    f'Pure EM ({n_total} Gaussians)',
                    pure_em_metrics,
                )

        if iter_records:
            pd.DataFrame(iter_records).to_csv(
                output_dir / f'{img_path.stem}_iterations.csv', index=False)

        # ---- Gaussian parameter log --------------------------------------- #
        if exp_cfg.get('save_raw_data', True) and gaussian_log_rows:
            pd.DataFrame(gaussian_log_rows).to_csv(
                output_dir / f'{img_path.stem}_gaussian_log.csv', index=False)

        # ---- Post-run diagnostic summary plots ---------------------------- #
        if exp_cfg.get('save_plots', True):
            stem = img_path.stem
            save_diag_residuals(
                output_dir / f'{stem}_diag_residuals.png',
                residual_snapshots, snapshot_iters,
            )
            save_diag_gaussians(
                output_dir / f'{stem}_diag_gaussians.png',
                all_means, all_covs, all_amplitudes, isize,
            )
            if iter_records:
                # Build baseline reference points so the diagnostic shows
                # Init EM and Pure EM on the SAME Gaussian-budget axis as the
                # Hybrid curve — an apples-to-apples quality comparison.
                diag_baselines = []
                _bl_keys = ('rmse', 'psnr', 'ssim', 'loss')
                if all(k in init_metrics for k in _bl_keys):
                    diag_baselines.append({
                        'label':       f'Init EM ({n_init} G)',
                        'n_gaussians': n_init,
                        **{k: init_metrics[k] for k in _bl_keys},
                        'style':       {'color': '#9b9ea2', 'marker': 'D'},
                    })
                if compare_pure_em and pure_em_metrics is not None and \
                        all(k in pure_em_metrics for k in _bl_keys):
                    diag_baselines.append({
                        'label':       f'Pure EM ({n_total} G)',
                        'n_gaussians': n_total,
                        **{k: pure_em_metrics[k] for k in _bl_keys},
                        'style':       {'color': '#f4a261', 'marker': 's'},
                    })
                save_diag_metrics(
                    output_dir / f'{stem}_diag_metrics.png',
                    iter_records,
                    baselines=diag_baselines,
                )

        # ---- Assemble video ----------------------------------------------- #
        if vis_enabled and render_frame_paths:
            print(f"  Assembling video ({len(render_frame_paths)} frames) ...")
            stem = img_path.stem
            if vis_make_mp4:
                assemble_mp4(
                    render_frame_paths,
                    output_dir / f'{stem}_evolution.mp4',
                    fps=vis_fps,
                )
            if vis_make_gif:
                assemble_gif(
                    render_frame_paths,
                    output_dir / f'{stem}_evolution.gif',
                    fps=vis_fps,
                )

        # Save the signed Gaussian set for external use / inspection
        if exp_cfg.get('save_raw_data', True):
            np.savez_compressed(
                output_dir / f'{img_path.stem}_gaussians.npz',
                means      = np.array(all_means),
                covs       = np.array(all_covs),
                colors     = np.array(all_colors),
                amplitudes = np.array(all_amplitudes),
                image_size = np.array([h, w]),
            )

        row.update({
            'n_init':            n_init,
            'n_gaussians_final': n_gaussians_total,
            'n_iterations':      len(iter_records),
            'total_time':        total_time,
            'init_psnr':         init_metrics['psnr'],
            'init_ssim':         init_metrics['ssim'],
            **{f'hybrid_{k}': v for k, v in final_metrics.items()},
        })
        all_results.append(row)

    # ---------------------------------------------------------------------- #
    # Aggregate summary                                                       #
    # ---------------------------------------------------------------------- #
    if not all_results:
        print("No results to summarise.")
        return

    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / 'results.csv', index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    show_cols = ['image', 'n_gaussians_final', 'init_psnr', 'hybrid_psnr',
                 'hybrid_ssim', 'total_time']
    if compare_pure_em:
        show_cols = ['image', 'n_gaussians_final',
                     'pure_em_psnr', 'init_psnr', 'hybrid_psnr',
                     'hybrid_ssim', 'total_time']
    avail = [c for c in show_cols if c in df.columns]
    print(df[avail].to_string(index=False))

    print(f"\nDone. Results saved to: {output_dir}")
    print("=" * 70 + "\n")
    return output_dir


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Hybrid EM Gaussian Splatting with Residual Refinement')
    parser.add_argument('--config', default='config.yml',
                        help='Path to configuration YAML file')
    args = parser.parse_args()
    run_hybrid_residual_refinement(args.config)
