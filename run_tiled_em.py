"""
run_tiled_em.py — Tile-based EM Gaussian Splatting with post-hoc gradient refinement.

Algorithm overview
------------------
1. Divide the input image into fixed-size tiles (e.g. 64×64).
2. For each tile independently (in parallel):
   a. Run full-batch EM initialisation (n_init_per_tile Gaussians).
   b. Run iterative residual refinement (≤ max_iter iterations, no minibatch).
   c. Return rendered tile + Gaussian parameters in LOCAL tile-pixel coords.
3. Transform Gaussian means from tile-local → global pixel coordinates
   (covariances need no transformation — pixels are isotropic).
4. Merge tile renders into the full composite image.
5. Evaluate initial composite metrics (PSNR/SSIM/RMSE).
6. [Optional] Global gradient-descent refinement:
   - Differentiable torch renderer parameterised on all Gaussians jointly.
   - Loss: (1-λ)·L1 + λ·dSSIM (optionally down-weighted inside tiles,
     up-weighted at tile boundaries to target block artefacts).
   - A few hundred steps at a small learning rate.
7. Final evaluation; save renders, diagnostics, results CSV.

Coordinate convention
---------------------
Tile-local pixel coords:  (x,  y)  ∈ [0, tile_w) × [0, tile_h)
Global pixel coords:       (X,  Y)  = (x + x0, y + y0)
  where (y0, x0) is the top-left corner of the tile in the full image.

Gaussian means from EM are computed as::

    means_px = gmm.means_[:, :2] * np.array([[tile_w, tile_h]])

i.e. in tile-local pixel space.  The global transform is simply::

    means_global = means_local + np.array([[x0, y0]])

Covariances::

    covs_px = gmm.covariances_[:, :2, :2] * [[tile_w², tile_w·tile_h],
                                              [tile_w·tile_h, tile_h²]]

are already in units of (pixel)² and require no further scaling when
placed in the global image.

Configuration (tiled_em section of config YAML)
-----------------------------------------------
tile_h          — tile height in pixels                           [64]
tile_w          — tile width  in pixels                           [64]
n_total         — total Gaussian budget across ALL tiles          [1024]
n_init_fraction — fraction of per-tile budget used for init EM   [0.25]
n_per_iter      — residuals Gaussians added per iteration/tile   [auto]
max_iter        — max residual iterations per tile               [14]
residual_threshold — per-tile early-stop threshold               [0.001]
step_size       — residual correction damping                    [0.5]
n_workers       — parallel workers (0 = sequential)             [4]
gd:
  enabled       — run global GD refinement                       [true]
  n_steps       — gradient descent steps                         [150]
  lr_means      — learning rate for Gaussian centres             [0.5]
  lr_colors     — learning rate for Gaussian colours             [0.005]
  lr_scales     — learning rate for log-scale params             [0.002]
  lambda_dssim  — dSSIM weight in loss (mirrors GS config)       [0.2]
  boundary_weight — extra weight for tile-boundary pixels        [3.0]
  border_px     — pixels from tile edge counted as boundary      [3]
  log_interval  — print loss every N steps (0 = silent)         [25]
compare_full_image — also run full-image hybrid for comparison   [true]
"""

from __future__ import annotations

import time
import yaml
import multiprocessing as mp
import concurrent.futures
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image


from utils.metrics_utils   import evaluate_metrics
from utils.dataset_utils   import select_images
from utils.em_utils        import (
    load_config, download_kodak_dataset, _prepare_data,
    render_gaussians, render_gaussians_signed, fit_em_to_distribution,
)
from utils.tiling_utils    import (
    compute_tile_bboxes, extract_tiles, merge_tiles,
    local_to_global_means,
    boundary_mask, tile_grid_overlay,
    save_tile_diagnostic, save_boundary_comparison,
)
from utils.gd_utils        import (
    build_gd_params, render_gaussians_torch, gradient_descent_refinement,
    _TORCH_AVAILABLE as _GD_TORCH_AVAILABLE,
)


# ===========================================================================
# Per-tile pipeline  (must be importable at module level for ProcessPoolExecutor)
# ===========================================================================

def _initial_em_fit_tile(tile_np: np.ndarray, n_init: int, em_cfg: dict):
    """
    Full-batch EM initialisation on one tile.

    Normalises coordinates to [0,1] within the tile, fits a GMM, then
    transforms the GMM means back to tile-local pixel coordinates.
    Covariances are kept in tile-local pixel² units.

    Args:
        tile_np : (th, tw, 3) float32 image segment.
        n_init  : Number of Gaussian components.
        em_cfg  : EM configuration dict (covariance_type, max_iter, …).

    Returns:
        (render, means_px, covs_px, colors, weights) or raises on failure.
    """
    from sklearn.mixture import GaussianMixture
    th, tw = tile_np.shape[:2]
    data_5d, _, _ = _prepare_data(tile_np)  # normalised in tile space

    n_comp = min(n_init, max(1, len(data_5d) // 5))

    gmm = GaussianMixture(
        n_components   = n_comp,
        covariance_type= em_cfg.get('covariance_type', 'full'),
        max_iter       = em_cfg.get('max_iter', 100),
        init_params    = 'kmeans',
        random_state   = 42,
        reg_covar      = 1e-5,
    )
    gmm.fit(data_5d)

    # Tile-local pixel-space means (x_tile, y_tile)
    means_px = gmm.means_[:, :2] * np.array([[tw, th]], dtype=np.float32)
    # Jacobian J = diag(tw, th):  Σ_px = J Σ_norm J^T
    cov_scale = np.array([[[tw**2, tw * th], [tw * th, th**2]]])
    covs_px   = gmm.covariances_[:, :2, :2] * cov_scale
    colors    = np.clip(gmm.means_[:, 2:5], 0.0, 1.0)
    weights   = gmm.weights_

    render = render_gaussians(means_px, covs_px, colors, weights, (th, tw))
    return render, means_px, covs_px, colors, weights


def _run_tile_pipeline(
    tile_np  : np.ndarray,
    k_tile   : int,
    em_cfg   : dict,
    hr_cfg   : dict,
) -> dict:
    """
    Run full-batch EM + iterative residual refinement on a single tile.

    No minibatch sampling is used at any stage (use_minibatch=False).

    Returns a dict with keys:
      render        — (tile_h, tile_w, 3) float32 final render
      means_local   — (K, 2) Gaussian means in tile-local pixel coords
      covs          — (K, 2, 2) covariance matrices (pixel² units)
      colors        — (K, 3)
      amplitudes    — (K,)  +1 for positive, -1 for negative Gaussians
      weights       — (K,)  mixture weights (positive Gaussians only; -1 amps get weight 1/K)
      n_gaussians   — total Gaussians used
      iters_run     — number of residual iterations executed
      init_psnr     — PSNR after initial EM (dB)
      final_psnr    — PSNR after residual refinement (dB)
      tile_h, tile_w — tile dimensions
    """
    from utils.metrics_utils import evaluate_metrics

    th, tw = tile_np.shape[:2]

    n_init_frac        = hr_cfg.get('n_init_fraction', 0.25)
    n_init             = max(1, round(k_tile * n_init_frac))
    _npi               = hr_cfg.get('n_per_iter_per_tile')  # None when YAML null
    n_per_iter         = max(1, k_tile // 8) if _npi is None else int(_npi)
    max_iter           = hr_cfg.get('max_iter', 14)
    res_thr     = hr_cfg.get('residual_threshold', 0.001)
    step_size   = float(hr_cfg.get('step_size', 0.5))

    # ---- Step 1: initial EM ------------------------------------------------ #
    I_cur, means, covs, colors, weights = _initial_em_fit_tile(tile_np, n_init, em_cfg)

    init_m = evaluate_metrics(I_cur, tile_np, n_components=n_init,
                               compute_lpips_flag=False)
    init_psnr = init_m['psnr']

    n_used  = n_init
    all_means, all_covs, all_colors, all_amps, all_wts = (
        list(means), list(covs), list(colors),
        [1.0] * n_init, list(weights),
    )

    # ---- Steps 2–5: residual refinement ------------------------------------ #
    for it in range(max_iter):
        if n_used >= k_tile:
            break
        I_res        = tile_np - I_cur
        mean_abs_res = float(np.mean(np.abs(I_res)))
        if mean_abs_res < res_thr:
            break

        I_res_pos = np.maximum( I_res, 0.0).astype(np.float32)
        I_res_neg = np.maximum(-I_res, 0.0).astype(np.float32)

        n_avail   = min(n_per_iter, k_tile - n_used)
        pos_mass  = float(I_res_pos.sum())
        neg_mass  = float(I_res_neg.sum())
        if n_avail < 2:
            n_pos = n_avail if pos_mass >= neg_mass else 0
            n_neg = n_avail - n_pos
        else:
            pos_frac = pos_mass / (pos_mass + neg_mass + 1e-10)
            n_pos = max(1, round(n_avail * pos_frac))
            n_neg = n_avail - n_pos
            n_neg = max(1, n_neg)
            n_pos = n_avail - n_neg

        I_pos_corr = np.zeros_like(tile_np)
        if pos_mass > 1e-8 and n_pos >= 1:
            pp = fit_em_to_distribution(I_res_pos, n_pos, em_cfg, use_minibatch=False)
            if pp is not None:
                m, c, col, wt = pp
                I_pos_corr = render_gaussians_signed(
                    m, c, col, np.ones(len(m), dtype=np.float32), (th, tw))
                all_means.extend(list(m)); all_covs.extend(list(c))
                all_colors.extend(list(col)); all_amps.extend([1.0] * len(m))
                all_wts.extend(list(wt))
                n_used += len(m)

        I_neg_corr = np.zeros_like(tile_np)
        if neg_mass > 1e-8 and n_neg >= 1:
            np_ = fit_em_to_distribution(I_res_neg, n_neg, em_cfg, use_minibatch=False)
            if np_ is not None:
                m, c, col, wt = np_
                I_neg_corr = render_gaussians_signed(
                    m, c, col, np.ones(len(m), dtype=np.float32), (th, tw))
                all_means.extend(list(m)); all_covs.extend(list(c))
                all_colors.extend(list(col)); all_amps.extend([-1.0] * len(m))
                all_wts.extend(list(wt))
                n_used += len(m)

        I_cur = np.clip(I_cur + step_size * (I_pos_corr - I_neg_corr), 0.0, 1.0)

    final_m    = evaluate_metrics(I_cur, tile_np, n_components=n_used,
                                  compute_lpips_flag=False)
    final_psnr = final_m['psnr']

    return {
        'render'      : I_cur,
        'means_local' : np.array(all_means),
        'covs'        : np.array(all_covs),
        'colors'      : np.array(all_colors),
        'amplitudes'  : np.array(all_amps),
        'weights'     : np.array(all_wts),
        'n_gaussians' : n_used,
        'iters_run'   : it + 1 if n_used > n_init else 0,
        'init_psnr'   : init_psnr,
        'final_psnr'  : final_psnr,
        'tile_h'      : th,
        'tile_w'      : tw,
    }


def _tile_worker(args: tuple) -> dict:
    """
    Top-level worker function for ProcessPoolExecutor.

    Wraps ``_run_tile_pipeline`` so it can be dispatched to a subprocess.
    All arguments must be pickle-serialisable (numpy arrays and plain dicts).

    Args:
        args : (tile_idx, tile_np, k_tile, em_cfg, hr_cfg)

    Returns:
        Dict from ``_run_tile_pipeline`` with an added ``tile_idx`` key.
    """
    tile_idx, tile_np, k_tile, em_cfg, hr_cfg = args
    result = _run_tile_pipeline(tile_np, k_tile, em_cfg, hr_cfg)
    result['tile_idx'] = tile_idx
    return result





# ===========================================================================
# Main entry point
# ===========================================================================

def run_tiled_em(config_path: str = 'config_tiled.yml') -> Path | None:
    """
    Run tile-based EM Gaussian splatting with optional global GD refinement.

    Reads all settings from the ``tiled_em`` section of *config_path*.
    """
    config     = load_config(config_path)
    tiled_cfg  = config.get('tiled_em', {})

    if not tiled_cfg.get('enabled', True):
        print("Tiled EM disabled in config.")
        return None

    tile_h       = int(tiled_cfg.get('tile_h', 64))
    tile_w       = int(tiled_cfg.get('tile_w', 64))
    n_total      = int(tiled_cfg.get('n_total', 1024))
    max_iter     = int(tiled_cfg.get('max_iter', 14))
    res_thr      = float(tiled_cfg.get('residual_threshold', 0.001))
    step_size    = float(tiled_cfg.get('step_size', 0.5))
    n_init_frac  = float(tiled_cfg.get('n_init_fraction', 0.25))
    n_per_iter_cfg = tiled_cfg.get('n_per_iter_per_tile', None)
    n_workers    = int(tiled_cfg.get('n_workers', 4))
    gd_cfg       = tiled_cfg.get('gd', {})
    gd_enabled   = bool(gd_cfg.get('enabled', True)) and _GD_TORCH_AVAILABLE

    exp_cfg  = config.get('experiment', {})
    em_cfg   = config['em']

    # Per-tile HR sub-config passed to workers
    hr_cfg = {
        'n_init_fraction'  : n_init_frac,
        'n_per_iter_per_tile': n_per_iter_cfg,  # None → auto inside worker
        'max_iter'         : max_iter,
        'residual_threshold': res_thr,
        'step_size'        : step_size,
    }

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(exp_cfg.get('output_dir', './outputs')) / f"tiled_em_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print("TILED EM GAUSSIAN SPLATTING")
    print(f"  tile=({tile_h}×{tile_w})  N_total={n_total}  "
          f"n_init_fraction={n_init_frac:.2f}  max_iter={max_iter}")
    print(f"  n_workers={n_workers}  gd_enabled={gd_enabled}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    with open(output_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    kodak_dir   = download_kodak_dataset(config['dataset']['path'])
    image_paths = select_images(kodak_dir, config['dataset'])
    print(f"Images: {len(image_paths)}")

    lambda_dssim  = config.get('gaussian_splatting', {}).get('lambda_dssim', 0.2)
    compute_lpips = config.get('metrics', {}).get('compute_lpips', False)

    all_results: list[dict] = []

    for img_idx, img_path in enumerate(image_paths, 1):
        print(f"\n[{img_idx}/{len(image_paths)}] {img_path.name}")

        img        = Image.open(img_path).convert('RGB')
        image_size = config['dataset'].get('image_size')
        if image_size:
            img = img.resize(tuple(image_size), Image.LANCZOS)
        gt_np = np.array(img).astype(np.float32) / 255.0
        H, W  = gt_np.shape[:2]

        t_img = time.time()

        # ------------------------------------------------------------------ #
        # Tile layout                                                         #
        # ------------------------------------------------------------------ #
        bboxes  = compute_tile_bboxes(H, W, tile_h, tile_w)
        n_tiles_total = len(bboxes)
        # Budget: floor division; any remainder goes to the last tile
        k_base   = n_total // n_tiles_total
        k_rem    = n_total  - k_base * n_tiles_total
        k_budget = [k_base + (1 if i < k_rem else 0)
                    for i in range(n_tiles_total)]

        print(f"  Tiles: {n_tiles_total} ({H//tile_h + (1 if H%tile_h else 0)} rows × "
              f"{W//tile_w + (1 if W%tile_w else 0)} cols), "
              f"K_per_tile ≈ {k_base} (total {n_total})")

        tiles_np = extract_tiles(gt_np, bboxes)

        # ------------------------------------------------------------------ #
        # Parallel tile processing                                            #
        # ------------------------------------------------------------------ #
        worker_args = [
            (i, tiles_np[i], k_budget[i], em_cfg, hr_cfg)
            for i in range(n_tiles_total)
        ]

        tile_results: list[dict] = [None] * n_tiles_total  # type: ignore[list-item]
        t_tile = time.time()

        if n_workers <= 1:
            # Sequential (easier to debug)
            for args in worker_args:
                r = _tile_worker(args)
                tile_results[r['tile_idx']] = r
        else:
            ctx = mp.get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx
            ) as exe:
                futures = {exe.submit(_tile_worker, a): a[0] for a in worker_args}
                for fut in concurrent.futures.as_completed(futures):
                    r = fut.result()
                    tile_results[r['tile_idx']] = r

        t_tiles_done = time.time() - t_tile
        print(f"  Tile processing done in {t_tiles_done:.1f}s")

        # ------------------------------------------------------------------ #
        # Merge: tile → global coordinate transform + composite image        #
        # ------------------------------------------------------------------ #
        rendered_tiles = [r['render'] for r in tile_results]
        composite      = merge_tiles(rendered_tiles, bboxes, gt_np.shape)
        composite      = np.clip(composite, 0.0, 1.0)

        # Build global Gaussian arrays from all tile results
        all_means_g   : list[np.ndarray] = []
        all_covs_g    : list[np.ndarray] = []
        all_colors_g  : list[np.ndarray] = []
        all_amps_g    : list[np.ndarray] = []
        all_weights_g : list[np.ndarray] = []

        for r, bbox in zip(tile_results, bboxes):
            means_global = local_to_global_means(r['means_local'], bbox)
            all_means_g.append(means_global)
            all_covs_g.append(r['covs'])
            all_colors_g.append(r['colors'])
            all_amps_g.append(r['amplitudes'])
            all_weights_g.append(r['weights'])

        all_means_global   = np.concatenate(all_means_g,   axis=0)
        all_covs_global    = np.concatenate(all_covs_g,    axis=0)
        all_colors_global  = np.concatenate(all_colors_g,  axis=0)
        all_amps_global    = np.concatenate(all_amps_g,    axis=0)
        all_weights_global = np.concatenate(all_weights_g, axis=0)

        n_total_used = int(sum(r['n_gaussians'] for r in tile_results))

        # Evaluate composite
        m_composite = evaluate_metrics(composite, gt_np,
                                        n_components=n_total_used,
                                        compute_lpips_flag=compute_lpips)
        m_composite['l1'] = float(np.mean(np.abs(composite - gt_np)))
        m_composite['loss'] = ((1 - lambda_dssim) * m_composite['l1']
                               + lambda_dssim * m_composite['dssim'])

        print(f"  Composite: PSNR={m_composite['psnr']:.2f} dB  "
              f"SSIM={m_composite['ssim']:.4f}  "
              f"RMSE={m_composite['rmse']:.4f}  "
              f"(K_total={n_total_used})")

        # ------------------------------------------------------------------ #
        # Gradient-descent refinement                                        #
        # ------------------------------------------------------------------ #
        I_refined     = None
        m_refined     = None
        ref_means     = all_means_global
        ref_covs      = all_covs_global
        ref_colors    = all_colors_global
        ref_weights   = all_weights_global
        t_gd = 0.0

        if gd_enabled:
            print(f"  Gradient descent refinement …")
            t0 = time.time()

            # Effective amplitudes: amp_sign * gmm_weight.
            #   Initial Gaussians      : amp=+1, wt=w_k  →  eff=+w_k
            #   Positive residual      : amp=+1, wt=w_k  →  eff=+w_k
            #   Negative residual      : amp=-1, wt=w_k  →  eff=-w_k
            # This matches the additive signed renderer, no color flipping needed.
            eff_amps = all_amps_global * all_weights_global

            try:
                refined_render, ref_means, ref_covs, ref_colors, ref_weights = \
                    gradient_descent_refinement(
                        all_means_global, all_covs_global, all_colors_global,
                        eff_amps,
                        gt_np, composite,   # composite passed as starting point
                        bboxes, gd_cfg,
                    )
                I_refined = refined_render
                t_gd      = time.time() - t0

                m_refined = evaluate_metrics(I_refined, gt_np,
                                              n_components=n_total_used,
                                              compute_lpips_flag=compute_lpips)
                m_refined['l1'] = float(np.mean(np.abs(I_refined - gt_np)))
                m_refined['loss'] = ((1 - lambda_dssim) * m_refined['l1']
                                     + lambda_dssim * m_refined['dssim'])
                delta_p = m_refined['psnr'] - m_composite['psnr']
                sign    = '+' if delta_p >= 0 else ''
                print(f"  After GD: PSNR={m_refined['psnr']:.2f} dB "
                      f"({sign}{delta_p:.2f})  "
                      f"SSIM={m_refined['ssim']:.4f}  "
                      f"time={t_gd:.1f}s")
            except Exception as exc:
                print(f"  WARNING GD failed: {exc}")
                I_refined = None
        elif not _GD_TORCH_AVAILABLE and gd_cfg.get('enabled', True):
            print("  WARNING: torch not available — GD refinement skipped.")

        # ------------------------------------------------------------------ #
        # Per-tile diagnostics                                                #
        # ------------------------------------------------------------------ #
        per_tile_rows = []
        for r, bbox in zip(tile_results, bboxes):
            per_tile_rows.append({
                'tile_idx'  : r['tile_idx'],
                'y0': bbox[0], 'x0': bbox[1], 'y1': bbox[2], 'x1': bbox[3],
                'k_budget'  : k_budget[r['tile_idx']],
                'n_gaussians': r['n_gaussians'],
                'iters_run' : r['iters_run'],
                'init_psnr' : r['init_psnr'],
                'final_psnr': r['final_psnr'],
                'tile_h'    : r['tile_h'],
                'tile_w'    : r['tile_w'],
            })
        pd.DataFrame(per_tile_rows).to_csv(
            output_dir / f'{img_path.stem}_tile_stats.csv', index=False)

        # ------------------------------------------------------------------ #
        # Save renders                                                        #
        # ------------------------------------------------------------------ #
        if exp_cfg.get('save_renders', True):
            Image.fromarray((composite * 255).astype(np.uint8)).save(
                output_dir / f'{img_path.stem}_composite.png')
            if I_refined is not None:
                Image.fromarray((I_refined * 255).astype(np.uint8)).save(
                    output_dir / f'{img_path.stem}_refined.png')
            # Tile grid overlay for reference
            Image.fromarray(
                (tile_grid_overlay(gt_np, bboxes) * 255).astype(np.uint8)
            ).save(output_dir / f'{img_path.stem}_tile_grid.png')

        # ------------------------------------------------------------------ #
        # Diagnostic figures                                                  #
        # ------------------------------------------------------------------ #
        if exp_cfg.get('save_plots', True):
            save_tile_diagnostic(
                output_dir / f'{img_path.stem}_diagnostic.png',
                gt_np, composite, I_refined, bboxes,
                m_composite, m_refined,
            )
            save_boundary_comparison(
                output_dir / f'{img_path.stem}_boundary.png',
                gt_np, composite, I_refined, bboxes,
            )

        # ------------------------------------------------------------------ #
        # Persist Gaussian set                                                #
        # ------------------------------------------------------------------ #
        if exp_cfg.get('save_raw_data', True):
            np.savez_compressed(
                output_dir / f'{img_path.stem}_gaussians.npz',
                means      = ref_means,
                covs       = ref_covs,
                colors     = ref_colors,
                weights    = ref_weights,
                amplitudes = all_amps_global,
                image_size = np.array([H, W]),
            )

        total_time = time.time() - t_img
        row: dict = {
            'image'          : img_path.name,
            'H'              : H,
            'W'              : W,
            'n_tiles'        : n_tiles_total,
            'tile_h'         : tile_h,
            'tile_w'         : tile_w,
            'n_total_used'   : n_total_used,
            'tile_time'      : t_tiles_done,
            'gd_time'        : t_gd,
            'total_time'     : total_time,
            **{f'composite_{k}': v for k, v in m_composite.items()},
        }
        if m_refined is not None:
            row.update({f'refined_{k}': v for k, v in m_refined.items()})
        all_results.append(row)

    # ---------------------------------------------------------------------- #
    # Aggregate summary                                                       #
    # ---------------------------------------------------------------------- #
    if not all_results:
        print("No results collected.")
        return None

    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / 'results.csv', index=False)

    print(f"\n{'='*70}")
    print("TILED EM SUMMARY")
    print(f"{'='*70}")

    show = ['image', 'n_tiles', 'n_total_used',
            'composite_psnr', 'composite_ssim',
            'tile_time', 'total_time']
    if 'refined_psnr' in df.columns:
        show = ['image', 'n_tiles', 'n_total_used',
                'composite_psnr', 'refined_psnr', 'composite_ssim',
                'refined_ssim', 'tile_time', 'gd_time', 'total_time']
    avail = [c for c in show if c in df.columns]
    print(df[avail].to_string(index=False))

    print(f"\nResults saved to: {output_dir}")
    print(f"{'='*70}\n")
    return output_dir


if __name__ == '__main__':
    import argparse
    # ProcessPoolExecutor on Windows requires spawn context and this guard.
    mp.freeze_support()
    parser = argparse.ArgumentParser(description='Tiled EM Gaussian Splatting')
    parser.add_argument('--config', default='config_tiled.yml',
                        help='Path to configuration YAML')
    args = parser.parse_args()
    run_tiled_em(args.config)
