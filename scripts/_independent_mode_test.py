"""
Test honest (independent residual) vs oracle mode on kodim03.
Both modes now compute their own residuals independently.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.em_utils import (
    load_config, render_gaussians, render_residual_correction,
    fit_em_to_distribution,
)
from utils.metrics_utils import evaluate_metrics
from run_hybrid_residual import _initial_em_fit
from PIL import Image

cfg = load_config('config.yml')
em_cfg = cfg['em']

img = Image.open('kodak/kodim03.png').convert('RGB')
gt_np = np.array(img).astype(np.float32) / 255.0
h, w = gt_np.shape[:2]
print(f"Image size: {h}x{w}")

I_init, *_ = _initial_em_fit(gt_np, 16, em_cfg, 'minibatch')
m0 = evaluate_metrics(I_init, gt_np, compute_lpips_flag=False)
print(f"Init PSNR={m0['psnr']:.2f}")

def run_mode(I_start, honest: bool, n_total=128, n_per_iter=8):
    I_cur = I_start.copy()
    n_g = 16
    print(f"\n{'HONEST (render_residual_correction)' if honest else 'ORACLE (render_gaussians + clamp)'}")
    for it in range(14):
        n_avail = min(n_per_iter, n_total - n_g)
        if n_avail <= 0:
            break
        # Each mode uses its OWN current image's residual
        I_res = gt_np - I_cur
        I_res_pos = np.maximum(I_res, 0).astype(np.float32)
        I_res_neg = np.maximum(-I_res, 0).astype(np.float32)
        pos_mass = float(I_res_pos.sum())
        neg_mass = float(I_res_neg.sum())
        pos_frac = pos_mass / (pos_mass + neg_mass + 1e-10)
        n_pos = max(1, round(n_avail * pos_frac))
        n_neg = max(1, n_avail - n_pos)
        n_pos = n_avail - n_neg

        I_pos_corr = np.zeros_like(gt_np)
        I_neg_corr = np.zeros_like(gt_np)
        n_added = 0

        if pos_mass > 1e-8 and n_pos >= 1:
            pp = fit_em_to_distribution(I_res_pos, n_pos, em_cfg, use_minibatch=True)
            if pp is not None:
                m, c, col, wt = pp
                if honest:
                    I_pos_corr = render_residual_correction(I_res_pos, m, c, wt, (h, w))
                else:
                    I_pos_corr = render_gaussians(m, c, col, wt, (h, w))
                n_added += len(m)

        if neg_mass > 1e-8 and n_neg >= 1:
            np2 = fit_em_to_distribution(I_res_neg, n_neg, em_cfg, use_minibatch=True)
            if np2 is not None:
                m, c, col, wt = np2
                if honest:
                    I_neg_corr = render_residual_correction(I_res_neg, m, c, wt, (h, w))
                else:
                    I_neg_corr = render_gaussians(m, c, col, wt, (h, w))
                n_added += len(m)

        n_g += n_added

        if honest:
            # No oracle: correction bounded by construction
            I_cur = np.clip(I_cur + I_pos_corr - I_neg_corr, 0, 1).astype(np.float32)
        else:
            # Oracle: clamp to residual bounds
            raw_delta = I_pos_corr - I_neg_corr
            I_cur = np.clip(I_cur + np.clip(raw_delta, -I_res_neg, I_res_pos), 0, 1).astype(np.float32)

        m_it = evaluate_metrics(I_cur, gt_np, compute_lpips_flag=False)
        print(f"  Iter {it+1:2d} n_g={n_g:3d}  PSNR={m_it['psnr']:.2f}  RMSE={m_it['rmse']:.4f}")

    return I_cur

I_oracle = run_mode(I_init, honest=False)
I_honest = run_mode(I_init, honest=True)

m_o = evaluate_metrics(I_oracle, gt_np, compute_lpips_flag=False)
m_h = evaluate_metrics(I_honest, gt_np, compute_lpips_flag=False)
m0  = evaluate_metrics(I_init, gt_np, compute_lpips_flag=False)
print(f"\nSummary (kodim03, 128 Gaussians):")
print(f"  Init   PSNR={m0['psnr']:.2f}  SSIM={m0['ssim']:.4f}")
print(f"  Oracle PSNR={m_o['psnr']:.2f}  SSIM={m_o['ssim']:.4f}  (DATA LEAKAGE)")
print(f"  Honest PSNR={m_h['psnr']:.2f}  SSIM={m_h['ssim']:.4f}  (TRUE QUALITY)")
print(f"  Inflation: +{m_o['psnr'] - m_h['psnr']:.2f} dB from oracle clamping")
