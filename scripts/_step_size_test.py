"""Step-size sensitivity check for honest (no oracle) mode."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.em_utils import load_config, render_gaussians, fit_em_to_distribution
from utils.metrics_utils import evaluate_metrics
from run_hybrid_residual import _initial_em_fit
from PIL import Image

cfg = load_config('config.yml')
cfg['hybrid_residual']['n_total'] = 128
cfg['hybrid_residual']['n_init'] = 16
cfg['hybrid_residual']['n_per_iter'] = 8
cfg['hybrid_residual']['max_iter'] = 14
cfg['dataset']['image_size'] = None
em_cfg = cfg['em']

img = Image.open('kodak/kodim03.png').convert('RGB')
gt_np = np.array(img).astype(np.float32) / 255.0
h, w = gt_np.shape[:2]

I_init, *_ = _initial_em_fit(gt_np, 16, em_cfg, 'minibatch')
m0 = evaluate_metrics(I_init, gt_np, compute_lpips_flag=False)
print(f"Init PSNR={m0['psnr']:.2f}")

for step_size in [0.3, 0.5, 1.0]:
    I_oracle = I_init.copy()
    I_honest = I_init.copy()
    n_g = 16
    print(f"\n--- step_size={step_size} ---")
    for it in range(14):
        I_res = gt_np - I_oracle
        I_res_pos = np.maximum(I_res, 0).astype(np.float32)
        I_res_neg = np.maximum(-I_res, 0).astype(np.float32)
        n_avail = min(8, 128 - n_g)
        if n_avail <= 0:
            break
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
                I_pos_corr = render_gaussians(m, c, col, wt, (h, w))
                n_added += len(m)
        if neg_mass > 1e-8 and n_neg >= 1:
            np2 = fit_em_to_distribution(I_res_neg, n_neg, em_cfg, use_minibatch=True)
            if np2 is not None:
                m, c, col, wt = np2
                I_neg_corr = render_gaussians(m, c, col, wt, (h, w))
                n_added += len(m)
        n_g += n_added
        raw_delta = step_size * (I_pos_corr - I_neg_corr)
        I_delta_o = np.clip(raw_delta, -I_res_neg, I_res_pos)
        I_oracle = np.clip(I_oracle + I_delta_o, 0, 1).astype(np.float32)
        I_honest = np.clip(I_honest + raw_delta, 0, 1).astype(np.float32)
        mo = evaluate_metrics(I_oracle, gt_np, compute_lpips_flag=False)
        mh = evaluate_metrics(I_honest, gt_np, compute_lpips_flag=False)
        print(f"  Iter {it+1:2d} n_g={n_g:3d}  oracle_PSNR={mo['psnr']:.2f}  honest_PSNR={mh['psnr']:.2f}")

print("\nDone.")
