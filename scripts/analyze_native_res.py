"""
analyze_native_res.py
Analyze hybrid EM-GS minibatch results at native Kodak resolution.
Computes aggregate stats, JPEG baselines, per-image breakdown, and scaling gains.
"""
import numpy as np
import pandas as pd
import io
from pathlib import Path
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SEP  = '=' * 72
SEP2 = '-' * 72

BASE = Path(__file__).parent.parent / 'outputs'
KODAK = Path(__file__).parent.parent / 'kodak'

configs = {
    128:  BASE / 'hybrid_residual_fullkodak_128gaussians_minibatch' / 'results.csv',
    256:  BASE / 'hybrid_residual_fullkodak_256gaussians_minibatch' / 'results.csv',
    512:  BASE / 'hybrid_residual_fullkodak_512gaussians_minibatch' / 'results.csv',
    1024: BASE / 'hybrid_residual_fullkodak_1024gaussians_minibatch' / 'results.csv',
    2048: BASE / 'hybrid_residual_fullkodak_2048gaussians_minibatch' / 'results.csv',
}
dfs = {k: pd.read_csv(v) for k, v in configs.items()}

# ── 1. Aggregate rate-distortion table ────────────────────────────────────
print(SEP)
print('HYBRID EM-GS MINIBATCH — NATIVE KODAK RESOLUTION (768x512)')
print('Rate-distortion: K in {128, 256, 512, 1024, 2048}')
print(SEP)
header = f"{'K':>5}  {'bpp':>6}  {'CR':>6}  {'PSNR':>7} {'±':>1}{'std':>4}  {'SSIM':>7} {'±':>1}{'std':>5}  {'LPIPS':>7} {'±':>1}{'std':>5}  {'s/img':>7}"
print(header)
print(SEP2)
for k, df in dfs.items():
    bpp   = df['hybrid_bits_per_pixel'].iloc[0]
    cr    = df['hybrid_compression_ratio'].iloc[0]
    psnr  = df['hybrid_psnr'].mean();  pstd = df['hybrid_psnr'].std()
    ssim  = df['hybrid_ssim'].mean();  sstd = df['hybrid_ssim'].std()
    lpips = df['hybrid_lpips'].mean(); lstd = df['hybrid_lpips'].std()
    ttime = df['total_time'].mean()
    print(f"{k:>5}  {bpp:>6.4f}  {cr:>5.0f}x  {psnr:>7.2f} ±{pstd:>4.2f}  {ssim:>7.4f} ±{sstd:>5.4f}  {lpips:>7.4f} ±{lstd:>5.4f}  {ttime:>7.1f}")

# ── 2. JPEG baseline at matching bpp ─────────────────────────────────────
print()
print(SEP)
print('JPEG BASELINE at matched bpp (full 24-image Kodak, native resolution)')
print(SEP)

kodak_imgs = sorted(KODAK.glob('kodim*.png'))
target_bpps = {128: 0.09375, 256: 0.1875, 512: 0.375, 1024: 0.75, 2048: 1.50}

jpeg_results = {}
for K, target_bpp in target_bpps.items():
    psnrs, ssims, actual_bpps = [], [], []
    for img_path in kodak_imgs:
        img_pil = Image.open(img_path).convert('RGB')
        h, w = img_pil.size[1], img_pil.size[0]
        n_pixels = h * w
        target_bits = target_bpp * n_pixels
        gt_np = np.array(img_pil).astype(np.float32) / 255.0
        # Find JPEG quality closest to target bpp
        best_q, best_diff = 1, float('inf')
        for q in range(1, 96):
            buf = io.BytesIO()
            img_pil.save(buf, format='JPEG', quality=q)
            diff = abs(buf.tell() * 8 - target_bits)
            if diff < best_diff:
                best_diff = diff
                best_q = q
        buf = io.BytesIO()
        img_pil.save(buf, format='JPEG', quality=best_q)
        actual_bits = buf.tell() * 8
        buf.seek(0)
        recon = np.array(Image.open(buf).convert('RGB')).astype(np.float32) / 255.0
        psnrs.append(peak_signal_noise_ratio(gt_np, recon, data_range=1.0))
        ssims.append(structural_similarity(gt_np, recon, channel_axis=2, data_range=1.0))
        actual_bpps.append(actual_bits / n_pixels)
    jpeg_results[K] = dict(
        psnr_mean=np.mean(psnrs), psnr_std=np.std(psnrs),
        ssim_mean=np.mean(ssims), ssim_std=np.std(ssims),
        bpp_mean=np.mean(actual_bpps)
    )

header2 = f"{'K':>5}  {'target bpp':>10}  {'JPEG bpp':>8}  {'JPEG PSNR':>9} {'±':>1}{'std':>4}  {'JPEG SSIM':>9} {'±':>1}{'std':>5}"
print(header2)
print(SEP2)
for K, r in jpeg_results.items():
    print(f"{K:>5}  {target_bpps[K]:>10.4f}  {r['bpp_mean']:>8.4f}  {r['psnr_mean']:>9.2f} ±{r['psnr_std']:>4.2f}  {r['ssim_mean']:>9.4f} ±{r['ssim_std']:>5.4f}")

# ── 3. Delta: Hybrid vs JPEG ──────────────────────────────────────────────
print()
print(SEP)
print('DELTA: Hybrid PSNR minus JPEG PSNR at matched bpp')
print(SEP)
header3 = f"{'K':>5}  {'bpp':>6}  {'Hybrid PSNR':>11}  {'JPEG PSNR':>9}  {'Delta':>7}  Verdict"
print(header3)
print(SEP2)
for K, df in dfs.items():
    h_psnr = df['hybrid_psnr'].mean()
    j_psnr = jpeg_results[K]['psnr_mean']
    delta  = h_psnr - j_psnr
    bpp    = df['hybrid_bits_per_pixel'].iloc[0]
    verdict = 'ABOVE JPEG' if delta >= 0 else f'{abs(delta):.2f} dB BELOW JPEG'
    print(f"{K:>5}  {bpp:>6.4f}  {h_psnr:>11.2f}  {j_psnr:>9.2f}  {delta:>+7.2f}  {verdict}")

# ── 4. Per-image breakdown at 2048G ──────────────────────────────────────
print()
print(SEP)
print('PER-IMAGE — 2048G (1.50 bpp), sorted by PSNR ascending')
print(SEP)
df2 = dfs[2048].copy()
print(f"{'Image':>12}  {'PSNR':>7}  {'SSIM':>7}  {'LPIPS':>7}  {'Time':>6}")
print(SEP2)
for _, row in df2.sort_values('hybrid_psnr').iterrows():
    print(f"{row['image']:>12}  {row['hybrid_psnr']:>7.2f}  {row['hybrid_ssim']:>7.4f}  {row['hybrid_lpips']:>7.4f}  {row['total_time']:>5.0f}s")

# ── 5. PSNR gain per K-doubling ───────────────────────────────────────────
print()
print(SEP)
print('PSNR SCALING GAIN (mean dB improvement per K doubling)')
print(SEP)
ks    = [128, 256, 512, 1024, 2048]
means = [dfs[k]['hybrid_psnr'].mean() for k in ks]
for i in range(1, len(ks)):
    gain = means[i] - means[i-1]
    print(f"  {ks[i-1]:>4}G -> {ks[i]:>4}G : +{gain:.2f} dB")
print(f"  Total 128G -> 2048G : +{means[-1] - means[0]:.2f} dB")

# ── 6. Hardest / easiest images across all K ─────────────────────────────
print()
print(SEP)
print('HARDEST / EASIEST IMAGES (ranked by 2048G PSNR)')
print(SEP)
ranked = df2.sort_values('hybrid_psnr')[['image', 'hybrid_psnr', 'hybrid_ssim']].reset_index(drop=True)
print('Hardest 5:')
for _, r in ranked.head(5).iterrows():
    print(f"  {r['image']:>12}  PSNR={r['hybrid_psnr']:.2f}  SSIM={r['hybrid_ssim']:.4f}")
print('Easiest 5:')
for _, r in ranked.tail(5).iloc[::-1].iterrows():
    print(f"  {r['image']:>12}  PSNR={r['hybrid_psnr']:.2f}  SSIM={r['hybrid_ssim']:.4f}")

print()
print(SEP)
print('DONE')
print(SEP)
