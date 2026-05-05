import csv, math, os

def mean(xs): return sum(xs)/len(xs) if xs else 0

# ── Standard EM (from existing paper_comparison summary) ──────────────
print("=== Standard EM (fullbatch, from paper_comparison) ===")
pc = list(csv.DictReader(open('results/paper_comparison/summary_fullbatch.csv')))
for r in pc:
    print(f"  K={r['K']}: {float(r['em_psnr']):.3f} dB")

# ── EM Minibatch ───────────────────────────────────────────────────────
mb_paths = {
    128:  'results/em_fullkodak_k128/em_20260504_074917/results.csv',
    256:  'results/em_fullkodak_k256/em_20260504_075234/results.csv',
    512:  'results/em_fullkodak_k512/em_20260504_080240/results.csv',
    1024: 'results/em_fullkodak_k1024/em_20260504_081525/results.csv',
    2048: 'results/em_fullkodak_k2048/em_20260504_084109/results.csv',
}
print("\n=== EM Minibatch ===")
for k, path in mb_paths.items():
    rows = list(csv.DictReader(open(path)))
    psnrs = [float(r['psnr']) for r in rows]
    times = [float(r['fit_time']) for r in rows]
    print(f"  K={k}: mean_PSNR={mean(psnrs):.3f} dB  avg_time={mean(times):.1f}s  n={len(rows)}")

# ── Hybrid K=2048 ──────────────────────────────────────────────────────
print("\n=== Hybrid Residual K=2048 ===")
hpath = 'results/hybrid_fullkodak_k2048/hybrid_residual_20260504_100305/results.csv'
hrows = list(csv.DictReader(open(hpath)))
h_psnrs = [float(r['hybrid_psnr']) for r in hrows]
h_times = [float(r['total_time']) for r in hrows]
outlier = 'kodim21.png'
h_excl = [(p,t) for p,t,r in zip(h_psnrs,h_times,hrows) if r['image']!=outlier]
outlier_row = next(r for r in hrows if r['image']==outlier)
print(f"  Mean PSNR (all 24):       {mean(h_psnrs):.3f} dB")
print(f"  Mean PSNR (excl kodim21): {mean([x[0] for x in h_excl]):.3f} dB")
print(f"  kodim21 PSNR:             {float(outlier_row['hybrid_psnr']):.3f} dB  (time={float(outlier_row['total_time'])/3600:.2f}h — POWER OUTLIER)")
print(f"  Avg time/image (excl kodim21): {mean([x[1] for x in h_excl])/60:.1f} min")
print(f"  Per-image PSNR:")
for r in hrows:
    flag = ' ← OUTLIER time' if r['image']==outlier else ''
    print(f"    {r['image']}: {float(r['hybrid_psnr']):.3f} dB  t={float(r['total_time'])/60:.1f}min{flag}")

# ── Hybrid K=1024 (existing) ───────────────────────────────────────────
print("\n=== Hybrid Residual K=1024 (existing, from paper_comparison) ===")
for r in pc:
    print(f"  K={r['K']}: hybrid_psnr={float(r['hybrid_psnr']):.3f} dB")

# ── GS Baseline ───────────────────────────────────────────────────────
gs_paths = {
    128:  'results/gs_fullkodak_k128/2dgs_20260504_074116/results.csv',
    256:  'results/gs_fullkodak_k256/2dgs_20260504_122425/results.csv',
    512:  'results/gs_fullkodak_k512/2dgs_20260504_142519/results.csv',
}
print("\n=== 2D Gaussian Splatting (GS) ===")
for k, path in gs_paths.items():
    rows = list(csv.DictReader(open(path)))
    psnrs = [float(r['psnr']) for r in rows]
    times = [float(r['total_time']) for r in rows]
    print(f"  K={k}: mean_PSNR={mean(psnrs):.3f} dB  avg_time={mean(times):.1f}s  n={len(rows)}")

print("\n  K=1024 (partial, kodim01-12 from history.csv):")
import glob
k1024_dir = 'results/gs_fullkodak_k1024/2dgs_20260504_163449'
hist_files = sorted(glob.glob(k1024_dir + '/*_history.csv'))
psnrs_1024 = []
for f in hist_files:
    rows = list(csv.DictReader(open(f)))
    if rows:
        mse = float(rows[-1]['mse'])
        psnr = -10*math.log10(mse) if mse > 0 else 99.0
        psnrs_1024.append(psnr)
        print(f"    {os.path.basename(f).replace('_history.csv','')}: {psnr:.3f} dB")
print(f"  Partial mean (n={len(psnrs_1024)}): {mean(psnrs_1024):.3f} dB")
