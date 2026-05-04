"""Quick evaluation of completed fullbatch paper-comparison runs."""
import pandas as pd
from pathlib import Path

RUNS = {
    128:  "results/paper_comparison/per_run/fullbatch_k128/hybrid_residual_20260503_061838/results.csv",
    256:  "results/paper_comparison/per_run/fullbatch_k256/hybrid_residual_20260503_070145/results.csv",
    512:  "results/paper_comparison/per_run/fullbatch_k512/hybrid_residual_20260503_083018/results.csv",
    1024: "results/paper_comparison/per_run/fullbatch_k1024/hybrid_residual_20260503_114547/results.csv",
}

TILED = {
    128:  "results/tiled_fullbatch_128/tiled_em_20260502_163908/results.csv",
    256:  "results/tiled_fullbatch_256/tiled_em_20260502_164523/results.csv",
    512:  "results/tiled_fullbatch_512/tiled_em_20260502_165051/results.csv",
    1024: "results/tiled_fullbatch_1024/tiled_em_20260502_165724/results.csv",
    2048: "results/tiled_fullbatch_2048/tiled_em_20260502_170702/results.csv",
}

# ── Three-way summary table ──────────────────────────────────────────────────
print(f"{'K':>6}  {'EM one-shot':>11}  {'EM SSIM':>8}  "
      f"{'Hybrid FB':>10}  {'Hyb SS':>8}  {'delta':>7}  "
      f"{'Tiled EM':>9}  {'T-SSIM':>8}")
print("─" * 90)
rows = []
for K, path in RUNS.items():
    df  = pd.read_csv(path)
    tdf = pd.read_csv(TILED[K])
    em_p  = df.pure_em_psnr.mean()
    em_s  = df.pure_em_ssim.mean()
    hy_p  = df.hybrid_psnr.mean()
    hy_s  = df.hybrid_ssim.mean()
    ti_p  = tdf.composite_psnr.mean()
    ti_s  = tdf.composite_ssim.mean()
    rows.append(dict(K=K, n=len(df),
                     em_psnr=em_p, em_ssim=em_s,
                     hybrid_psnr=hy_p, hybrid_ssim=hy_s,
                     delta_psnr=hy_p - em_p,
                     tiled_psnr=ti_p, tiled_ssim=ti_s))
    print(f"{K:>6}  {em_p:>11.3f}  {em_s:>8.4f}  "
          f"{hy_p:>10.3f}  {hy_s:>8.4f}  {hy_p-em_p:>+7.3f}  "
          f"{ti_p:>9.3f}  {ti_s:>8.4f}")

# Also print tiled K=2048 row (no hybrid counterpart)
tdf = pd.read_csv(TILED[2048])
print(f"{'2048':>6}  {'(no run)':>11}             "
      f"{'(no run)':>10}            {'':>7}  "
      f"{tdf.composite_psnr.mean():>9.3f}  {tdf.composite_ssim.mean():>8.4f}")

# ── Per-image  K=1024 ────────────────────────────────────────────────────────
print("\nPer-image detail  K=1024")
df  = pd.read_csv(RUNS[1024])
tdf = pd.read_csv(TILED[1024])
tdf = tdf.set_index('image')
print(f"  {'image':22s}  {'EM':>7}  {'Hybrid':>7}  {'delta':>7}  {'Tiled':>7}")
for _, r in df.iterrows():
    d = r.hybrid_psnr - r.pure_em_psnr
    ti = tdf.loc[r['image'], 'composite_psnr']
    print(f"  {r['image']:22s}  {r.pure_em_psnr:>7.2f}  {r.hybrid_psnr:>7.2f}  {d:>+7.2f}  {ti:>7.2f}")

# ── Save summary ─────────────────────────────────────────────────────────────
out = Path("results/paper_comparison")
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / "summary_fullbatch.csv", index=False)
print(f"\nSaved: {out / 'summary_fullbatch.csv'}")
