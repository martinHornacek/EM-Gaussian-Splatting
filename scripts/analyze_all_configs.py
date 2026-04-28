"""
Comprehensive four-way comparison:
  1024 / 2048 Gaussians  ×  Baseline (fullbatch) / Minibatch
Outputs:
  - Printed summary tables
  - results/hybrid_kodak_eval_combined/ (plots + CSVs)
"""
import os, glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, "results", "hybrid_kodak_eval_combined")
os.makedirs(OUT, exist_ok=True)

CFGS = {
    "1024 Baseline": os.path.join(ROOT, "outputs", "hybrid_residual_1024_gaussians_baseline"),
    "1024 Minibatch": os.path.join(ROOT, "outputs", "hybrid_residual_1024_gaussians_minibatch"),
    "2048 Baseline": os.path.join(ROOT, "outputs", "hybrid_residual_2048_gaussians_baseline"),
    "2048 Minibatch": os.path.join(ROOT, "outputs", "hybrid_residual_2048_gaussians_minibatch"),
}
COLORS = {
    "1024 Baseline": "#2196F3",
    "1024 Minibatch": "#03A9F4",
    "2048 Baseline": "#FF5722",
    "2048 Minibatch": "#FF9800",
}
MARKERS = {
    "1024 Baseline": "o",
    "1024 Minibatch": "s",
    "2048 Baseline": "^",
    "2048 Minibatch": "D",
}

# ── load results ──────────────────────────────────────────────────────────────
dfs = {k: pd.read_csv(os.path.join(v, "results.csv")) for k, v in CFGS.items()}

# ── 1. Aggregate stats table ──────────────────────────────────────────────────
print("=" * 90)
print("AGGREGATE PERFORMANCE — Hybrid Residual Refinement (24-image Kodak set)")
print("=" * 90)
print(f"{'Config':<22}  {'PSNR':>8}  {'SSIM':>8}  {'RMSE':>8}  {'LPIPS':>8}  "
      f"{'Time(s)':>8}  {'PSNR±σ':>10}  {'SSIM±σ':>10}")
print("-" * 90)
agg_rows = []
for name, df in dfs.items():
    row = {
        "config":    name,
        "psnr_mean": df.hybrid_psnr.mean(),
        "psnr_std":  df.hybrid_psnr.std(),
        "psnr_min":  df.hybrid_psnr.min(),
        "psnr_max":  df.hybrid_psnr.max(),
        "ssim_mean": df.hybrid_ssim.mean(),
        "ssim_std":  df.hybrid_ssim.std(),
        "rmse_mean": df.hybrid_rmse.mean(),
        "rmse_std":  df.hybrid_rmse.std(),
        "lpips_mean":df.hybrid_lpips.mean(),
        "lpips_std": df.hybrid_lpips.std(),
        "time_mean": df.total_time.mean(),
        "time_total":df.total_time.sum(),
    }
    agg_rows.append(row)
    print(f"{name:<22}  {row['psnr_mean']:>8.3f}  {row['ssim_mean']:>8.4f}  "
          f"{row['rmse_mean']:>8.4f}  {row['lpips_mean']:>8.4f}  "
          f"{row['time_mean']:>8.1f}  "
          f"{row['psnr_mean']:.2f}±{row['psnr_std']:.2f}  "
          f"{row['ssim_mean']:.4f}±{row['ssim_std']:.4f}")

agg_df = pd.DataFrame(agg_rows)
agg_df.to_csv(os.path.join(OUT, "aggregate_stats_all4.csv"), index=False)

# ── 2. Per-image PSNR table ───────────────────────────────────────────────────
print()
print("=" * 90)
print("PER-IMAGE PSNR (dB)")
print("=" * 90)
imgs = dfs["1024 Baseline"]["image"].values
print(f"{'Image':<14}  {'1024-Base':>10}  {'1024-Mini':>10}  {'2048-Base':>10}  "
      f"{'2048-Mini':>10}  {'Best Config':>16}  {'Init PSNR':>10}")
print("-" * 90)
per_image = []
for img in imgs:
    rows = {k: df[df.image == img].iloc[0] for k, df in dfs.items()}
    psnrs = {k: float(rows[k].hybrid_psnr) for k in dfs}
    best  = max(psnrs, key=psnrs.get)
    init_psnr = float(rows["1024 Baseline"].init_psnr)
    print(f"{img:<14}  {psnrs['1024 Baseline']:>10.3f}  {psnrs['1024 Minibatch']:>10.3f}  "
          f"{psnrs['2048 Baseline']:>10.3f}  {psnrs['2048 Minibatch']:>10.3f}  "
          f"{best:>16}  {init_psnr:>10.3f}")
    per_image.append({"image": img, **{k: psnrs[k] for k in dfs},
                      "best": best, "init_psnr": init_psnr})

pd.DataFrame(per_image).to_csv(os.path.join(OUT, "per_image_psnr_all4.csv"), index=False)

# ── 3. Δ comparisons ──────────────────────────────────────────────────────────
print()
print("=" * 90)
print("DELTA ANALYSIS")
print("=" * 90)
df1b = dfs["1024 Baseline"].set_index("image")
df1m = dfs["1024 Minibatch"].set_index("image")
df2b = dfs["2048 Baseline"].set_index("image")
df2m = dfs["2048 Minibatch"].set_index("image")

d_mini_1024 = df1m.hybrid_psnr - df1b.hybrid_psnr
d_mini_2048 = df2m.hybrid_psnr - df2b.hybrid_psnr
d_scale_base = df2b.hybrid_psnr - df1b.hybrid_psnr
d_scale_mini = df2m.hybrid_psnr - df1m.hybrid_psnr

speedup_1024 = df1b.total_time.mean() / df1m.total_time.mean()
speedup_2048 = df2b.total_time.mean() / df2m.total_time.mean()

print(f"  Minibatch vs Baseline PSNR gain @ 1024G: {d_mini_1024.mean():+.3f} ± {d_mini_1024.std():.3f} dB")
print(f"  Minibatch vs Baseline PSNR gain @ 2048G: {d_mini_2048.mean():+.3f} ± {d_mini_2048.std():.3f} dB")
print(f"  2048 vs 1024 PSNR gain (Baseline):       {d_scale_base.mean():+.3f} ± {d_scale_base.std():.3f} dB")
print(f"  2048 vs 1024 PSNR gain (Minibatch):      {d_scale_mini.mean():+.3f} ± {d_scale_mini.std():.3f} dB")
print()
print(f"  Timing speedup Minibatch/Baseline @ 1024G: {speedup_1024:.1f}x")
print(f"  Timing speedup Minibatch/Baseline @ 2048G: {speedup_2048:.1f}x")
print()
print(f"  Quality/Time tradeoff @ 1024G: +{d_mini_1024.mean():.2f} dB for {speedup_1024:.1f}x speedup")
print(f"  Quality/Time tradeoff @ 2048G: +{d_mini_2048.mean():.2f} dB for {speedup_2048:.1f}x speedup")

# fraction of images where minibatch beats baseline
pct_1024 = (d_mini_1024 > 0).mean() * 100
pct_2048 = (d_mini_2048 > 0).mean() * 100
print(f"\n  Minibatch > Baseline on {pct_1024:.0f}% of images @ 1024G")
print(f"  Minibatch > Baseline on {pct_2048:.0f}% of images @ 2048G")

# ── 4. Hardest / easiest images ───────────────────────────────────────────────
print()
print("=" * 90)
print("HARDEST / EASIEST IMAGES (by 2048 Minibatch PSNR)")
print("=" * 90)
ranked = df2m["hybrid_psnr"].sort_values()
print("  Hardest 5:")
for img, v in ranked.head(5).items():
    print(f"    {img}: {v:.3f} dB")
print("  Easiest 5:")
for img, v in ranked.tail(5).items():
    print(f"    {img}: {v:.3f} dB")

# ── 5. Convergence curves (kodim15 — benchmark image) ────────────────────────
print()
print("=" * 90)
print("CONVERGENCE (kodim15 — each iteration adds Gaussians)")
print("=" * 90)
iter_data = {}
for key, path in CFGS.items():
    fp = os.path.join(path, "kodim15_iterations.csv")
    if os.path.exists(fp):
        iter_data[key] = pd.read_csv(fp)

for key, idf in iter_data.items():
    final = idf.iloc[-1]
    print(f"  {key:<22}  iter1 PSNR={idf.iloc[0].psnr:.2f} dB  "
          f"final PSNR={final.psnr:.2f} dB  "
          f"gain={final.psnr - idf.iloc[0].psnr:.2f} dB")

# ── 6. PLOTS ──────────────────────────────────────────────────────────────────

# --- 6a. Bar chart: mean PSNR/SSIM/RMSE/LPIPS across configs ------------------
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
metrics_plot = [
    ("psnr_mean", "PSNR (dB)", False),
    ("ssim_mean", "SSIM", False),
    ("rmse_mean", "RMSE", False),
    ("lpips_mean", "LPIPS", False),
]
names  = list(dfs.keys())
colors = [COLORS[n] for n in names]
for ax, (col, label, _) in zip(axes, metrics_plot):
    vals = [agg_df.loc[agg_df.config == n, col].values[0] for n in names]
    errs = [agg_df.loc[agg_df.config == n, col.replace("mean", "std")].values[0] for n in names]
    bars = ax.bar(range(len(names)), vals, color=colors, width=0.6, yerr=errs,
                  capsize=5, error_kw={"elinewidth": 1.5})
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax.set_ylabel(label)
    ax.set_title(label)
    # annotate
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

fig.suptitle("Hybrid Residual Refinement — 24-image Kodak Aggregate Metrics", fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "aggregate_bar_chart.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- 6b. Per-image PSNR line plot ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 5))
x = np.arange(len(imgs))
for name, df in dfs.items():
    psnrs = [float(df[df.image == img].iloc[0].hybrid_psnr) for img in imgs]
    ax.plot(x, psnrs, marker=MARKERS[name], color=COLORS[name], label=name,
            linewidth=1.5, markersize=5)

ax.set_xticks(x)
ax.set_xticklabels([i.replace(".png", "") for i in imgs], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("PSNR (dB)")
ax.set_title("Per-Image PSNR — All 4 Configurations")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "per_image_psnr.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- 6c. PSNR delta: minibatch vs baseline ────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
x = np.arange(len(imgs))
w = 0.35
ax.bar(x - w/2, d_mini_1024.values, width=w, color="#2196F3", alpha=0.85, label="Δ 1024G (mini−base)")
ax.bar(x + w/2, d_mini_2048.values, width=w, color="#FF5722", alpha=0.85, label="Δ 2048G (mini−base)")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([i.replace(".png", "") for i in imgs], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("ΔPSNR (dB)  [+ve = minibatch better]")
ax.set_title("Minibatch vs Fullbatch PSNR Delta — per image")
ax.legend()
ax.grid(axis="y", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "minibatch_vs_baseline_delta.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- 6d. Convergence curves for kodim15 ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for key, idf in iter_data.items():
    axes[0].plot(idf.n_gaussians, idf.psnr, marker=MARKERS[key], color=COLORS[key],
                 label=key, linewidth=1.8, markersize=5)
    axes[1].plot(idf.n_gaussians, idf.ssim, marker=MARKERS[key], color=COLORS[key],
                 label=key, linewidth=1.8, markersize=5)
axes[0].set_xlabel("# Gaussians"); axes[0].set_ylabel("PSNR (dB)")
axes[0].set_title("Convergence: PSNR vs #Gaussians (kodim15)")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.4)
axes[1].set_xlabel("# Gaussians"); axes[1].set_ylabel("SSIM")
axes[1].set_title("Convergence: SSIM vs #Gaussians (kodim15)")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "convergence_kodim15.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- 6e. Scatter: PSNR vs Time ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for name, df in dfs.items():
    ax.scatter(df.total_time, df.hybrid_psnr, color=COLORS[name],
               marker=MARKERS[name], label=name, s=60, alpha=0.8)
ax.set_xlabel("Total time (s)")
ax.set_ylabel("PSNR (dB)")
ax.set_title("Quality vs. Runtime — per image (all configs)")
ax.legend(fontsize=9)
ax.grid(alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "psnr_vs_time_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()

# --- 6f. Init PSNR vs final PSNR scatter for best config (2048 mini) ---------
fig, ax = plt.subplots(figsize=(7, 6))
init_p  = df2m["init_psnr"].values
final_p = df2m["hybrid_psnr"].values
ax.scatter(init_p, final_p, color="#FF9800", s=70, label="2048 Minibatch")
for i, img in enumerate(imgs):
    ax.annotate(img.replace(".png", ""), (init_p[i], final_p[i]),
                fontsize=5.5, xytext=(3, 3), textcoords="offset points")
lims = [min(init_p.min(), final_p.min()) - 1, max(init_p.max(), final_p.max()) + 1]
ax.plot(lims, lims, "k--", linewidth=0.8, label="baseline (no gain)")
ax.set_xlabel("Init PSNR (dB)")
ax.set_ylabel("Final Hybrid PSNR (dB)")
ax.set_title("Init vs Final PSNR — 2048 Minibatch")
ax.legend(); ax.grid(alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "init_vs_final_psnr.png"), dpi=150, bbox_inches="tight")
plt.close()

# ── Done ──────────────────────────────────────────────────────────────────────
print()
print("=" * 90)
print(f"All outputs saved to:  {OUT}/")
print("  aggregate_bar_chart.png")
print("  per_image_psnr.png")
print("  minibatch_vs_baseline_delta.png")
print("  convergence_kodim15.png")
print("  psnr_vs_time_scatter.png")
print("  init_vs_final_psnr.png")
print("  aggregate_stats_all4.csv")
print("  per_image_psnr_all4.csv")
print("=" * 90)
