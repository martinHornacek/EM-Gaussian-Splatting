"""
Generate all paper figures:
  fig_psnr_vs_k.pdf   — PSNR vs K curve: EM one-shot + Tiled EM
  fig_visual.pdf      — 3-panel: GT / EM K=1024 / Tiled EM K=1024 + residual strip
  fig_tiling.pdf      — Tiled EM tile-grid overlay example (shows tiling structure)

Run from the repo root:  python scripts/generate_paper_figures.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

OUT = Path("paper_elmar/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── colour / style constants ──────────────────────────────────────────────────
EM_COLOR       = "#2166ac"   # blue
TILED_COLOR    = "#d6604d"   # red-orange
MINIBATCH_COLOR = "#33a02c"  # green
HYBRID_COLOR    = "#762a83"  # purple
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})

# ── Data ─────────────────────────────────────────────────────────────────────
EM_RUNS = {
    128:  "results/paper_comparison/per_run/fullbatch_k128/hybrid_residual_20260503_061838/results.csv",
    256:  "results/paper_comparison/per_run/fullbatch_k256/hybrid_residual_20260503_070145/results.csv",
    512:  "results/paper_comparison/per_run/fullbatch_k512/hybrid_residual_20260503_083018/results.csv",
    1024: "results/paper_comparison/per_run/fullbatch_k1024/hybrid_residual_20260503_114547/results.csv",
}
TILED_RUNS = {
    128:  "results/tiled_fullbatch_128/tiled_em_20260502_163908/results.csv",
    256:  "results/tiled_fullbatch_256/tiled_em_20260503_202322/results.csv",
    512:  "results/tiled_fullbatch_512/tiled_em_20260502_165051/results.csv",
    1024: "results/tiled_fullbatch_1024/tiled_em_20260502_165724/results.csv",
    2048: "results/tiled_fullbatch_2048/tiled_em_20260502_170702/results.csv",
}

# Mini-batch EM: find latest run in each K directory
def _latest_csv(base_dir):
    runs = sorted(Path(base_dir).glob("*/results.csv"))
    if not runs:
        raise FileNotFoundError(f"No results.csv found under {base_dir}")
    return str(runs[-1])

MINIBATCH_RUNS = {
    128:  _latest_csv("results/em_fullkodak_k128"),
    256:  _latest_csv("results/em_fullkodak_k256"),
    512:  _latest_csv("results/em_fullkodak_k512"),
    1024: _latest_csv("results/em_fullkodak_k1024"),
    2048: _latest_csv("results/em_fullkodak_k2048"),
}

em_k, em_psnr, em_ssim = [], [], []
for K, p in EM_RUNS.items():
    df = pd.read_csv(p)
    em_k.append(K);  em_psnr.append(df.pure_em_psnr.mean());  em_ssim.append(df.pure_em_ssim.mean())

ti_k, ti_psnr, ti_ssim = [], [], []
for K, p in TILED_RUNS.items():
    df = pd.read_csv(p)
    ti_k.append(K);  ti_psnr.append(df.composite_psnr.mean());  ti_ssim.append(df.composite_ssim.mean())

mb_k, mb_psnr, mb_ssim, mb_time = [], [], [], []
for K, p in MINIBATCH_RUNS.items():
    df = pd.read_csv(p)
    df = df[df.variant == "minibatch"]
    mb_k.append(K);  mb_psnr.append(df.psnr.mean());  mb_ssim.append(df.ssim.mean())
    mb_time.append(df.fit_time.mean())

# Hybrid data: K=128-1024 from summary_fullbatch.csv; K=2048 from hybrid_fullkodak_k2048
_hyb_summary = pd.read_csv("results/paper_comparison/summary_fullbatch.csv")
hyb_k    = list(_hyb_summary["K"].astype(int))
hyb_psnr = list(_hyb_summary["hybrid_psnr"].astype(float))
_hyb2048 = pd.read_csv(_latest_csv("results/hybrid_fullkodak_k2048"))
hyb_k.append(2048);  hyb_psnr.append(_hyb2048.hybrid_psnr.mean())

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — PSNR vs K  (4 methods)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 2.6))

ax.plot(em_k,  em_psnr,  "o-",  color=EM_COLOR,        lw=1.4, ms=5, label="EM one-shot")
ax.plot(mb_k,  mb_psnr,  "D:",  color=MINIBATCH_COLOR,  lw=1.4, ms=4, label="Mini-batch EM")
ax.plot(ti_k,  ti_psnr,  "s--", color=TILED_COLOR,      lw=1.4, ms=5, label="Tiled EM")
ax.plot(hyb_k, hyb_psnr, "^-.", color=HYBRID_COLOR,     lw=1.2, ms=4, label="Hybrid (residual)")

all_k = sorted(set(em_k) | set(ti_k) | set(mb_k) | set(hyb_k))
ax.set_xscale("log", base=2)
ax.set_xticks(all_k)
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.set_xlabel("Number of Gaussians $K$")
ax.set_ylabel("Mean PSNR (dB)")
ax.set_title("Reconstruction quality on Kodak (24 images)")
ax.legend(loc="lower right", fontsize=7)
ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
ax.set_ylim(16, 26)

fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_psnr_vs_k.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_psnr_vs_k.png", bbox_inches="tight", dpi=200)
print(f"  Saved: {OUT / 'fig_psnr_vs_k.pdf'}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Visual comparison:  GT | EM K=1024 | Tiled EM K=1024 | residual
# Use kodim15 (lighthouse — visually rich)
# ═══════════════════════════════════════════════════════════════════════════════
IMG = "kodim15"
gt_path   = f"kodak/{IMG}.png"
em_path   = f"results/paper_comparison/per_run/fullbatch_k1024/hybrid_residual_20260503_114547/{IMG}_pure_em_render.png"
tiled_path= f"results/tiled_fullbatch_1024/tiled_em_20260502_165724/{IMG}_composite.png"

gt    = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
em_r  = np.array(Image.open(em_path)).astype(np.float32) / 255.0
ti_r  = np.array(Image.open(tiled_path)).astype(np.float32) / 255.0

def _resize_to(arr, h, w):
    """Resize HxWxC float32 array to (h, w)."""
    if arr.shape[:2] == (h, w):
        return arr
    img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    img = img.resize((w, h), Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0

H, W = gt.shape[:2]
em_r  = _resize_to(em_r,  H, W)
ti_r  = _resize_to(ti_r,  H, W)

res_em    = np.abs(gt - em_r).mean(axis=2)
res_tiled = np.abs(gt - ti_r).mean(axis=2)

def psnr(a, b):
    return -10 * np.log10(np.mean((a - b) ** 2))

em_psnr_img    = psnr(gt, em_r)
tiled_psnr_img = psnr(gt, ti_r)

fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.1))
panels = [
    (gt,         "Ground truth",                    None, 1.0),
    (em_r,       f"EM, $K\\!=\\!1024$\n({em_psnr_img:.2f} dB)", None, 1.0),
    (ti_r,       f"Tiled EM, $K\\!=\\!1024$\n({tiled_psnr_img:.2f} dB)", None, 1.0),
    (res_em,     "Abs. residual (EM)",               "hot", 0.12),
]
for ax, (img, title, cmap, vmax) in zip(axes, panels):
    if cmap:
        ax.imshow(img, cmap=cmap, vmin=0, vmax=vmax)
    else:
        ax.imshow(np.clip(img, 0, 1))
    ax.set_title(title, fontsize=7.5, pad=2)
    ax.axis("off")

fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_visual.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_visual.png", bbox_inches="tight", dpi=200)
print(f"  Saved: {OUT / 'fig_visual.pdf'}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Tiled EM structure: side-by-side composite + tile grid overlay
# ═══════════════════════════════════════════════════════════════════════════════
grid_path = f"results/tiled_fullbatch_1024/tiled_em_20260502_165724/{IMG}_tile_grid.png"
grid_r = np.array(Image.open(grid_path)).astype(np.float32) / 255.0
grid_r = _resize_to(grid_r, H, W)

fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.0))
for ax, (img, title) in zip(axes, [
    (gt,     "Ground truth"),
    (ti_r,   f"Tiled EM composite\n({tiled_psnr_img:.2f} dB)"),
    (grid_r, "Tile boundaries"),
]):
    ax.imshow(np.clip(img, 0, 1))
    ax.set_title(title, fontsize=7.5, pad=2)
    ax.axis("off")

fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_tiling.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_tiling.png", bbox_inches="tight", dpi=200)
print(f"  Saved: {OUT / 'fig_tiling.pdf'}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — SSIM vs K (supplemental / optional)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.plot(em_k, em_ssim, "o-", color=EM_COLOR,    lw=1.4, ms=5, label="EM (one-shot)")
ax.plot(ti_k, ti_ssim, "s--", color=TILED_COLOR, lw=1.4, ms=5, label="Tiled EM")
ax.set_xscale("log", base=2)
ax.set_xticks(sorted(set(em_k) | set(ti_k)))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
ax.set_xlabel("Number of Gaussians $K$")
ax.set_ylabel("Mean SSIM")
ax.set_title("Structural similarity on Kodak (24 images)")
ax.legend(loc="lower right")
ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_ssim_vs_k.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_ssim_vs_k.png", bbox_inches="tight", dpi=200)
print(f"  Saved: {OUT / 'fig_ssim_vs_k.pdf'}")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Mini-batch EM: fitting time vs PSNR trade-off
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(3.5, 2.4))

ax.plot(mb_time, mb_psnr, "o-", color=MINIBATCH_COLOR, lw=1.4, ms=6)
for t, p, k in zip(mb_time, mb_psnr, mb_k):
    ax.annotate(f"$K\\!=\\!{k}$", xy=(t, p),
                xytext=(6, -2), textcoords="offset points",
                fontsize=7, color=MINIBATCH_COLOR)

ax.set_xscale("log")
ax.set_xlabel("Mean fitting time per image (s)")
ax.set_ylabel("Mean PSNR (dB)")
ax.set_title("Mini-batch EM: speed–quality trade-off")
ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)

fig.tight_layout(pad=0.4)
fig.savefig(OUT / "fig_speed_quality.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_speed_quality.png", bbox_inches="tight", dpi=200)
print(f"  Saved: {OUT / 'fig_speed_quality.pdf'}")

print("\nAll figures generated.")
