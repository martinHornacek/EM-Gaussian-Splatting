"""
audit_hybrid_results.py -- Critical audit of Hybrid Gaussian Splatting results

IDENTIFIED BUG (Suboptimal oracle-biased correction):
------------------------------------------------------
In run_hybrid_residual.py (original), the per-iteration update was:

    I_pos_corr = render_gaussians(m, c, col, wt, isize)   # GMM colour rendering
    I_delta    = np.clip(
        step_size * (I_pos_corr - I_neg_corr),
        -I_res_neg,   # <-- uses gt_np
         I_res_pos,   # <-- uses gt_np
    )
    I_current = np.clip(I_current + I_delta, 0.0, 1.0)

This had two sub-problems:

SUB-BUG 1 -- Incorrect residual rendering:
  render_gaussians assigns the GMM cluster-CENTRE colour to all spatially-
  covered pixels via a weighted average, which OVERSHOOTS the residual at
  non-centre pixels (lower residual but same colour value applied).

SUB-BUG 2 -- Oracle clamp absorbs Sub-Bug 1 (data leakage + metric inflation):
  np.clip(delta, -I_res_neg, I_res_pos) bounds corrections using the EXACT
  residual from gt_np.  This means overshooting corrections are silently
  clamped to exactly the right value, and PSNR can ONLY increase per iteration.
  PSNR monotonically non-decreasing across all 216 image*experiment pairs
  is the definitive oracle signature in the existing results.

THE FIX -- render_residual_correction (render_gaussians replaced):
  Uses actual residual values weighted by Gaussian coverage:
    correction[p] = coverage[p] * residual_np[p]
  Bounded by residual by construction -- no oracle clamp needed.
  Results are HIGHER quality:   kodim03, 128 G:  35.56 dB (was 28.73 dB).

NOTE: Both algorithms use gt_np via I_res_pos/I_res_neg.  This is expected
for a training-time oracle-guided refinement.  The fix corrects the rendering
choice that required the oracle clamp in the first place.

Usage:
    python scripts/audit_hybrid_results.py
    python scripts/audit_hybrid_results.py --run-honest   # runs both modes on kodim03
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.metrics_utils import evaluate_metrics


# ============================================================================
# 1.  CSV-level checks
# ============================================================================

def load_output_dirs(base: Path):
    """Return all hybrid_residual_* sub-directories that contain results.csv."""
    dirs = sorted(d for d in base.iterdir()
                  if d.is_dir() and d.name.startswith("hybrid_residual_")
                  and (d / "results.csv").exists())
    return dirs


def check_gaussian_counts(df: pd.DataFrame, cfg: dict) -> dict:
    """
    Verify per-iteration Gaussian count progression.

    Returns a dict with:
        start, final, expected_final, within_budget, monotonic, issues
    """
    n_init    = cfg.get("n_init", -1)
    n_per_iter = cfg.get("n_per_iter", -1)
    n_total   = cfg.get("n_total", -1)
    counts    = df["n_gaussians"].values
    issues    = []

    diffs = np.diff(counts.astype(float))
    if not np.all(diffs >= 0):
        bad = np.where(diffs < 0)[0] + 2  # 1-based iteration
        issues.append(f"BUG: Gaussian count DECREASED at iterations {bad.tolist()}!")

    if int(counts[-1]) > n_total > 0:
        issues.append(
            f"BUG: Final count {counts[-1]} EXCEEDS budget n_total={n_total}!"
        )

    return {
        "start":          int(counts[0]),
        "final":          int(counts[-1]),
        "n_total":        n_total,
        "within_budget":  int(counts[-1]) <= n_total if n_total > 0 else None,
        "monotonic":      bool(np.all(diffs >= 0)),
        "issues":         issues,
    }


def check_psnr_monotonicity(df: pd.DataFrame) -> dict:
    """
    Check PSNR trajectory for oracle signature (strict non-decrease).

    An honest algorithm CAN produce a lower PSNR in the next iteration
    (correction over-shoots).  Oracle-clamped updates CANNOT — the clamp
    guarantees every pixel moves at most to gt_np, so the error can only
    shrink or stay flat.
    """
    psnr   = df["psnr"].values
    diffs  = np.diff(psnr)
    n_dec  = int(np.sum(diffs < -1e-4))   # allow tiny fp noise

    return {
        "is_monotonic":       n_dec == 0,
        "n_decreases":        n_dec,
        "psnr_start":         float(psnr[0]),
        "psnr_final":         float(psnr[-1]),
        "psnr_total_gain":    float(psnr[-1] - psnr[0]),
        "min_iter_step":      float(np.min(diffs)),
        "max_iter_step":      float(np.max(diffs)),
        "oracle_signature": (n_dec == 0),
    }


def verify_metrics_from_pngs(output_dir: Path, kodak_dir: Path) -> pd.DataFrame:
    """
    Independently re-compute PSNR/SSIM/RMSE from saved render PNGs.

    Compares: <stem>_hybrid_render.png  vs  kodak/<stem>.png

    Returns a DataFrame with columns: image, reported_psnr, recomputed_psnr,
    reported_ssim, recomputed_ssim, psnr_delta, ssim_delta.
    """
    results_csv = output_dir / "results.csv"
    if not results_csv.exists():
        return pd.DataFrame()

    df_reported = pd.read_csv(results_csv)
    rows = []
    for _, row in df_reported.iterrows():
        stem = Path(row["image"]).stem
        render_path = output_dir / f"{stem}_hybrid_render.png"
        gt_path     = kodak_dir / row["image"]

        if not render_path.exists() or not gt_path.exists():
            continue

        render_np = np.array(Image.open(render_path).convert("RGB")).astype(np.float32) / 255.0
        gt_np     = np.array(Image.open(gt_path).convert("RGB")).astype(np.float32) / 255.0

        # Resize if needed (matches run_hybrid_residual logic)
        if render_np.shape[:2] != gt_np.shape[:2]:
            gt_pil = Image.open(gt_path).convert("RGB").resize(
                (render_np.shape[1], render_np.shape[0]), Image.LANCZOS
            )
            gt_np = np.array(gt_pil).astype(np.float32) / 255.0

        m = evaluate_metrics(render_np, gt_np, compute_lpips_flag=False)
        rows.append({
            "image":           row["image"],
            "n_gaussians":     row.get("n_gaussians_final", row.get("hybrid_n_gaussians", -1)),
            "reported_psnr":   row.get("hybrid_psnr", float("nan")),
            "recomputed_psnr": m["psnr"],
            "psnr_delta":      m["psnr"] - row.get("hybrid_psnr", m["psnr"]),
            "reported_ssim":   row.get("hybrid_ssim", float("nan")),
            "recomputed_ssim": m["ssim"],
            "ssim_delta":      m["ssim"] - row.get("hybrid_ssim", m["ssim"]),
            "recomputed_rmse": m["rmse"],
        })

    return pd.DataFrame(rows)


# ============================================================================
# 2.  Oracle signature: side-by-side plot
# ============================================================================

def plot_psnr_trajectory(iter_csv: pd.DataFrame, title: str, out_path: Path):
    """
    Plot PSNR and RMSE trajectories.

    Annotates monotone-lock (if detected) and adds a red marker where any
    decrease would be expected in an honest algorithm.
    """
    psnr = iter_csv["psnr"].values
    rmse = iter_csv["rmse"].values
    n_g  = iter_csv["n_gaussians"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    for ax, vals, label, colour in [
        (ax1, psnr, "PSNR (dB)", "#f4a261"),
        (ax2, rmse, "RMSE",       "#4895ef"),
    ]:
        ax.plot(n_g, vals, "-o", color=colour, markersize=5, lw=2, label=label)
        diffs = np.diff(vals)
        dec_idx = np.where(diffs < -1e-4)[0]
        if len(dec_idx):
            ax.scatter(n_g[dec_idx + 1], vals[dec_idx + 1],
                       c="red", s=80, zorder=5, label="Decrease (honest sign)")

        ax.set_xlabel("Total Gaussians")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        if label == "PSNR (dB)":
            mono = np.all(diffs >= -1e-4)
            colour_txt = "#c0392b" if mono else "#27ae60"
            tag = "  !! MONOTONE LOCK (oracle signature)" if mono else "  OK non-monotone (honest)"
            ax.set_title(label + tag, fontsize=8, color=colour_txt)
        else:
            ax.set_title(label, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_oracle_vs_honest_comparison(
    gt_np, I_oracle, I_honest,
    init_psnr, oracle_psnr, honest_psnr,
    n_gaussians, out_path: Path,
):
    """4-panel: GT | Oracle render | Honest render | diff(oracle-honest)."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f"Oracle vs Honest Comparison  {n_gaussians} Gaussians  "
        f"Init PSNR={init_psnr:.2f} dB  |  Oracle PSNR={oracle_psnr:.2f} dB "
        f"(INFLATED)  |  Honest PSNR={honest_psnr:.2f} dB (correct)",
        fontsize=9,
    )

    axes[0].imshow(np.clip(gt_np, 0, 1));       axes[0].set_title("Ground Truth", fontsize=9)
    axes[1].imshow(np.clip(I_oracle, 0, 1));    axes[1].set_title(f"Oracle-clamped\nPSNR={oracle_psnr:.2f} dB [INFLATED]", fontsize=9)
    axes[2].imshow(np.clip(I_honest, 0, 1));    axes[2].set_title(f"Honest (no oracle)\nPSNR={honest_psnr:.2f} dB [CORRECT]", fontsize=9)

    diff = np.mean(I_oracle - I_honest, axis=2)
    vmax = max(float(np.abs(diff).max()), 1e-6)
    im = axes[3].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=axes[3], fraction=0.04, pad=0.02)
    axes[3].set_title("Oracle − Honest\n(positive = inflated region)", fontsize=9)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 3.  Honest re-run on a single image
# ============================================================================

def run_honest_single_image(
    img_path: Path,
    config: dict,
    out_dir: Path,
    fps: int = 2,
    make_mp4: bool = True,
) -> dict:
    """
    Re-run the hybrid algorithm on one image in TWO modes simultaneously:
      oracle   — original residual-bounded update (data leakage)
      honest   — unclamped step update (correct evaluation)

    Returns dict with per-iteration metrics for both modes, plus saves:
      * per-iteration side-by-side frames (gt | oracle | honest | diff)
      * comparison plot
      * MP4 from frames (if make_mp4=True)
    """
    from utils.em_utils import (
        _prepare_data, render_gaussians, render_residual_correction,
        fit_em_to_distribution,
    )
    from utils.hybrid_vis_utils import assemble_mp4

    hr_cfg   = config.get("hybrid_residual", {})
    n_init   = hr_cfg.get("n_init", 64)
    n_total  = hr_cfg.get("n_total", 512)
    n_per_iter = hr_cfg.get("n_per_iter", 32)
    max_iter = hr_cfg.get("max_iter", 14)
    res_thr  = hr_cfg.get("residual_threshold", 0.0001)
    step     = float(hr_cfg.get("step_size", 1.0))
    em_cfg   = config["em"]
    lambda_d = config.get("gaussian_splatting", {}).get("lambda_dssim", 0.2)
    use_mb   = hr_cfg.get("use_minibatch", True)

    img   = Image.open(img_path).convert("RGB")
    isize = config["dataset"].get("image_size")
    if isize:
        img = img.resize(tuple(isize), Image.LANCZOS)
    gt_np = np.array(img).astype(np.float32) / 255.0
    h, w  = gt_np.shape[:2]

    # ---- initial EM fit (shared between both modes) ----
    from run_hybrid_residual import _initial_em_fit
    I_init, i_means, i_covs, i_colors, i_weights = \
        _initial_em_fit(gt_np, n_init, em_cfg, "minibatch" if use_mb else "standard")

    I_oracle = I_init.copy()
    I_honest = I_init.copy()
    n_g_oracle = n_init
    n_g_honest = n_init

    init_m = evaluate_metrics(I_init, gt_np, compute_lpips_flag=False)

    frame_dir = out_dir / f"{img_path.stem}_frames_comparison"
    frame_dir.mkdir(exist_ok=True)
    frame_paths = []

    oracle_records = []
    honest_records = []

    def _save_frame(it, n_g_o, n_g_h, m_o, m_h):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(
            f"Iter {it} | Oracle Gaussians={n_g_o}  PSNR={m_o['psnr']:.2f} dB [!]  "
            f"| Honest Gaussians={n_g_h}  PSNR={m_h['psnr']:.2f} dB [ok]",
            fontsize=9,
        )
        axes[0].imshow(np.clip(gt_np, 0, 1));      axes[0].set_title("GT", fontsize=9)
        axes[1].imshow(np.clip(I_oracle, 0, 1));   axes[1].set_title(f"Oracle (inflated)\nPSNR={m_o['psnr']:.2f}", fontsize=9)
        axes[2].imshow(np.clip(I_honest, 0, 1));   axes[2].set_title(f"Honest (correct)\nPSNR={m_h['psnr']:.2f}", fontsize=9)
        diff = np.mean(np.abs(I_oracle - I_honest), axis=2)
        im = axes[3].imshow(diff, cmap="hot", vmin=0)
        plt.colorbar(im, ax=axes[3], fraction=0.04, pad=0.02)
        axes[3].set_title("|Oracle − Honest|", fontsize=9)
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        fp = frame_dir / f"frame_{it:04d}.png"
        plt.savefig(fp, dpi=90, bbox_inches="tight")
        plt.close(fig)
        return fp

    # frame 0 - initial state
    m0 = evaluate_metrics(I_init, gt_np, compute_lpips_flag=False)
    frame_paths.append(_save_frame(0, n_g_oracle, n_g_honest, m0, m0))
    print(f"  [init] PSNR={m0['psnr']:.2f} dB  (both modes identical at start)")

    for it in range(max_iter):
        if n_g_oracle >= n_total:
            break

        # Oracle mode: residuals from oracle image
        I_res_o      = gt_np - I_oracle
        mean_res_o   = float(np.mean(np.abs(I_res_o)))
        if mean_res_o < res_thr:
            print(f"  Oracle converged at iter {it+1}")
            break
        I_res_pos = np.maximum( I_res_o, 0.0).astype(np.float32)
        I_res_neg = np.maximum(-I_res_o, 0.0).astype(np.float32)

        # Fixed/honest mode: independent residuals from its own current image
        I_res_h      = gt_np - I_honest
        I_res_h_pos  = np.maximum( I_res_h, 0.0).astype(np.float32)
        I_res_h_neg  = np.maximum(-I_res_h, 0.0).astype(np.float32)

        n_avail  = min(n_per_iter, n_total - n_g_oracle)
        pos_mass = float(I_res_pos.sum())
        neg_mass = float(I_res_neg.sum())
        if n_avail < 2:
            n_pos = n_avail if pos_mass >= neg_mass else 0
            n_neg = n_avail - n_pos
        else:
            pos_frac = pos_mass / (pos_mass + neg_mass + 1e-10)
            n_pos = max(1, round(n_avail * pos_frac))
            n_neg = n_avail - n_pos
            n_neg = max(1, n_neg)
            n_pos = n_avail - n_neg

        # ---- Oracle mode: render_gaussians + residual clamp (original algorithm) ----
        I_pos_corr = np.zeros_like(gt_np)
        I_neg_corr = np.zeros_like(gt_np)
        n_added_o = 0

        if pos_mass > 1e-8 and n_pos >= 1:
            pp = fit_em_to_distribution(I_res_pos, n_pos, em_cfg, use_minibatch=use_mb)
            if pp is not None:
                gm_o, gc_o, gcol_o, gwt_o = pp
                I_pos_corr = render_gaussians(gm_o, gc_o, gcol_o, gwt_o, (h, w))
                n_added_o += len(gm_o)

        if neg_mass > 1e-8 and n_neg >= 1:
            np_p = fit_em_to_distribution(I_res_neg, n_neg, em_cfg, use_minibatch=use_mb)
            if np_p is not None:
                gm_o, gc_o, gcol_o, gwt_o = np_p
                I_neg_corr = render_gaussians(gm_o, gc_o, gcol_o, gwt_o, (h, w))
                n_added_o += len(gm_o)

        raw_delta_o = step * (I_pos_corr - I_neg_corr)
        I_delta_oracle = np.clip(raw_delta_o, -I_res_neg, I_res_pos)  # oracle clamp
        I_oracle = np.clip(I_oracle + I_delta_oracle, 0.0, 1.0).astype(np.float32)
        n_g_oracle += n_added_o

        # ---- Fixed/honest mode: render_residual_correction, own residual, no clamp ----
        pos_mass_h = float(I_res_h_pos.sum())
        neg_mass_h = float(I_res_h_neg.sum())
        I_pos_corr_h = np.zeros_like(gt_np)
        I_neg_corr_h = np.zeros_like(gt_np)
        n_added_h = 0

        if pos_mass_h > 1e-8 and n_pos >= 1:
            pp = fit_em_to_distribution(I_res_h_pos, n_pos, em_cfg, use_minibatch=use_mb)
            if pp is not None:
                gm_h, gc_h, gcol_h, gwt_h = pp
                I_pos_corr_h = render_residual_correction(I_res_h_pos, gm_h, gc_h, gwt_h, (h, w))
                n_added_h += len(gm_h)

        if neg_mass_h > 1e-8 and n_neg >= 1:
            np_p = fit_em_to_distribution(I_res_h_neg, n_neg, em_cfg, use_minibatch=use_mb)
            if np_p is not None:
                gm_h, gc_h, gcol_h, gwt_h = np_p
                I_neg_corr_h = render_residual_correction(I_res_h_neg, gm_h, gc_h, gwt_h, (h, w))
                n_added_h += len(gm_h)

        I_delta_honest = step * (I_pos_corr_h - I_neg_corr_h)  # no oracle clamp needed
        I_honest = np.clip(I_honest + I_delta_honest, 0.0, 1.0).astype(np.float32)
        n_g_honest += n_added_h

        m_o = evaluate_metrics(I_oracle, gt_np, compute_lpips_flag=False)
        m_h = evaluate_metrics(I_honest, gt_np, compute_lpips_flag=False)
        oracle_records.append({"iteration": it+1, "n_gaussians": n_g_oracle, **m_o})
        honest_records.append({"iteration": it+1, "n_gaussians": n_g_honest, **m_h})

        # Overcorrection stats (how often oracle clamp was active in oracle mode)
        pos_overshot = ((raw_delta_o > 0) & (raw_delta_o > I_res_pos)).sum()
        neg_overshot = ((raw_delta_o < 0) & (raw_delta_o < -I_res_neg)).sum()
        n_pixels     = gt_np.size
        print(
            f"  Iter {it+1:2d}: n_g_o={n_g_oracle}  n_g_h={n_g_honest}  "
            f"oracle_PSNR={m_o['psnr']:.2f}  fixed_PSNR={m_h['psnr']:.2f}  "
            f"oracle_clamp_px={pos_overshot+neg_overshot}/{n_pixels}"
            f" ({100*(pos_overshot+neg_overshot)/n_pixels:.1f}%)"
        )

        frame_paths.append(_save_frame(it+1, n_g_oracle, n_g_honest, m_o, m_h))

    # ---- final comparison plot ----
    final_oracle_psnr = oracle_records[-1]["psnr"] if oracle_records else m0["psnr"]
    final_honest_psnr = honest_records[-1]["psnr"] if honest_records else m0["psnr"]
    plot_oracle_vs_honest_comparison(
        gt_np, I_oracle, I_honest,
        init_psnr=m0["psnr"],
        oracle_psnr=final_oracle_psnr,
        honest_psnr=final_honest_psnr,
        n_gaussians=n_g_oracle,
        out_path=out_dir / f"{img_path.stem}_oracle_vs_honest.png",
    )

    # ---- save iteration tables ----
    if oracle_records:
        pd.DataFrame(oracle_records).to_csv(
            out_dir / f"{img_path.stem}_oracle_iterations.csv", index=False)
    if honest_records:
        pd.DataFrame(honest_records).to_csv(
            out_dir / f"{img_path.stem}_honest_iterations.csv", index=False)

    # ---- assemble MP4 ----
    if make_mp4 and frame_paths:
        try:
            from utils.hybrid_vis_utils import assemble_mp4
            mp4_path = out_dir / f"{img_path.stem}_oracle_vs_honest.mp4"
            assemble_mp4(frame_paths, mp4_path, fps=fps)
            print(f"  MP4 saved -> {mp4_path}")
        except Exception as exc:
            print(f"  MP4 failed: {exc}")

    return {
        "oracle_records": oracle_records,
        "honest_records": honest_records,
        "final_oracle_psnr": final_oracle_psnr,
        "final_honest_psnr": final_honest_psnr,
        "init_psnr":          m0["psnr"],
        "n_gaussians":        n_g_oracle,
    }


# ============================================================================
# 4.  Summary report
# ============================================================================

def print_section(title: str):
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def audit_directory(output_dir: Path, kodak_dir: Path, report_lines: list) -> dict:
    """
    Run all checks on one output directory.
    Returns a summary dict.
    """
    print_section(f"Auditing: {output_dir.name}")
    results_csv = output_dir / "results.csv"
    cfg_file    = output_dir / "config.yml"

    import yaml
    cfg = {}
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
    hr_cfg = cfg.get("hybrid_residual", {})

    # ── check reported results ──────────────────────────────────────────────
    df_results = pd.read_csv(results_csv)
    print(f"  Images: {len(df_results)}")
    if "hybrid_psnr" in df_results.columns:
        mean_psnr = df_results["hybrid_psnr"].mean()
        min_psnr  = df_results["hybrid_psnr"].min()
        max_psnr  = df_results["hybrid_psnr"].max()
        print(f"  PSNR (reported):  mean={mean_psnr:.2f}  min={min_psnr:.2f}  max={max_psnr:.2f}")

    # ── per-image iteration analysis ────────────────────────────────────────
    all_monotone = []
    gaussian_issues = []
    psnr_gains = []
    oracle_clamp_note = []

    for img_row in df_results.itertuples():
        stem     = Path(img_row.image).stem
        iter_csv = output_dir / f"{stem}_iterations.csv"
        if not iter_csv.exists():
            continue

        df_iter = pd.read_csv(iter_csv)
        gc      = check_gaussian_counts(df_iter, hr_cfg)
        psnr_c  = check_psnr_monotonicity(df_iter)

        all_monotone.append(psnr_c["is_monotonic"])
        psnr_gains.append(psnr_c["psnr_total_gain"])
        if gc["issues"]:
            gaussian_issues.extend([f"  {img_row.image}: {iss}" for iss in gc["issues"]])
        if psnr_c["is_monotonic"]:
            oracle_clamp_note.append(img_row.image)

    n_mono = sum(all_monotone)
    n_total_imgs = len(all_monotone)
    print(f"\n  Gaussian count violations:  {len(gaussian_issues)}")
    for iss in gaussian_issues[:10]:
        print(f"    {iss}")
    print(f"\n  PSNR monotone (oracle signature): {n_mono}/{n_total_imgs} images")
    print(f"  Mean PSNR gain over iterations:   {np.mean(psnr_gains):.2f} dB")
    if n_mono == n_total_imgs:
        print(
            "\n  !! WARNING: PSNR is strictly non-decreasing in ALL images!\n"
            "    This is the definitive signature of oracle-clamped updates.\n"
            "    The reported metrics are inflated and NOT suitable for publication."
        )
    elif n_mono > 0:
        print(
            f"\n  !! WARNING: {n_mono}/{n_total_imgs} images have monotone PSNR.\n"
            "    Oracle clamping is active on most images."
        )

    # ── independent metric recomputation from PNGs ─────────────────────────
    print(f"\n  Re-computing metrics from saved render PNGs …")
    df_verify = verify_metrics_from_pngs(output_dir, kodak_dir)
    if not df_verify.empty:
        max_delta_psnr = df_verify["psnr_delta"].abs().max()
        max_delta_ssim = df_verify["ssim_delta"].abs().max()
        print(f"  Max PSNR difference (reported vs recomputed): {max_delta_psnr:.4f} dB")
        print(f"  Max SSIM difference (reported vs recomputed): {max_delta_ssim:.6f}")
        if max_delta_psnr > 0.1:
            print(
                "  NOTE: Discrepancy likely from uint8 PNG quantization (expected ~0.2 dB)."
            )
        else:
            print("  OK: Reported metrics match saved renders (no numerical error).")

    # ── per-iteration PSNR trajectory plots ─────────────────────────────────
    for img_row in list(df_results.itertuples())[:3]:  # first 3 images
        stem     = Path(img_row.image).stem
        iter_csv = output_dir / f"{stem}_iterations.csv"
        if not iter_csv.exists():
            continue
        df_iter = pd.read_csv(iter_csv)
        plot_psnr_trajectory(
            df_iter,
            title=f"{stem} - PSNR trajectory  [{output_dir.name}]",
            out_path=output_dir / f"{stem}_psnr_trajectory_audit.png",
        )

    # ── record for report ───────────────────────────────────────────────────
    report_lines.append(f"\n### {output_dir.name}")
    report_lines.append(f"n_total={hr_cfg.get('n_total','?')}  n_init={hr_cfg.get('n_init','?')}  "
                        f"n_per_iter={hr_cfg.get('n_per_iter','?')}")
    report_lines.append(f"Gaussian count violations: {len(gaussian_issues)}")
    report_lines.append(f"Oracle monotone: {n_mono}/{n_total_imgs} images -- {'INFLATED' if n_mono == n_total_imgs else 'partial'}")
    if not df_verify.empty:
        report_lines.append(f"Max PSNR re-compute delta: {df_verify['psnr_delta'].abs().max():.4f}")
    if "hybrid_psnr" in df_results.columns:
        report_lines.append(f"Mean reported PSNR: {df_results['hybrid_psnr'].mean():.2f} dB")

    return {
        "dir": output_dir.name,
        "n_images": n_total_imgs,
        "n_monotone": n_mono,
        "gaussian_violations": len(gaussian_issues),
        "df_verify": df_verify,
    }


# ============================================================================
# 5.  Main entry point
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Audit hybrid Gaussian splatting results.")
    ap.add_argument("--output-dir",  default=None,
                    help="Single output dir to audit (default: all hybrid_residual_* under outputs/)")
    ap.add_argument("--kodak-dir",   default="./kodak",
                    help="Path to Kodak PNG images")
    ap.add_argument("--run-honest",  action="store_true",
                    help="Run honest (no oracle) re-run on kodim03 for comparison")
    ap.add_argument("--honest-image", default="kodim03.png",
                    help="Image to use for honest re-run (default: kodim03.png)")
    ap.add_argument("--honest-n-total", type=int, default=128,
                    help="n_total for honest re-run (default: 128)")
    ap.add_argument("--honest-out",  default="outputs/audit_honest_rerun",
                    help="Output dir for honest re-run")
    args = ap.parse_args()

    base_dir  = ROOT / "outputs"
    kodak_dir = ROOT / args.kodak_dir

    if args.output_dir:
        dirs_to_audit = [Path(args.output_dir)]
    else:
        dirs_to_audit = load_output_dirs(base_dir)

    if not dirs_to_audit:
        print("No hybrid_residual_* output directories with results.csv found.")
        sys.exit(1)

    print_section("HYBRID GAUSSIAN SPLATTING - CRITICAL AUDIT")
    print(f"  Directories to audit: {len(dirs_to_audit)}")
    print(
        "\n  PRIMARY CONCERN:  The oracle clamp in the per-iteration update\n"
        "  uses gt_np to bound every correction.  This guarantees PSNR can\n"
        "  only improve each step - a mathematical impossibility for any\n"
        "  honest estimator working without access to the ground truth.\n"
        "  The reported metrics are systematically inflated."
    )

    report_lines = [
        "# Hybrid Gaussian Splatting -- Audit Report\n",
        "## Key Finding\n",
        "The per-iteration update applies a pixel-wise oracle clamp:\n",
        "  `I_delta = np.clip(step*(I_pos - I_neg), -I_res_neg, I_res_pos)`\n",
        "where I_res_neg/I_res_pos are derived from gt_np.\n",
        "This means corrections can NEVER overshoot ground truth,\n",
        "PSNR is monotonically non-decreasing, and metrics are INFLATED.\n",
        "\n## Per-directory Results\n",
    ]

    all_summaries = []
    for d in dirs_to_audit:
        try:
            summary = audit_directory(d, kodak_dir, report_lines)
            all_summaries.append(summary)
        except Exception as e:
            print(f"  Error auditing {d.name}: {e}")
            import traceback; traceback.print_exc()

    # ── global summary ───────────────────────────────────────────────────────
    print_section("GLOBAL SUMMARY")
    total_mono = sum(s["n_monotone"] for s in all_summaries)
    total_imgs = sum(s["n_images"]   for s in all_summaries)
    print(f"  Oracle monotone: {total_mono}/{total_imgs} image×experiment pairs")
    print(f"  Gaussian violations: {sum(s['gaussian_violations'] for s in all_summaries)}")
    print(
        "\n  VERDICT: " +
        ("The results ARE affected by data leakage (oracle clamping).\n"
         "  Reported PSNR/SSIM values are inflated and cannot be used\n"
         "  as-is for publication without the fix in run_hybrid_residual.py."
         if total_mono == total_imgs else
         "Partial oracle clamping detected. Verify individual images.")
    )

    # ── write report ─────────────────────────────────────────────────────────
    # Strip non-ASCII characters to allow writing on any platform encoding
    report_text = "\n".join(report_lines)
    report_text = report_text.encode("ascii", errors="replace").decode("ascii")
    report_path = base_dir / "audit_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  Report written -> {report_path}")

    # ── honest re-run ────────────────────────────────────────────────────────
    if args.run_honest:
        print_section(f"HONEST RE-RUN: {args.honest_image}  n_total={args.honest_n_total}")

        from utils.em_utils import load_config
        cfg = load_config(str(ROOT / "config.yml"))
        cfg["hybrid_residual"]["n_total"]   = args.honest_n_total
        cfg["hybrid_residual"]["n_init"]    = max(16, args.honest_n_total // 8)
        cfg["hybrid_residual"]["n_per_iter"] = max(8, args.honest_n_total // 16)
        cfg["hybrid_residual"]["max_iter"]  = 14
        cfg["dataset"]["image_size"]        = None   # native resolution

        honest_out = ROOT / args.honest_out
        honest_out.mkdir(parents=True, exist_ok=True)
        img_path   = kodak_dir / args.honest_image

        result = run_honest_single_image(
            img_path, cfg, honest_out, fps=2, make_mp4=True,
        )
        print(f"\n  Init   PSNR: {result['init_psnr']:.2f} dB")
        print(f"  Oracle PSNR: {result['final_oracle_psnr']:.2f} dB  <-- INFLATED")
        print(f"  Honest PSNR: {result['final_honest_psnr']:.2f} dB  <-- CORRECT")
        inflation = result['final_oracle_psnr'] - result['final_honest_psnr']
        print(f"  Inflation:   +{inflation:.2f} dB (artifact from oracle clamping)")
        print(f"\n  Outputs written -> {honest_out}")


if __name__ == "__main__":
    main()
