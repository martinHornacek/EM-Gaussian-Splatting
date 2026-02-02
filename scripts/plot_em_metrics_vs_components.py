"""Aggregate EM results across `results/em_*` runs and plot metrics vs number of components.

Usage:
    python scripts/plot_em_metrics_vs_components.py --results-dir results --outdir results/comparison/em_components

Produces:
 - `results/comparison/em_components/em_aggregated_metrics.csv`
 - PNG plots for PSNR, SSIM, LPIPS, Bits-per-pixel and total/run times

Notes:
 - The script reads `results.csv` files inside directories beginning with `em_`.
 - The number of components is taken from the `n_gaussians` column in each results.csv (robust to folder naming).
"""

from pathlib import Path
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# metrics we care about for single-image analysis (no bits-per-pixel)
METRICS_TO_PLOT = ["psnr", "rmse", "ssim", "lpips"]
TIME_COLS = ["fit_time", "render_time", "total_time"]


def find_em_dirs(results_dir: Path):
    """Return only directories whose names match `em_<number>` exactly.

    Examples: `em_10`, `em_2500` are matched. `em_full_kodak_2500_gaussians` is ignored.
    """
    out = []
    for p in sorted(results_dir.iterdir()):
        if p.is_dir() and re.match(r"^em_\d+$", p.name):
            results_csv = p / "results.csv"
            if results_csv.exists():
                out.append(p)
    return out


def collect_per_image_metrics(dirs, image_name: str):
    """For each em_* dir, find the row for image_name and collect metrics and times."""
    rows = []
    for d in dirs:
        df = pd.read_csv(d / "results.csv")
        # find row for image
        if "image" not in df.columns:
            continue
        match = df[df["image"] == image_name]
        if match.empty:
            continue
        row = match.iloc[0].to_dict()
        # get n_components robustly
        if "n_gaussians" in df.columns:
            n_vals = df["n_gaussians"].unique()
            n_components = int(n_vals[0]) if len(n_vals) >= 1 else None
        else:
            search = re.search(r"(\d+)", d.name)
            n_components = int(search.group(1)) if search else None
        out = {
            "folder": d.name,
            "n_components": n_components,
        }
        # metrics
        for m in METRICS_TO_PLOT:
            out[m] = float(row.get(m, float('nan'))) if m in row else float('nan')
        # times
        for t in TIME_COLS:
            out[t] = float(row.get(t, float('nan'))) if t in row else float('nan')
        rows.append(out)
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out = df_out.dropna(subset=["n_components"]).sort_values(by="n_components")
    return df_out


def plot_single_metric(df_img, image_name: str, metric: str, outdir: Path, logx=False):
    """Plot a single metric vs number of components and save it."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4.5))
    x = df_img["n_components"]
    y = df_img[metric]
    plt.plot(x, y, marker="o")
    plt.xlabel("Number of components")

    if metric == "psnr":
        plt.ylabel("PSNR (dB)")
        title_metric = "PSNR (dB)"
    elif metric == "rmse":
        plt.ylabel("RMSE (unitless)")
        title_metric = "RMSE"
    elif metric == "ssim":
        plt.ylabel("SSIM (unitless)")
        title_metric = "SSIM"
    elif metric == "lpips":
        plt.ylabel("LPIPS (unitless)")
        title_metric = "LPIPS"
    else:
        plt.ylabel(metric)
        title_metric = metric

    if logx:
        plt.xscale("log")

    plt.title(f"{title_metric} for {image_name} vs Number of components")
    plt.tight_layout()
    outpath = outdir / f"{metric}_{os.path.splitext(image_name)[0]}_vs_components.png"
    plt.savefig(outpath)
    plt.close()
    print(f"Saved plot: {outpath}")


def plot_times(df_img, image_name: str, outdir: Path, logx=False):
    """Make a clearer timing plot: stacked area for fit+render and an overlaid total_time line.

    Falls back to line plots if fit/render not available.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 4.5))

    x = df_img["n_components"]

    fit_present = "fit_time" in df_img.columns and not df_img["fit_time"].isna().all()
    render_present = "render_time" in df_img.columns and not df_img["render_time"].isna().all()
    total_present = "total_time" in df_img.columns and not df_img["total_time"].isna().all()

    if fit_present and render_present:
        # stacked area: fit_time + render_time
        y1 = df_img["fit_time"].values
        y2 = df_img["render_time"].values
        plt.stackplot(x, y1, y2, labels=["fit_time (s)", "render_time (s)"], colors=["#1f77b4", "#ff7f0e"], alpha=0.8)
        if total_present:
            plt.plot(x, df_img["total_time"], marker="o", color="k", linestyle="--", label="total_time (s)")
        plt.xlabel("Number of components")
        plt.ylabel("Time (s)")
        plt.title(f"Timings for {image_name} — stacked fit+render with total overlay")
    else:
        # Fall back to separate lines for whatever is available
        for t in TIME_COLS:
            if t in df_img.columns:
                plt.plot(x, df_img[t], marker="o", label=f"{t} (s)")
        plt.xlabel("Number of components")
        plt.ylabel("Time (s)")
        plt.title(f"Timings for {image_name} vs Number of components")

    if total_present and not (fit_present and render_present):
        # ensure legend shows total_time even in stacked case it's already added
        pass

    # Annotate total_time points for clarity
    if total_present:
        y_total = df_img["total_time"].values
        npts = len(y_total)
        x_vals = x.values if hasattr(x, 'values') else np.array(x)
        x_min, x_max = x_vals.min(), x_vals.max() if len(x_vals) else (0, 0)
        dx = (x_max - x_min) * 0.04 if x_max > x_min else 1.0

        # If many points, annotate only a subset (first 2, last 2, and max) and place labels outside plot
        if npts > 8:
            idxs = [0, 1, int(np.argmax(y_total)), npts - 2, npts - 1]
            idxs = [i for i in idxs if 0 <= i < npts]
            used = set()
            for i in idxs:
                if i in used:
                    continue
                used.add(i)
                xi = x_vals[i]
                yi = y_total[i]
                if i in (0, 1):
                    # place label to the left outside plot
                    xytext = (-60, 0)
                    ha = 'right'
                    va = 'center'
                    text_xy = (xi - dx, yi)
                elif i in (npts - 2, npts - 1):
                    # place label to the right outside plot
                    xytext = (60, 0)
                    ha = 'left'
                    va = 'center'
                    text_xy = (xi + dx, yi)
                else:
                    # place above the point for the max
                    xytext = (0, 12)
                    ha = 'center'
                    va = 'bottom'
                    text_xy = (xi, yi)

                plt.annotate(f"{yi:.0f}s", xy=(xi, yi), xytext=xytext, textcoords='offset points',
                             ha=ha, va=va, fontsize=8,
                             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'),
                             arrowprops=dict(arrowstyle='->', lw=0.7, color='gray', shrinkA=0, shrinkB=0))
        else:
            # few points: alternate vertical offsets but move labels slightly outward for readability
            for i, (xi, yi) in enumerate(zip(x_vals, y_total)):
                offset_y = 8 if (i % 2 == 0) else -10
                xytext = (0, offset_y)
                va = 'bottom' if offset_y > 0 else 'top'
                plt.annotate(f"{yi:.0f}s", xy=(xi, yi), xytext=xytext, textcoords='offset points',
                             ha='center', va=va, fontsize=8,
                             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

    if logx:
        try:
            plt.xscale('log')
        except Exception:
            # log scaling may not be appropriate for bar/stack plots
            pass

    plt.legend(loc="best")
    plt.tight_layout()
    outpath = outdir / f"timings_{os.path.splitext(image_name)[0]}_vs_components.png"
    plt.savefig(outpath)
    plt.close()
    print(f"Saved plot: {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", help="Top-level results directory")
    parser.add_argument("--outdir", default="results/comparison/em_components", help="Output dir for plots and CSV")
    parser.add_argument("--image", required=True, help="Image filename to analyze, e.g. kodim01.png")
    parser.add_argument("--logx", action="store_true", help="Plot number of components on log scale")
    parser.add_argument("--show", action="store_true", help="Show plots interactively (useful in notebooks)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    em_dirs = find_em_dirs(results_dir)
    if not em_dirs:
        raise FileNotFoundError(f"No em_* directories with results.csv found in {results_dir}")

    # collect per-image data across runs
    df_img = collect_per_image_metrics(em_dirs, args.image)
    if df_img.empty:
        raise ValueError(f"Image {args.image} not found in any em_* results.csv under {results_dir}")

    # save per-image CSV
    csv_out = outdir / f"em_image_{os.path.splitext(args.image)[0]}_metrics.csv"
    df_img.to_csv(csv_out, index=False)
    print(f"Saved per-image CSV to {csv_out}")

    # Plot individual quality metrics (PSNR, SSIM, LPIPS, RMSE)
    for m in METRICS_TO_PLOT:
        if m in df_img.columns:
            plot_single_metric(df_img, args.image, m, outdir, logx=args.logx)

    # Plot timings vs components
    plot_times(df_img, args.image, outdir, logx=args.logx)

    if args.show:
        print("--show requested but GUI mode may not be available in headless environments")


if __name__ == "__main__":
    main()
