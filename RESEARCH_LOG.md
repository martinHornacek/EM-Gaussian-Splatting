# Research Journal: EM vs 2D Gaussian Splatting

**Project**: Comparison of Expectation-Maximization and Gradient-Based Gaussian Splatting for image reconstruction

**Date Started**: January 2025

---

## Summary (short)
- Dataset: Kodak PhotoCD (24 images), resized to 256×256, RGB, normalized.
- Baseline EM setup used for reported experiments: 2500 Gaussians, full covariance, K-means init, max_iter=100.
- Key findings: EM is stable (100% convergence), runtime dominated by GMM fitting (~15–20 min/image on Apple M2), and quality averages around PSNR ~24.5 dB, SSIM ~0.68, LPIPS ~0.41.
- Main limitation: high-frequency details are not well captured (higher LPIPS, lower PSNR on textured scenes).

---

## Key Observations
- Convergence time correlates with scene complexity: smooth images converge faster than highly textured ones.
- GMM fitting is the bottleneck (~99% of runtime); rendering is negligible.
- Quality vs components: diminishing returns observed at high component counts; further sweeps required to find elbow point.

---

## Next Experiments (prioritized)
1. 2D-GS baseline: run with 2500 Gaussians, 200 epochs (compare quality & runtime).
2. Component sweep for EM: [500, 1000, 1500, 2000, 2500, 3000] on representative subset.
3. Covariance ablation: full vs diagonal vs spherical.
4. Image-statistics study: predict convergence time from entropy/edge density.

---

## Data & Code
- Results directory example: `results/em_full_kodak_2500_gaussians/` (contains `config.yml`, `results.csv`, `summary.csv`).
- Runners: `run_em_algorithm.py`, `run_gaussian_splatting_2d.py`, `run_compression_analysis.py`, `run_comparison_analysis.py`.

---

## Notes
- Reported runs were executed on Apple M2 (MacBook Air) — timings will vary by hardware.
- Implementation details: EM uses `sklearn.mixture.GaussianMixture`; LPIPS (optional) uses `lpips` with AlexNet backbone.

---

## Recent updates
- **2025-01-27**: Initial EM experiments on full Kodak dataset (2500 Gaussians).  Findings summarized above.
