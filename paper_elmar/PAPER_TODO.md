# ELMAR 2025 Paper — "Image Reconstruction Using EM with 2D Gaussian Mixtures"
# PAPER_TODO.md — updated 2026-05-04

---

## Current state summary

| Item | Status |
|------|--------|
| EM one-shot results (K=128–1024, 24 images) | ✅ Done |
| EM one-shot K=2048 | ⏳ Half-done (12/24 images); run rest with `config_em_k2048_standard.yml` |
| EM mini-batch (K=128–2048, 24 images) | ✅ Done — in paper (Table III) |
| Tiled EM results (K=128–2048, 24 images) | ✅ Done — K=256 anomaly confirmed, explained correctly |
| Hybrid residual ablation (K=128–1024, 24 images) | ✅ Done — Table II |
| Hybrid residual K=2048 | ✅ Done (22.71 dB) — in Table II with mini-batch reference |
| GS gradient baseline (K=128–512, 24 images) | ❌ Invalid — no densification, 7–11 dB results unusable |
| GS K=1024 (12/24 images) | ⏳ Partial — run rest with `config_gs_1024.yml` |
| PSNR vs K figure | ✅ `paper_elmar/figures/fig_psnr_vs_k.pdf` |
| Visual comparison figure | ✅ `paper_elmar/figures/fig_visual.pdf` — caption corrected to 22.43 dB |
| Tiling structure figure | ✅ `paper_elmar/figures/fig_tiling.pdf` — caption corrected to 22.43 dB |
| LaTeX paper draft | ✅ Updated 2026-05-04 with real numbers and new results |

---

## Fixes applied 2026-05-04

- **FIXED:** Table I footnote was wrong ("per-tile residual refinement") — replaced with correct explanation (K_tile=16 Gaussians too few for reliable convergence)
- **FIXED:** Experiments section same wrong cause — corrected to per-tile budget argument
- **FIXED:** Fig 2 (visual) caption: 22.54 dB → 22.43 dB (actual kodim15 tiled K=1024)
- **FIXED:** Fig tiling caption: same 22.54 → 22.43 correction
- **FIXED:** "K/28" mechanical claim — replaced with defensible qualitative description
- **ADDED:** Table II K=2048 hybrid row (22.71 dB, mini-batch EM reference 23.64 dB, Δ=−0.93 dB)
- **ADDED:** Section 4.4 Mini-Batch EM (Table III with K=128–2048, PSNR/SSIM/RMSE/time)
- **ADDED:** 4th contribution for mini-batch
- **UPDATED:** Abstract: mini-batch speedup sentence added
- **UPDATED:** Conclusion: mini-batch findings + hybrid K=2048 extension

---

## Remaining experiments (priority order)

### High priority

- [ ] **Re-run GS baseline with densification enabled** for K=128, 256, 512, 1024
  Current results (7–11 dB at K=128–512) are invalid — no densification means GS never grows.
  Update `config_gs_*.yml`: set `densification: true`, `pruning: true`.
  Needed for a meaningful GS comparison in the paper.

- [ ] **Complete EM one-shot K=2048** (12/24 images done)
  Run: `python run_em_algorithm.py --config config_em_k2048_standard.yml`
  (resume from image 13 if possible, otherwise re-run all)
  Once complete: update Table II K=2048 EM column with actual standard EM value.

- [ ] **Complete GS K=1024** (12/24 images done)
  Run: `python run_gaussian_splatting_2d.py --config config_gs_1024.yml`

### Medium priority

- [ ] **Add GS comparison to Table I** once valid GS results are available.
  Add rows: GS (K=128/256/512/1024) with PSNR, SSIM, RMSE, training time.

- [ ] **Rate-distortion figure** (PSNR vs bpp) for EM one-shot, tiled, mini-batch.
  All bpp values are computed (9 × 32 × K / (256×256) bits/pixel).
  This would strengthen the compression framing.

- [ ] **Re-run kodim21 hybrid K=2048** to confirm PSNR (21.02 dB).
  The PSNR value looks normal but time was 23.7 min (power-loss outlier vs ~13 min normal).
  Command: run `config_hybrid_k2048.yml` with `image_list: [21]`.

### Low priority / future work (paper text already covers this)

- [ ] Coverage-correction implementation in `run_hybrid_residual.py`
- [ ] GD fine-tuning on top of EM initialisation (hybrid EM+GD)
- [ ] Evaluation on additional datasets beyond Kodak


---

## Current state summary

| Item | Status |
|------|--------|
| EM one-shot results (K=128–1024, 24 images) | ✅ Done |
| Tiled EM results (K=128–2048, 24 images) | ✅ Done (K=256 anomaly — re-run) |
| Hybrid residual ablation (K=128–1024, 24 images) | ✅ Done (negative result) |
| Minibatch hybrid (K=128) | ✅ Done |
| Minibatch hybrid (K=256–1024) | ❌ Not started |
| GS gradient baseline (K=2500, 9/24 images) | 🔶 Partial (9 images) |
| PSNR vs K figure | ✅ `paper_elmar/figures/fig_psnr_vs_k.pdf` |
| Visual comparison figure | ✅ `paper_elmar/figures/fig_visual.pdf` |
| Tiling structure figure | ✅ `paper_elmar/figures/fig_tiling.pdf` |
| LaTeX paper draft | ✅ Updated with real numbers |

---

## Remaining experiments

### High priority

- [ ] **Re-run Tiled EM K=256** (suspicious 18.35 dB result, likely bad init)
  ```
  python run_tiled_em.py --config config_tiled.yml  # set n_total=256
  ```
  Expected: ~21–22 dB (should be between K=128 and K=512)

- [ ] **Run full GS baseline (all 24 Kodak images, K=2500)**
  ```
  python run_gaussian_splatting_2d.py --config config_kodak_full.yml
  ```
  Currently only 9 images done. Needed for a fair comparison in Table I.
  Estimated time: ~4–8h on CPU.

### Medium priority

- [ ] **Minibatch hybrid K=256, 512, 1024** (was interrupted mid-run)
  ```
  python run_paper_comparison.py --only-minibatch
  ```
  Minibatch results for K=128 available (mean PSNR 18.09 dB — also worse than EM).

- [ ] **EM one-shot K=2048** (skipped; tiled gives ~24.1 dB at this K)
  Useful to confirm one-shot also scales well to K=2048.

### Low priority / Future work

- [ ] Gradient-based refinement on top of EM init (EM + GS fine-tuning)
- [ ] Larger image experiments (tiling shines on images > 256×256)
- [ ] Per-channel EM fitting (currently fits in 5D joint space)

---

## Missing figures

- [ ] **fig_pipeline.pdf** — block diagram of the hybrid residual scheme
  (negative result but referenced in text; can be TikZ/draw.io)
- [ ] **fig_progression.pdf** — optional: EM renders at K=128,256,512,1024 side-by-side
- [ ] Re-generate `fig_psnr_vs_k` after K=256 tiled re-run

---

## Paper sections needing refinement

- [ ] **Introduction**: Add 1–2 sentences on related 2D GS / image compression work
- [ ] **Method § Rendering (Eq. 3)**: Verify weight normalisation matches `utils/em_utils.py`
- [ ] **Table I**: Add GS comparison row once full 24-image run is done
- [ ] **Bibliography**: Add at least one 2D GS / image representation reference
- [ ] **Page limit check**: Trim to 4 IEEE pages

---

## Overleaf file recommendations

**Keep:**
- `em_gaussian_splatting.tex`
- `IEEEtran.cls`
- `reference.bib`
- All `figures/*.pdf`

**Remove / ignore:**
- `spmpsci.bst` — Springer style, not used (`\bibliographystyle{ieeetr}` is built-in).
  Safe to leave in the project but do NOT change the bibliographystyle line.

**Compile sequence:**
```
pdflatex em_gaussian_splatting
bibtex   em_gaussian_splatting
pdflatex em_gaussian_splatting
pdflatex em_gaussian_splatting
```

---

## Key numbers (reference)

| Method | K | Mean PSNR (dB) | Mean SSIM | Source |
|--------|---|----------------|-----------|--------|
| EM one-shot | 128 | 20.89 | 0.4917 | fullbatch_k128 (24 imgs) |
| EM one-shot | 256 | 21.91 | 0.5342 | fullbatch_k256 (24 imgs) |
| EM one-shot | 512 | 23.00 | 0.5872 | fullbatch_k512 (24 imgs) |
| EM one-shot | 1024 | 23.79 | 0.6323 | fullbatch_k1024 (24 imgs) |
| Tiled EM | 128 | 19.10 | 0.4184 | tiled_fullbatch_128 |
| Tiled EM | 256 | 18.35 ⚠ | 0.4174 | tiled_fullbatch_256 (anomaly) |
| Tiled EM | 512 | 20.44 | 0.4774 | tiled_fullbatch_512 |
| Tiled EM | 1024 | 22.33 | 0.5516 | tiled_fullbatch_1024 |
| Tiled EM | 2048 | 24.10 | 0.6401 | tiled_fullbatch_2048 |
| Hybrid (fullbatch) | 128 | 18.09 | 0.4125 | paper_comparison k128 |
| Hybrid (fullbatch) | 256 | 19.12 | 0.4326 | paper_comparison k256 |
| Hybrid (fullbatch) | 512 | 20.38 | 0.4705 | paper_comparison k512 |
| Hybrid (fullbatch) | 1024 | 21.45 | 0.5209 | paper_comparison k1024 |
| GS (gradient) | 2500 | 25.55 | — | gs_full_kodak (9/24 imgs only) |

