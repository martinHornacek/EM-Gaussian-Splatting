# EM vs 2D Gaussian Splatting Comparison

Clean, modular implementation comparing EM algorithm and gradient-based 2D Gaussian Splatting for image reconstruction.

## 📁 Project Structure

```
.
├── config.yml                          # Main configuration file
├── utils/                              # Reusable helpers (dataset, metrics, em)
│   ├── dataset_utils.py
│   ├── metrics_utils.py
│   └── em_utils.py
├── run_em_algorithm.py                 # Run EM algorithm pipeline
├── run_gaussian_splatting_2d.py        # Run 2D Gaussian Splatting pipeline
├── scripts/                            # Small helper scripts (summarize & compare)
│   └── summarize_and_compare.py
├── kodak/                              # Auto-downloaded Kodak dataset
└── results/                            # Output directory
    ├── em_TIMESTAMP/                   # EM results
    └── gs_TIMESTAMP/                   # 2D-GS results
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision numpy matplotlib pillow pyyaml scikit-learn scipy scikit-image pandas
```

**Optional (for LPIPS metric):**
```bash
pip install lpips
```

### 2. Configure Experiment

Edit `config.yml` to adjust settings:
- Number of Gaussians (recommended: 1000-2500 for Kodak)
- Training epochs (200 for 2D-GS, 100 for EM)
- Which methods to run
- Number of images to process

### 3. Run Experiments

**Option A: Run a single method**
```bash
# Run EM only
python run_em_algorithm.py --config config.yml

# Run 2D-GS only
python run_gaussian_splatting_2d.py --config config.yml
```

## 📊 Tracked Metrics

### Core Metrics (always computed):
- **Loss**: Total training loss (L1 + λ·dSSIM)
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index
- **dSSIM**: Dissimilarity SSIM (1 - SSIM)

### Optional Metrics:
- **LPIPS**: Learned Perceptual Image Patch Similarity (requires `lpips` library)

### Compression Metrics:
- **n_gaussians**: Number of Gaussian components
- **bits_per_pixel**: Compressed size per pixel (9 params × bits_per_param)
  - 2 position parameters (x, y)
  - 3 covariance parameters (sigma_xx, sigma_yy, sigma_xy)
  - 3 color parameters (r, g, b)
  - 1 opacity parameter (alpha)
- **compression_ratio**: Original size / Compressed size
- **storage_mb**: Total storage in megabytes

### Timing:
- **total_time**: End-to-end time
- **fit_time**: EM fitting time (EM only)
- **render_time**: Rendering time (EM only)

## 📝 Configuration Guide

### Key Settings in `config.yml`:

```yaml
dataset:
  num_images: 3              # Use 3 for quick test, null for all 24

em:
  enabled: true              # Run EM algorithm
  n_components: 1000         # Number of Gaussians

gaussian_splatting:
  enabled: true              # Run 2D-GS
  primary_samples: 1000      # Number of Gaussians
  num_epochs: 500            # Training epochs
  densification: false       # Enable/disable densification
  pruning: true              # Enable/disable pruning

metrics:
  compute_lpips: true        # Compute LPIPS (needs lpips library)
```

## 📈 Output Files

### For Each Method (EM and 2D-GS):

```
results/[method]_TIMESTAMP/
├── config.yml                    # Copy of config used
├── results.csv                   # Summary metrics for all images
├── summary.csv                   # Statistical summary
├── [imagename]_render.png        # Final render
├── [imagename]_comparison.png    # GT vs render (EM)
├── [imagename]_final.png         # Full visualization (2D-GS)
├── [imagename]_history.csv       # Epoch-by-epoch metrics (2D-GS)
└── [imagename]_epoch_XXXX.png    # Intermediate results (2D-GS)
```

## 📊 Understanding the Results

### results.csv columns:
```
image, method, mse, rmse, psnr, ssim, dssim, lpips, 
n_gaussians, bits_per_pixel, compression_ratio, storage_mb,
total_time, fit_time, render_time, converged, n_iter
```

### For 2D-GS, [imagename]_history.csv tracks:
```
epoch, loss, l1, dssim, mse, rmse, psnr, ssim, active_gaussians
```

## 🔬 Algorithm Details

### EM Algorithm
- **Initialization**: K-means clustering in 5D space [x, y, r, g, b]
- **Gaussians**: Anisotropic (full covariance matrix)
- **Rotation**: Yes, implicitly in covariance structure
- **Optimization**: Expectation-Maximization (guaranteed convergence)

### 2D Gaussian Splatting
- **Initialization**: Random positions, colors sampled from image
- **Gaussians**: Anisotropic with explicit rotation parameter
- **Rotation**: Yes, theta ∈ [-PI/2, PI/2] (learned via gradient descent)
- **Optimization**: Adam optimizer with L1 + dSSIM loss
- **Parameters per Gaussian**: 9 values
  - 2 floats: scale (sx, sy)
  - 1 float: rotation angle
  - 1 float: alpha (opacity)
  - 3 floats: color (r, g, b)
  - 2 floats: position (x, y)

## 🎯 Recommended Experiments

### Quick Test
```yaml
dataset:
  num_images: 3

em:
  n_components: 500

gaussian_splatting:
  primary_samples: 500
  num_epochs: 200
```

### Medium Test
```yaml
dataset:
  num_images: null  # All 24 images

em:
  n_components: 1000

gaussian_splatting:
  primary_samples: 1000
  num_epochs: 500
```

### Extensive Run
```yaml
dataset:
  num_images: null

em:
  n_components: 1500

gaussian_splatting:
  primary_samples: 1500
  num_epochs: 800
  densification: true
  
metrics:
  compute_lpips: true
```

## 🔍 Troubleshooting

### Out of Memory
- Reduce `chunk_size` in config (default: 512 → try 256)
- Reduce `n_components` / `primary_samples`
- Use CPU: set `device: "cpu"` in config

### LPIPS Error
```bash
pip install lpips
```
Or disable in config: `compute_lpips: false`

### Slow Training (2D-GS)
- Reduce `num_epochs`
- Disable densification: `densification: false`
- Reduce plot interval: `plot_interval: 100`

### EM Doesn't Converge
- Increase `max_iter` (100 → 200)
- Try different `covariance_type` ("full" → "diag")
- Check if enough components for image complexity

## 📊 Analyzing Results

### 1. Compare Overall Performance
```python
import pandas as pd

em_results = pd.read_csv('results/em_*/results.csv')
gs_results = pd.read_csv('results/2dgs_*/results.csv')

print("EM Mean PSNR:", em_results['psnr'].mean())
print("2D-GS Mean PSNR:", gs_results['psnr'].mean())
```

### 2. Plot Training Evolution (2D-GS)
```python
history = pd.read_csv('results/2dgs_*/kodim01_history.csv')

import matplotlib.pyplot as plt
plt.plot(history['epoch'], history['psnr'])
plt.xlabel('Epoch')
plt.ylabel('PSNR (dB)')
plt.title('Training Evolution')
plt.show()
```

### 3. Rate-Distortion Curve
```python
import matplotlib.pyplot as plt

plt.scatter(em_results['bits_per_pixel'], em_results['psnr'], label='EM')
plt.scatter(gs_results['bits_per_pixel'], gs_results['psnr'], label='2D-GS')
plt.xlabel('Bits per Pixel')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.show()
```

## 📚 Research

### Key Findings:

1. **EM is faster**:  speedup compared to gradient descent
2. **EM is more stable**: Lower variance across runs
3. **EM achieves competitive quality**: Similar or better PSNR/SSIM
4. **EM requires fewer iterations**: Converges in ~50-100 iterations vs 500+ epochs
5. **No hyperparameter tuning needed**: EM automatically optimizes Gaussian parameters

## 🔧 Customization

### Add Your Own Dataset

1. Create folder with images (PNG/JPG)
2. Update config:
```yaml
dataset:
  path: "./my_dataset"
```

### Modify Metrics

Edit `metrics.py` to add custom metrics:
```python
def my_custom_metric(pred_np, gt_np):
    # Your metric here
    return value

# Add to evaluate_metrics() function
metrics['my_metric'] = my_custom_metric(pred_np, gt_np)
```

### Change Loss Function

Edit `gaussian_splatting_2d.py`, function `combined_loss()`:
```python
def combined_loss(pred, target, lambda_param=0.2):
    l1 = F.l1_loss(pred, target)
    dssim = compute_ssim_loss(pred, target)
    # Add your custom loss here
    return loss, l1, dssim
```

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@misc{em_gaussian_splatting,
  title={EM Algorithm for 2D Gaussian Splatting},
  author={Martin Hornáček,
  year={2025},
  howpublished={\url{https://github.com/martinHornacek/em-gaussian-splatting}}
}
```

## 📄 License

MIT License - Feel free to use for research and commercial purposes.

---

**Questions? Issues?** Open an issue or contact martin.hornacek@stuba.sk