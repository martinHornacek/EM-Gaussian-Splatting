# Hybrid Gaussian Splatting -- Audit Report

## Key Finding

The per-iteration update applies a pixel-wise oracle clamp:

  `I_delta = np.clip(step*(I_pos - I_neg), -I_res_neg, I_res_pos)`

where I_res_neg/I_res_pos are derived from gt_np.

This means corrections can NEVER overshoot ground truth,

PSNR is monotonically non-decreasing, and metrics are INFLATED.


## Per-directory Results


### hybrid_residual_1024_gaussians_baseline
n_total=1024  n_init=128  n_per_iter=64
Gaussian count violations: 0
Oracle monotone: 24/24 images -- INFLATED
Max PSNR re-compute delta: 0.2371
Mean reported PSNR: 31.59 dB

### hybrid_residual_1024_gaussians_minibatch
n_total=1024  n_init=128  n_per_iter=64
Gaussian count violations: 0
Oracle monotone: 24/24 images -- INFLATED
Max PSNR re-compute delta: 0.2202
Mean reported PSNR: 31.83 dB

### hybrid_residual_2048_gaussians_baseline
n_total=2048  n_init=256  n_per_iter=128
Gaussian count violations: 0
Oracle monotone: 24/24 images -- INFLATED
Max PSNR re-compute delta: 0.2850
Mean reported PSNR: 33.61 dB

### hybrid_residual_2048_gaussians_minibatch
n_total=2048  n_init=256  n_per_iter=128
Gaussian count violations: 0
Oracle monotone: 24/24 images -- INFLATED
Max PSNR re-compute delta: 0.2744
Mean reported PSNR: 34.02 dB

### hybrid_residual_fullkodak_1024gaussians_minibatch
n_total=1024  n_init=128  n_per_iter=64
Gaussian count violations: 0
Oracle monotone: 24/24 images -- INFLATED
Max PSNR re-compute delta: 0.2235
Mean reported PSNR: 29.13 dB

### hybrid_residual_fullkodak_128gaussians_minibatch
n_total=128  n_init=16  n_per_iter=8
Gaussian count violations: 0
Oracle monotone: 24/24 images -- INFLATED
Max PSNR re-compute delta: 0.1974
Mean reported PSNR: 22.93 dB

### hybrid_residual_fullkodak_2048gaussians_minibatch
n_total=2048  n_init=256  n_per_iter=128
Gaussian count violations: 0
Oracle monotone: 24/24 images -- INFLATED
Max PSNR re-compute delta: 0.2289
Mean reported PSNR: 30.58 dB

### hybrid_residual_fullkodak_256gaussians_minibatch
n_total=256  n_init=32  n_per_iter=16
Gaussian count violations: 0
Oracle monotone: 24/24 images -- INFLATED
Max PSNR re-compute delta: 0.2171
Mean reported PSNR: 25.54 dB

### hybrid_residual_fullkodak_512gaussians_minibatch
n_total=512  n_init=64  n_per_iter=32
Gaussian count violations: 0
Oracle monotone: 24/24 images -- INFLATED
Max PSNR re-compute delta: 0.2145
Mean reported PSNR: 28.09 dB