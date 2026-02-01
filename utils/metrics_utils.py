"""
metrics.py - Shared Metrics Computation for EM and 2D-GS
Tracks: Loss, MSE, RMSE, PSNR, dSSIM, LPIPS, Compression metrics
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio


# ==================== SSIM / dSSIM ====================

_SSIM_CACHE = {}

def _ssim_window(window_size, channel, device, dtype):
    """Create cached Gaussian window for SSIM"""
    key = (window_size, channel, str(device), str(dtype))
    if key in _SSIM_CACHE:
        return _SSIM_CACHE[key]
    coords = torch.arange(window_size, device=device, dtype=dtype)
    g1 = torch.exp(-((coords - window_size//2)**2) / (2*1.5*1.5))
    g1 = (g1 / g1.sum()).unsqueeze(1)
    g2 = (g1 @ g1.T).float().unsqueeze(0).unsqueeze(0)
    win = g2.expand(channel, 1, window_size, window_size).contiguous()
    _SSIM_CACHE[key] = win
    return win


def compute_ssim_loss(img1, img2, window_size=11):
    """
    Compute dSSIM loss (differentiable SSIM for training)
    Returns: dSSIM value (0 = identical, 1 = completely different)
    """
    x = img1.permute(2,0,1).unsqueeze(0)
    y = img2.permute(2,0,1).unsqueeze(0)
    device, dtype = x.device, x.dtype
    C1, C2 = 0.01**2, 0.03**2
    win = _ssim_window(window_size, 3, device, dtype)
    mu1 = F.conv2d(x, win, padding=window_size//2, groups=3)
    mu2 = F.conv2d(y, win, padding=window_size//2, groups=3)
    mu1_sq, mu2_sq, mu1_mu2 = mu1*mu1, mu2*mu2, mu1*mu2
    sigma1_sq = F.conv2d(x*x, win, padding=window_size//2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(y*y, win, padding=window_size//2, groups=3) - mu2_sq
    sigma12   = F.conv2d(x*y, win, padding=window_size//2, groups=3) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2) + 1e-8)
    return ((1 - ssim_map.clamp(0,1)) * 0.5).mean()


def compute_ssim_metric(pred_np, gt_np):
    """
    Compute SSIM metric (numpy version for evaluation)
    Returns: SSIM value (0-1, higher is better)
    """
    from skimage.metrics import structural_similarity
    return structural_similarity(gt_np, pred_np, channel_axis=2, data_range=1.0)


# ==================== LPIPS ====================

_LPIPS_MODEL = None

def compute_lpips(pred_np, gt_np, device='cuda'):
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity)
    Returns: LPIPS value (lower is better), or -1 if unavailable
    """
    global _LPIPS_MODEL
    
    try:
        import lpips
        
        # Use CPU if CUDA not available
        lpips_device = device if torch.cuda.is_available() else 'cpu'
        
        # Initialize model once (cache it)
        if _LPIPS_MODEL is None:
            _LPIPS_MODEL = lpips.LPIPS(net='alex').to(lpips_device)
            _LPIPS_MODEL.eval()
        
        # Convert to torch tensors [1, 3, H, W] in range [-1, 1]
        pred_t = torch.from_numpy(pred_np).permute(2, 0, 1).unsqueeze(0).float().to(lpips_device)
        gt_t = torch.from_numpy(gt_np).permute(2, 0, 1).unsqueeze(0).float().to(lpips_device)
        pred_t = pred_t * 2.0 - 1.0
        gt_t = gt_t * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_val = _LPIPS_MODEL(pred_t, gt_t).item()
        
        return lpips_val
    except (ImportError, AssertionError) as e:
        # Return -1 if lpips not available or CUDA error
        return -1.0


# ==================== Compression Metrics ====================

def compute_compression_metrics(n_components, image_size=(256, 256), quantization='fp32'):
    """
    Compute compression-related metrics following 2D Gaussian Splatting standards
    
    Each Gaussian stores:
    - 2 floats: position (x, y)
    - 3 floats: covariance (sigma_xx, sigma_yy, sigma_xy) - symmetric 2x2 matrix
    - 3 floats: color (r, g, b)
    - 1 float: alpha (opacity/weight)
    Total: 9 parameters per Gaussian
    
    Args:
        n_components: Number of Gaussians
        image_size: Image resolution (h, w)
        quantization: 'fp32' (32-bit), 'fp16' (16-bit), 'int8' (8-bit)
                     Default is 'fp32' for comparison with papers
    
    Returns:
        dict with compression metrics
    """
    h, w = image_size
    
    # Bits per parameter based on quantization
    bits_per_param = {
        'fp32': 32,
        'fp16': 16,
        'int8': 8
    }
    
    if quantization not in bits_per_param:
        raise ValueError(f"Unknown quantization: {quantization}. Choose from {list(bits_per_param.keys())}")
    
    bits_per_float = bits_per_param[quantization]
    
    # Parameters per Gaussian: 2 (pos) + 3 (cov) + 3 (color) + 1 (alpha) = 9
    params_per_gaussian = 9
    bits_per_gaussian = params_per_gaussian * bits_per_float
    
    # Original image: 24 bits per pixel (8 bits × 3 channels RGB)
    original_bits = h * w * 24
    
    # Gaussian representation
    gaussian_bits = n_components * bits_per_gaussian
    
    # Metrics
    bits_per_pixel = gaussian_bits / (h * w)
    compression_ratio = original_bits / gaussian_bits
    storage_mb = gaussian_bits / (8 * 1024 * 1024)
    
    return {
        'n_gaussians': n_components,
        'bits_per_pixel': bits_per_pixel,
        'compression_ratio': compression_ratio,
        'storage_mb': storage_mb,
        'original_bits': original_bits,
        'compressed_bits': gaussian_bits,
        'quantization': quantization,
        'bits_per_gaussian': bits_per_gaussian
    }


# ==================== Complete Metrics Evaluation ====================

def evaluate_metrics(pred, gt, n_components=None, compute_lpips_flag=False, device='cuda'):
    """
    Compute all metrics for a prediction vs ground truth
    
    Args:
        pred: Prediction tensor/array (H, W, 3) in range [0, 1]
        gt: Ground truth tensor/array (H, W, 3) in range [0, 1]
        n_components: Number of Gaussians (for compression metrics)
        compute_lpips_flag: Whether to compute LPIPS
        device: torch device
    
    Returns:
        dict with all metrics
    """
    # Convert to numpy if needed
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = pred
    
    if isinstance(gt, torch.Tensor):
        gt_np = gt.detach().cpu().numpy()
    else:
        gt_np = gt
    
    # Ensure values in [0, 1]
    pred_np = np.clip(pred_np, 0, 1)
    gt_np = np.clip(gt_np, 0, 1)
    
    # MSE & RMSE
    mse = np.mean((pred_np - gt_np) ** 2)
    rmse = np.sqrt(mse)
    
    # PSNR (in dB)
    psnr = peak_signal_noise_ratio(gt_np, pred_np, data_range=1.0)
    
    # SSIM
    ssim = compute_ssim_metric(pred_np, gt_np)
    
    # dSSIM (for compatibility with loss tracking)
    dssim = 1.0 - ssim
    
    # LPIPS (optional)
    if compute_lpips_flag:
        lpips_val = compute_lpips(pred_np, gt_np, device)
    else:
        lpips_val = -1.0
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'psnr': float(psnr),
        'ssim': float(ssim),
        'dssim': float(dssim),
        'lpips': float(lpips_val)
    }
    
    # Add compression metrics if n_components provided
    if n_components is not None:
        comp_metrics = compute_compression_metrics(n_components, pred_np.shape[:2])
        metrics.update(comp_metrics)
    
    return metrics


def print_metrics(metrics, method_name="", epoch=None):
    """Pretty print metrics"""
    if epoch is not None:
        prefix = f"[{method_name}] Epoch {epoch}: "
    else:
        prefix = f"[{method_name}] "
    
    print(f"{prefix}PSNR={metrics['psnr']:.2f} dB, RMSE={metrics['rmse']:.6f}, "
          f"SSIM={metrics['ssim']:.4f}, dSSIM={metrics['dssim']:.6f}", end="")
    
    if metrics.get('lpips', -1) >= 0:
        print(f", LPIPS={metrics['lpips']:.4f}", end="")
    
    if 'bits_per_pixel' in metrics:
        print(f", bpp={metrics['bits_per_pixel']:.2f}, CR={metrics['compression_ratio']:.1f}x", end="")
    
    print()