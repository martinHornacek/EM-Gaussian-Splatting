"""
em_utils.py - Shared utilities for EM fitting and rendering

Contains: load_config, download_kodak_dataset, _prepare_data, render_gaussians.
"""

import os
import yaml
import numpy as np
from pathlib import Path
from scipy.stats import multivariate_normal


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

def load_config(config_path='config.yml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_kodak_dataset(output_dir="./kodak"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    existing = list(Path(output_dir).glob("kodim*.png"))
    if len(existing) >= 24:
        print(f"✓ Kodak dataset present ({len(existing)} images)")
        return output_dir
    print("Downloading Kodak dataset...")
    import urllib.request
    base_url = "http://r0k.us/graphics/kodak/kodak/"
    for i in range(1, 25):
        filename = f"kodim{i:02d}.png"
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            try:
                print(f"  Downloading {filename}...", end=' ')
                urllib.request.urlretrieve(base_url + filename, filepath)
                print("✓")
            except Exception as e:
                print(f"✗ {e}")
    return output_dir


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _prepare_data(img_np):
    """
    Build the 5D joint pixel array [x_norm, y_norm, r, g, b] used by all variants.
    Returns (data, h, w).
    """
    h, w, _ = img_np.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
    pixels = img_np.reshape(-1, 3)
    coords_norm = coords / np.array([[w, h]], dtype=np.float32)
    data = np.concatenate([coords_norm, pixels], axis=1).astype(np.float32)
    return data, h, w


def render_gaussians(means, covs, colors, weights, image_size):
    """Alpha-composite Gaussian splats onto a canvas."""
    h, w = image_size
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    weight_sum = np.zeros((h, w, 1), dtype=np.float32)
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    pos = np.dstack((X, Y))
    for mean, cov, color, wgt in zip(means, covs, colors, weights):
        try:
            rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
            pdf = rv.pdf(pos)
            pdf = pdf / (np.max(pdf) + 1e-8) * wgt
            pdf = np.expand_dims(pdf, axis=-1)
            canvas += pdf * color
            weight_sum += pdf
        except (np.linalg.LinAlgError, ValueError):
            continue
    render = np.divide(canvas, weight_sum, out=np.zeros_like(canvas), where=weight_sum > 1e-8)
    return np.clip(render, 0, 1)
