"""
em_utils.py - Shared utilities for EM and plotting
Contains: load_config, download_kodak_dataset, render_gaussians, fit_gmm_and_render
"""

import os
import time
import yaml
import numpy as np
from pathlib import Path
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal


def load_config(config_path='config.yml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def download_kodak_dataset(output_dir="./kodak"):
    """Download Kodak dataset if not present"""
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


def render_gaussians(means, covs, colors, weights, image_size):
    """Render Gaussian splats using scipy"""
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


def fit_gmm_and_render(img_np, config=None, n_components=None, verbose=False):
    """Fit Gaussian Mixture Model and render.

    Args:
        img_np: HxWx3 numpy image in [0,1]
        config: dict with EM params (may include 'n_components')
        n_components: if provided, overrides config['n_components']
        verbose: print status messages

    Returns:
        render_np, timing_info dict
    """
    h, w, _ = img_np.shape

    # Prepare data: [x, y, r, g, b] (5D joint space)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
    colors = img_np.reshape(-1, 3)
    coords_norm = coords / np.array([[w, h]])
    data = np.concatenate([coords_norm, colors], axis=1)

    # Resolve parameters
    if config is None:
        config = {}
    n_comp = n_components if n_components is not None else config.get('n_components')
    if n_comp is None:
        raise ValueError("n_components must be provided either as param or in config")

    covariance_type = config.get('covariance_type', 'full')
    max_iter = config.get('max_iter', 100)
    init_params = config.get('init_params', 'kmeans')

    if verbose:
        print(f"  Fitting GMM with {n_comp} components...")
    start_time = time.time()

    gmm = GaussianMixture(
        n_components=n_comp,
        covariance_type=covariance_type,
        max_iter=max_iter,
        init_params=init_params,
        random_state=42,
        verbose=0
    )
    gmm.fit(data)

    fit_time = time.time() - start_time
    if verbose:
        print(f"  ✓ GMM fitted in {fit_time:.2f}s (converged: {gmm.converged_}, iterations: {gmm.n_iter_})")

    # Extract parameters
    means_pos = gmm.means_[:, :2] * np.array([[w, h]])
    colors = gmm.means_[:, 2:]
    covs_pos = gmm.covariances_[:, :2, :2] * np.array([[[w, h], [w, h]]])**2
    weights = gmm.weights_

    # Render
    if verbose:
        print(f"  Rendering...")
    render_start = time.time()
    render_np = render_gaussians(means_pos, covs_pos, colors, weights, (h, w))
    render_time = time.time() - render_start

    total_time = time.time() - start_time

    return render_np, {
        'fit_time': fit_time,
        'render_time': render_time,
        'total_time': total_time,
        'converged': gmm.converged_,
        'n_iter': gmm.n_iter_
    }
