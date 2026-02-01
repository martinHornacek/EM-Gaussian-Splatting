"""
gaussian_splatting_2d.py - 2D Gaussian Splatting with Gradient Descent
Based on original 2D_Gaussian_Splatting.py implementation
"""

import os
import gc
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics_utils import evaluate_metrics, print_metrics, compute_ssim_loss
from utils.dataset_utils import select_images
from utils.em_utils import load_config, download_kodak_dataset


def generate_2D_gaussian_splatting(kernel_size, scale, rotation, coords, colours,
                                    image_size=(256, 256, 3), device="cpu"):
    """
    Generate 2D Gaussian splats (from original implementation)
    """
    batch_size = colours.shape[0]
    scale = scale.view(batch_size, 2)
    rotation = rotation.view(batch_size)

    # Rotation matrix
    cos_rot = torch.cos(rotation)
    sin_rot = torch.sin(rotation)
    R = torch.stack([
        torch.stack([cos_rot, -sin_rot], dim=-1),
        torch.stack([sin_rot, cos_rot], dim=-1),
    ], dim=-2)

    # Scale matrix
    S = torch.diag_embed(scale)

    # Covariance: R @ S @ S @ R^T
    covariance = R @ S @ S @ R.transpose(-1, -2)
    inv_covariance = torch.inverse(covariance)

    # Create kernel
    x = torch.linspace(-5, 5, kernel_size, device=device)
    y = torch.linspace(-5, 5, kernel_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    xy = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)

    z = torch.einsum("bxyi,bij,bxyj->bxy", xy, -0.5 * inv_covariance, xy)
    kernel = torch.exp(z) / (
        2 * torch.tensor(np.pi, device=device) *
        torch.sqrt(torch.det(covariance))
    ).view(batch_size, 1, 1)

    # Normalize
    kernel_max = kernel.amax(dim=(-2, -1), keepdim=True)
    kernel_normalized = kernel / kernel_max

    # RGB channels
    kernel_rgb = kernel_normalized.unsqueeze(1).expand(-1, 3, -1, -1)

    # Padding
    pad_h = image_size[0] - kernel_size
    pad_w = image_size[1] - kernel_size
    padding = (pad_w // 2, pad_w // 2 + pad_w % 2, pad_h // 2, pad_h // 2 + pad_h % 2)
    kernel_rgb_padded = F.pad(kernel_rgb, padding, "constant", 0)

    # Translation
    b, c, h, w = kernel_rgb_padded.shape
    theta = torch.zeros(b, 2, 3, dtype=torch.float32, device=device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = coords

    grid = F.affine_grid(theta, size=(b, c, h, w), align_corners=True)
    kernel_rgb_padded_translated = F.grid_sample(kernel_rgb_padded, grid, align_corners=True)

    # Apply colors
    rgb_values_reshaped = colours.unsqueeze(-1).unsqueeze(-1)
    final_image_layers = rgb_values_reshaped * kernel_rgb_padded_translated
    final_image = final_image_layers.sum(dim=0)
    final_image = torch.clamp(final_image, 0, 1)
    final_image = final_image.permute(1, 2, 0)

    return final_image


def combined_loss(pred, target, lambda_param=0.2):
    """Combined L1 + dSSIM loss"""
    l1loss = nn.L1Loss()
    l1 = l1loss(pred, target)
    dssim = compute_ssim_loss(pred, target)
    return (1 - lambda_param) * l1 + lambda_param * dssim, l1, dssim


def give_required_data(input_coords, image_size, image_array, device):
    """Prepare normalized coordinates and colors"""
    coords = torch.tensor(input_coords / [image_size[0], image_size[1]], device=device).float()
    center_coords_normalized = torch.tensor([0.5, 0.5], device=device).float()
    coords = (center_coords_normalized - coords) * 2.0

    colour_values = [image_array[coord[1], coord[0]] for coord in input_coords]
    colour_values_np = np.array(colour_values)
    colour_values_tensor = torch.tensor(colour_values_np, device=device).float()

    return colour_values_tensor, coords


def train_2d_gaussian_splatting(img_np, config, output_dir, image_name):
    """
    Train 2D Gaussian Splatting model
    """
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # Config
    gs_config = config['gaussian_splatting']
    num_samples = gs_config['primary_samples'] + gs_config['backup_samples']
    num_epochs = gs_config['num_epochs']
    lr = gs_config['learning_rate']
    kernel_size = gs_config['kernel_size']
    lambda_dssim = gs_config['lambda_dssim']
    
    h, w, _ = img_np.shape
    image_size = (w, h)
    
    # Prepare data
    target_tensor = torch.tensor(img_np, dtype=torch.float32, device=device)
    coords = np.random.randint(0, [w, h], size=(num_samples, 2))
    
    colour_values, pixel_coords = give_required_data(coords, image_size, img_np, device)
    
    # Initialize parameters
    colour_values = torch.logit(colour_values.clamp(1e-4, 1-1e-4))
    pixel_coords = torch.atanh(pixel_coords.clamp(-0.999, 0.999))
    
    scale_values = torch.logit(torch.rand(num_samples, 2, device=device).clamp(1e-4, 1-1e-4))
    rotation_values = torch.atanh(2 * torch.rand(num_samples, 1, device=device).clamp(1e-4, 1-1e-4) - 1)
    alpha_values = torch.logit(torch.rand(num_samples, 1, device=device).clamp(1e-4, 1-1e-4))
    
    W = nn.Parameter(torch.cat([scale_values, rotation_values, alpha_values,
                                colour_values, pixel_coords], dim=1))
    
    # Persistent mask for densification/pruning
    persistent_mask = torch.cat([
        torch.ones(gs_config['primary_samples'], dtype=torch.bool, device=device),
        torch.zeros(gs_config['backup_samples'], dtype=torch.bool, device=device)
    ], dim=0)
    current_marker = gs_config['primary_samples']
    
    # Optimizer
    optimizer = torch.optim.Adam([W], lr=lr)
    
    # Training history
    history = {
        'epoch': [],
        'loss': [],
        'l1': [],
        'dssim': [],
        'mse': [],
        'rmse': [],
        'psnr': [],
        'ssim': [],
        'active_gaussians': []
    }
    
    start_time = time.time()
    print(f"  Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Pruning
        if gs_config['pruning'] and epoch % (gs_config['densification_interval'] + 1) == 0 and epoch > 0:
            indices_to_remove = (torch.sigmoid(W[:, 3]) < 0.01).nonzero(as_tuple=True)[0]
            if len(indices_to_remove) > 0:
                persistent_mask[indices_to_remove] = False
                W.data[~persistent_mask] = 0.0
        
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Forward pass
        output = W[persistent_mask]
        batch_size = output.shape[0]
        
        scale = torch.sigmoid(output[:, 0:2])
        rotation = np.pi / 2 * torch.tanh(output[:, 2])
        alpha = torch.sigmoid(output[:, 3])
        colours = torch.sigmoid(output[:, 4:7])
        pixel_coords_norm = torch.tanh(output[:, 7:9])
        
        colours_with_alpha = colours * alpha.view(batch_size, 1)
        g_tensor_batch = generate_2D_gaussian_splatting(
            kernel_size, scale, rotation, pixel_coords_norm, colours_with_alpha,
            image_size, device=device
        )
        
        # Loss
        loss, l1_loss, dssim_loss = combined_loss(g_tensor_batch, target_tensor, lambda_dssim)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        if persistent_mask is not None:
            W.grad.data[~persistent_mask] = 0.0
        
        # Densification
        if gs_config['densification'] and epoch % gs_config['densification_interval'] == 0 and epoch > 0:
            gradient_norms = torch.norm(W.grad[persistent_mask][:, 7:9], dim=1, p=2)
            gaussian_norms = torch.norm(torch.sigmoid(W.data[persistent_mask][:, 0:2]), dim=1, p=2)
            
            sorted_grads, sorted_grads_indices = torch.sort(gradient_norms, descending=True)
            sorted_gauss, sorted_gauss_indices = torch.sort(gaussian_norms, descending=True)
            
            large_gradient_mask = sorted_grads > gs_config['gradient_threshold']
            large_gradient_indices = sorted_grads_indices[large_gradient_mask]
            large_gauss_mask = sorted_gauss > gs_config['gaussian_threshold']
            large_gauss_indices = sorted_gauss_indices[large_gauss_mask]
            
            common_indices_mask = torch.isin(large_gradient_indices, large_gauss_indices)
            common_indices = large_gradient_indices[common_indices_mask]
            distinct_indices = large_gradient_indices[~common_indices_mask]
            
            # Split
            if len(common_indices) > 0 and current_marker < num_samples:
                n_split = min(len(common_indices), num_samples - current_marker)
                start_idx = current_marker
                end_idx = current_marker + n_split
                persistent_mask[start_idx:end_idx] = True
                W.data[start_idx:end_idx, :] = W.data[common_indices[:n_split], :]
                W.data[start_idx:end_idx, 0:2] /= 1.6
                W.data[common_indices[:n_split], 0:2] /= 1.6
                current_marker += n_split
            
            # Clone
            if len(distinct_indices) > 0 and current_marker < num_samples:
                n_clone = min(len(distinct_indices), num_samples - current_marker)
                start_idx = current_marker
                end_idx = current_marker + n_clone
                persistent_mask[start_idx:end_idx] = True
                W.data[start_idx:end_idx, :] = W.data[distinct_indices[:n_clone], :]
                positional_gradients = W.grad[distinct_indices[:n_clone], 7:9]
                gradient_magnitudes = torch.norm(positional_gradients, dim=1, keepdim=True)
                normalized_gradients = positional_gradients / (gradient_magnitudes + 1e-8)
                W.data[start_idx:end_idx, 7:9] += 0.01 * normalized_gradients
                current_marker += n_clone
        
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            mse = torch.mean((g_tensor_batch - target_tensor) ** 2).item()
            rmse = np.sqrt(mse)
            psnr = 10.0 * np.log10(1.0 / (mse + 1e-10))
            ssim_val = 1.0 - dssim_loss.item()
            
            history['epoch'].append(epoch + 1)
            history['loss'].append(loss.item())
            history['l1'].append(l1_loss.item())
            history['dssim'].append(dssim_loss.item())
            history['mse'].append(mse)
            history['rmse'].append(rmse)
            history['psnr'].append(psnr)
            history['ssim'].append(ssim_val)
            history['active_gaussians'].append(int(persistent_mask.sum().item()))
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f"    Epoch {epoch+1}/{num_epochs}: Loss={loss.item():.6f}, "
                  f"PSNR={psnr:.2f} dB, SSIM={ssim_val:.4f}, "
                  f"Active={persistent_mask.sum()}, Epoch Time={epoch_time:.2f}s")
        
        # Plot intermediate results
        if config['experiment']['save_plots'] and (epoch + 1) % config['experiment']['plot_interval'] == 0:
            save_intermediate_plot(g_tensor_batch, target_tensor, history, 
                                  output_dir, image_name, epoch + 1)
    
    total_time = time.time() - start_time
    
    # Final render
    with torch.no_grad():
        output = W[persistent_mask]
        batch_size = output.shape[0]
        
        scale = torch.sigmoid(output[:, 0:2])
        rotation = np.pi / 2 * torch.tanh(output[:, 2])
        alpha = torch.sigmoid(output[:, 3])
        colours = torch.sigmoid(output[:, 4:7])
        pixel_coords_norm = torch.tanh(output[:, 7:9])
        
        colours_with_alpha = colours * alpha.view(batch_size, 1)
        final_render = generate_2D_gaussian_splatting(
            kernel_size, scale, rotation, pixel_coords_norm, colours_with_alpha,
            image_size, device=device
        )
        final_render_np = final_render.cpu().numpy()
    
    return final_render_np, history, {'total_time': total_time}


def save_intermediate_plot(pred, gt, history, output_dir, image_name, epoch):
    """Save intermediate training visualization"""
    fig = plt.figure(figsize=(15, 10))
    
    # Layout: 2 rows, 3 columns
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(gt.cpu().detach().numpy())
    ax1.set_title('Ground Truth', fontsize=12)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(pred.cpu().detach().numpy())
    ax2.set_title(f'2D-GS (Epoch {epoch})', fontsize=12)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    diff = np.abs(pred.cpu().detach().numpy() - gt.cpu().detach().numpy())
    ax3.imshow(diff)
    ax3.set_title('Absolute Difference', fontsize=12)
    ax3.axis('off')
    
    # Row 2: Metrics plots
    epochs = history['epoch']
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(epochs, history['loss'], 'b-', label='Total Loss', linewidth=1.5)
    ax4.plot(epochs, history['l1'], 'r--', label='L1', linewidth=1)
    ax4.plot(epochs, history['dssim'], 'g--', label='dSSIM', linewidth=1)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Loss Evolution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, history['psnr'], 'b-', linewidth=1.5)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('PSNR (dB)')
    ax5.set_title('PSNR Evolution')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(epochs, history['rmse'], 'r-', label='RMSE', linewidth=1.5)
    ax6.plot(epochs, history['ssim'], 'g-', label='SSIM', linewidth=1.5)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Metric Value')
    ax6.set_title('RMSE & SSIM Evolution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / f'{image_name}_epoch_{epoch:04d}.png', 
                dpi=150, 
                bbox_inches='tight')
    plt.close()


def create_final_visualizations(gt_np, render_np, history, metrics, output_dir, image_name):
    """Create final visualization with metrics evolution"""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Images
    ax1 = fig.add_subplot(gs[0, :])
    combined = np.concatenate([gt_np, render_np], axis=1)
    ax1.imshow(combined)
    title = f"Ground Truth vs 2D-GS Reconstruction\nPSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}, RMSE: {metrics['rmse']:.6f}"
    ax1.set_title(title, fontsize=14)
    ax1.axis('off')
    
    # Row 2 & 3: Metrics evolution
    epochs = history['epoch']
    
    axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2])
    ]
    
    # Loss
    axes[0].plot(epochs, history['loss'], 'b-', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)
    
    # L1 and dSSIM
    axes[1].plot(epochs, history['l1'], 'r-', label='L1', linewidth=1.5)
    axes[1].plot(epochs, history['dssim'], 'g-', label='dSSIM', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss Component')
    axes[1].set_title('L1 & dSSIM Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # PSNR
    axes[2].plot(epochs, history['psnr'], 'b-', linewidth=1.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('PSNR (dB)')
    axes[2].set_title('PSNR Evolution')
    axes[2].grid(True, alpha=0.3)
    
    # RMSE
    axes[3].plot(epochs, history['rmse'], 'r-', linewidth=1.5)
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('RMSE')
    axes[3].set_title('RMSE Evolution')
    axes[3].grid(True, alpha=0.3)
    
    # SSIM
    axes[4].plot(epochs, history['ssim'], 'g-', linewidth=1.5)
    axes[4].set_xlabel('Epoch')
    axes[4].set_ylabel('SSIM')
    axes[4].set_title('SSIM Evolution')
    axes[4].grid(True, alpha=0.3)
    
    # Active Gaussians
    axes[5].plot(epochs, history['active_gaussians'], 'm-', linewidth=1.5)
    axes[5].set_xlabel('Epoch')
    axes[5].set_ylabel('Count')
    axes[5].set_title('Active Gaussians')
    axes[5].grid(True, alpha=0.3)
    
    plt.savefig(output_dir / f'{image_name}_final.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_gaussian_splatting_2d(config_path='config.yml'):
    """Main 2D Gaussian Splatting runner"""
    
    # Load config
    config = load_config(config_path)
    
    if not config['gaussian_splatting']['enabled']:
        print("2D Gaussian Splatting disabled in config")
        return
    
    print("\n" + "="*70)
    print("2D GAUSSIAN SPLATTING")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['experiment']['output_dir']) / f"2dgs_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Download dataset
    kodak_dir = download_kodak_dataset(config['dataset']['path'])
    image_paths = select_images(kodak_dir, config['dataset'])
    
    print(f"Processing {len(image_paths)} images")
    print(f"Gaussians: {config['gaussian_splatting']['primary_samples']}")
    print(f"Epochs: {config['gaussian_splatting']['num_epochs']}")
    
    # Process each image
    all_results = []
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] {img_path.name}")
        print("-" * 70)
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = img.resize(tuple(config['dataset']['image_size']), Image.LANCZOS)
        gt_np = np.array(img).astype(np.float32) / 255.0
        
        # Train 2D-GS
        render_np, history, timing_info = train_2d_gaussian_splatting(
            gt_np, config, output_dir, img_path.stem
        )
        
        # Evaluate final metrics
        metrics = evaluate_metrics(
            render_np, gt_np,
            n_components=config['gaussian_splatting']['primary_samples'],
            compute_lpips_flag=config['metrics']['compute_lpips']
        )
        
        # Print results
        print_metrics(metrics, method_name="2D-GS", epoch=None)
        print(f"  Total time: {timing_info['total_time']:.2f}s")
        
        # Create final visualizations
        if config['experiment']['save_plots']:
            create_final_visualizations(gt_np, render_np, history, metrics, 
                                       output_dir, img_path.stem)
        
        # Save results
        result = {
            'image': img_path.name,
            'method': '2D-GS',
            **metrics,
            **timing_info
        }
        all_results.append(result)
        
        # Save history
        if config['experiment']['save_raw_data']:
            import pandas as pd
            history_df = pd.DataFrame(history)
            history_df.to_csv(output_dir / f'{img_path.stem}_history.csv', index=False)
        
        # Save render
        render_img = (render_np * 255).astype(np.uint8)
        Image.fromarray(render_img).save(output_dir / f'{img_path.stem}_render.png')
    
    # Save all results
    import pandas as pd
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'results.csv', index=False)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    summary = results_df[['psnr', 'rmse', 'ssim', 'lpips', 'bits_per_pixel',
                          'compression_ratio', 'total_time']].describe()
    print(summary)
    
    # Save summary
    summary.to_csv(output_dir / 'summary.csv')
    
    print(f"\n✓ 2D Gaussian Splatting complete! Results saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run 2D Gaussian Splatting')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config file')
    args = parser.parse_args()
    
    run_gaussian_splatting_2d(args.config)