"""
em_algorithm.py - EM Algorithm for Gaussian Splatting
Fits Gaussian Mixture Model and renders the result
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from PIL import Image
from utils.metrics_utils import evaluate_metrics, print_metrics
from utils.dataset_utils import select_images

from utils.em_utils import load_config, download_kodak_dataset, fit_gmm_and_render


def create_visualizations(gt_np, render_np, metrics, output_dir, image_name):
    """Create comparison plot"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ground Truth
    axes[0].imshow(gt_np)
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')
    
    # EM Reconstruction
    axes[1].imshow(render_np)
    title = f"EM Reconstruction\nPSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}"
    axes[1].set_title(title, fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{image_name}_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def run_em_algorithm(config_path='config.yml'):
    """Main EM algorithm runner"""
    
    # Load config
    config = load_config(config_path)
    
    if not config['em']['enabled']:
        print("EM algorithm disabled in config")
        return
    
    print("\n" + "="*70)
    print("EM ALGORITHM FOR GAUSSIAN SPLATTING")
    print("="*70)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['experiment']['output_dir']) / f"em_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / 'config.yml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Download dataset
    kodak_dir = download_kodak_dataset(config['dataset']['path'])
    image_paths = select_images(kodak_dir, config['dataset'])
    
    print(f"Processing {len(image_paths)} images")
    print(f"Components: {config['em']['n_components']}")
    
    # Process each image
    all_results = []
    
    for idx, img_path in enumerate(image_paths, 1):
        print(f"\n[{idx}/{len(image_paths)}] {img_path.name}")
        print("-" * 70)
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        img = img.resize(tuple(config['dataset']['image_size']), Image.LANCZOS)
        gt_np = np.array(img).astype(np.float32) / 255.0
        
        # Run EM
        render_np, timing_info = fit_gmm_and_render(gt_np, config['em'])
        
        # Evaluate metrics
        metrics = evaluate_metrics(
            render_np, gt_np,
            n_components=config['em']['n_components'],
            compute_lpips_flag=config['metrics']['compute_lpips']
        )
        
        # Print results
        print_metrics(metrics, method_name="EM", epoch=None)
        print(f"  Total time: {timing_info['total_time']:.2f}s")
        
        # Create visualizations
        if config['experiment']['save_plots']:
            create_visualizations(gt_np, render_np, metrics, output_dir, img_path.stem)
        
        # Save results
        result = {
            'image': img_path.name,
            'method': 'EM',
            **metrics,
            **timing_info
        }
        all_results.append(result)
        
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
    
    print(f"\n✓ EM Algorithm complete! Results saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run EM Algorithm for Gaussian Splatting')
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config file')
    args = parser.parse_args()
    
    run_em_algorithm(args.config)