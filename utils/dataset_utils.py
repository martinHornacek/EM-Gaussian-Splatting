"""
dataset_utils.py - Utilities for dataset handling and image selection
"""

import numpy as np
from pathlib import Path


def select_images(dataset_path, config_dataset):
    """
    Select images based on configuration settings.
    
    Args:
        dataset_path: Path to the kodak dataset directory
        config_dataset: Dataset configuration dict with:
            - selection_mode: "full", "random", or "list"
            - random_count: Number of random images (for random mode)
            - image_list: List of image numbers (for list mode)
    
    Returns:
        List of Path objects for selected images, sorted
    """
    # Get all available kodak images
    all_images = sorted(list(Path(dataset_path).glob("kodim*.png")))
    
    if not all_images:
        raise FileNotFoundError(f"No kodak images found in {dataset_path}")
    
    selection_mode = config_dataset.get('selection_mode', 'full')
    
    if selection_mode == 'full':
        return all_images
    
    elif selection_mode == 'random':
        random_count = config_dataset.get('random_count', 5)
        if random_count > len(all_images):
            print(f"⚠ Requested {random_count} images but only {len(all_images)} available. Using all.")
            return all_images
        
        # Randomly select images
        np.random.seed(42)  # For reproducibility
        selected_indices = np.random.choice(len(all_images), size=random_count, replace=False)
        selected_indices = sorted(selected_indices)
        selected_images = [all_images[i] for i in selected_indices]
        print(f"Selected {len(selected_images)} random images from {len(all_images)} available")
        return selected_images
    
    elif selection_mode == 'list':
        image_list = config_dataset.get('image_list', [])
        
        if not image_list:
            print("⚠ List mode selected but image_list is empty. Using all images.")
            return all_images
        
        # Map image numbers to paths
        selected_images = []
        for img_num in sorted(image_list):
            # Find image with matching number
            for img_path in all_images:
                # Extract number from kodim01.png -> 1
                num_str = img_path.stem.replace('kodim', '')
                if int(num_str) == img_num:
                    selected_images.append(img_path)
                    break
        
        if not selected_images:
            print(f"⚠ No images found matching list {image_list}. Using all images.")
            return all_images
        
        print(f"Selected {len(selected_images)} images from list: {image_list}")
        return selected_images
    
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}. "
                        f"Must be 'full', 'random', or 'list'")
