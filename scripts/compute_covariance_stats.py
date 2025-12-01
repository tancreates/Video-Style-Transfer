"""
Compute Covariance Statistics for Geometric Methods
Generates covariance matrices needed for MKL and IDT methods
"""

import os
import json
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from enhanced_color_transfer import EnhancedColorTransfer


def compute_covariance_for_director(director_name, frames_dir="Frames", output_dir="Palettes"):
    """
    Compute mean and covariance matrix for a director's frames
    
    Args:
        director_name: Name of director
        frames_dir: Base directory with frames
        output_dir: Where to save statistics
    """
    print(f"\n{'='*70}")
    print(f"Computing covariance statistics for: {director_name}")
    print(f"{'='*70}")
    
    # Initialize transfer object
    transfer = EnhancedColorTransfer()
    
    # Get frame paths
    director_frames = os.path.join(frames_dir, director_name)
    frame_paths = sorted(glob(f"{director_frames}/*.jpg"))
    
    if not frame_paths:
        print(f"Error: No frames found in {director_frames}")
        return None
    
    print(f"Found {len(frame_paths)} frames")
    
    # Collect pixels
    all_pixels = []
    
    for frame_path in tqdm(frame_paths, desc="Loading frames"):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        
        # Convert to lab
        lab = transfer.bgr_to_lab(frame)
        
        # Flatten
        pixels = lab.reshape(-1, 3)
        
        # Sample to save memory (1000 pixels per frame)
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[indices]
        
        all_pixels.append(pixels)
    
    # Concatenate
    all_pixels = np.vstack(all_pixels)
    print(f"Total pixels collected: {len(all_pixels)}")
    
    # Compute statistics
    print("Computing mean and covariance...")
    mean = np.mean(all_pixels, axis=0)
    cov = np.cov(all_pixels.T)
    
    # Create statistics dictionary
    stats = {
        'director': director_name,
        'mean': mean.tolist(),
        'covariance': cov.tolist(),
        'n_frames': len(frame_paths),
        'n_pixels': len(all_pixels),
        
        # Also store individual channel stats for backward compatibility
        'mean_l': float(mean[0]),
        'mean_alpha': float(mean[1]),
        'mean_beta': float(mean[2]),
        'std_l': float(np.sqrt(cov[0, 0])),
        'std_alpha': float(np.sqrt(cov[1, 1])),
        'std_beta': float(np.sqrt(cov[2, 2]))
    }
    
    # Save
    output_path = os.path.join(output_dir, f"{director_name}_stats.json")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nSaved to: {output_path}")
    print(f"Mean: {mean}")
    print(f"Covariance:\n{cov}")
    
    return stats


def main():
    """Compute covariance stats for all directors"""
    frames_dir = "Frames"
    
    # Find all directors
    if not os.path.exists(frames_dir):
        print(f"Error: {frames_dir} not found")
        return
    
    directors = [d for d in os.listdir(frames_dir) 
                if os.path.isdir(os.path.join(frames_dir, d))]
    
    if not directors:
        print(f"Error: No directors found in {frames_dir}")
        return
    
    print(f"Found {len(directors)} directors: {', '.join(directors)}")
    print("\nComputing covariance statistics...")
    
    success = []
    failed = []
    
    for director in directors:
        try:
            stats = compute_covariance_for_director(director)
            if stats:
                success.append(director)
        except Exception as e:
            print(f"\nError processing {director}: {e}")
            failed.append(director)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Successfully processed: {len(success)}/{len(directors)}")
    if success:
        print(f"Success: {', '.join(success)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main() 
