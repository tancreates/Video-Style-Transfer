"""
Color Statistics - Compute statistical descriptors using proper lab color space
Based on Reinhard et al. "Color Transfer between Images" (2001)
"""

import cv2
import numpy as np
import json
import os
from glob import glob
from tqdm import tqdm


class ColorStatisticsAnalyzer:
    """Analyze and compute color statistics for director palettes in lab space"""
    
    def __init__(self):
        pass
    
    def bgr_to_lab(self, bgr_image):
        """
        Convert BGR to lab color space (Ruderman et al.)
        Same as ColorTransfer class for consistency
        """
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = np.where(rgb == 0, 1e-10, rgb)
        
        # RGB to LMS
        lms_matrix = np.array([
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444]
        ])
        
        h, w, c = rgb.shape
        rgb_reshaped = rgb.reshape(-1, 3).T
        lms = lms_matrix @ rgb_reshaped
        lms = lms.T.reshape(h, w, 3)
        lms = np.log10(lms + 1e-10)
        
        # LMS to lab
        lab_matrix = np.array([
            [1/np.sqrt(3), 0, 0],
            [0, 1/np.sqrt(6), 0],
            [0, 0, 1/np.sqrt(2)]
        ]) @ np.array([
            [1, 1, 1],
            [1, 1, -2],
            [1, -1, 0]
        ])
        
        lms_reshaped = lms.reshape(-1, 3).T
        lab = lab_matrix @ lms_reshaped
        lab = lab.T.reshape(h, w, 3)
        
        return lab.astype(np.float32)
    
    def compute_frame_statistics(self, frame):
        """
        Compute color statistics directly from a frame in lab space
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            stats: Dictionary with statistical descriptors in lab
        """
        # Convert to lab
        lab_frame = self.bgr_to_lab(frame)
        
        # Compute channel statistics
        mean_l = np.mean(lab_frame[:, :, 0])
        mean_alpha = np.mean(lab_frame[:, :, 1])
        mean_beta = np.mean(lab_frame[:, :, 2])
        
        std_l = np.std(lab_frame[:, :, 0])
        std_alpha = np.std(lab_frame[:, :, 1])
        std_beta = np.std(lab_frame[:, :, 2])
        
        stats = {
            'mean_l': float(mean_l),
            'mean_alpha': float(mean_alpha),
            'mean_beta': float(mean_beta),
            'std_l': float(std_l),
            'std_alpha': float(std_alpha),
            'std_beta': float(std_beta),
        }
        
        return stats
    
    def compute_director_statistics(self, frames_dir, output_dir=None, director_name="Director"):
        """
        Compute aggregated statistics from all director frames in lab space
        
        Args:
            frames_dir: Directory containing extracted frames
            output_dir: Directory to save statistics
            director_name: Name of director
            
        Returns:
            stats: Dictionary with aggregated statistics
        """
        if output_dir is None:
            output_dir = "Palettes"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect all frame paths
        frame_paths = sorted(glob(os.path.join(frames_dir, "*.jpg")))
        
        if len(frame_paths) == 0:
            raise ValueError(f"No frames found in {frames_dir}")
        
        print(f"Computing statistics from {len(frame_paths)} frames for {director_name}...")
        
        # Collect statistics from all frames
        all_stats = []
        
        for frame_path in tqdm(frame_paths, desc="Computing frame statistics"):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            frame_stats = self.compute_frame_statistics(frame)
            all_stats.append(frame_stats)
        
        # Aggregate statistics
        aggregated_stats = {
            'director': director_name,
            'n_frames': len(all_stats),
            
            # Mean of means (overall average)
            'mean_l': float(np.mean([s['mean_l'] for s in all_stats])),
            'mean_alpha': float(np.mean([s['mean_alpha'] for s in all_stats])),
            'mean_beta': float(np.mean([s['mean_beta'] for s in all_stats])),
            
            # Mean of stds (average variation)
            'std_l': float(np.mean([s['std_l'] for s in all_stats])),
            'std_alpha': float(np.mean([s['std_alpha'] for s in all_stats])),
            'std_beta': float(np.mean([s['std_beta'] for s in all_stats])),
            
            # Cross-frame variance (how much frames differ from each other)
            'cross_frame_var_l': float(np.var([s['mean_l'] for s in all_stats])),
            'cross_frame_var_alpha': float(np.var([s['mean_alpha'] for s in all_stats])),
            'cross_frame_var_beta': float(np.var([s['mean_beta'] for s in all_stats])),
        }
        
        # Save statistics
        stats_path = os.path.join(output_dir, f"{director_name}_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(aggregated_stats, f, indent=2)
        
        print(f"Saved statistics to {stats_path}")
        
        return aggregated_stats
    
    def print_statistics(self, stats):
        """Pretty print statistics"""
        print(f"\n{'='*60}")
        print(f"COLOR STATISTICS (lab space): {stats.get('director', 'Unknown')}")
        print(f"{'='*60}")
        print(f"Frames analyzed: {stats.get('n_frames', 'N/A')}")
        print(f"\nlab Channel Means:")
        print(f"  l (lightness):  {stats['mean_l']:.4f}")
        print(f"  a (red-green):  {stats['mean_alpha']:.4f}")
        print(f"  b (yellow-blue): {stats['mean_beta']:.4f}")
        print(f"\nlab Channel Std Dev:")
        print(f"  l: {stats['std_l']:.4f}")
        print(f"  a: {stats['std_alpha']:.4f}")
        print(f"  b: {stats['std_beta']:.4f}")
        
        if 'cross_frame_var_l' in stats:
            print(f"\nCross-frame Consistency:")
            print(f"  l variance: {stats['cross_frame_var_l']:.4f}")
            print(f"  a variance: {stats['cross_frame_var_alpha']:.4f}")
            print(f"  b variance: {stats['cross_frame_var_beta']:.4f}")


def main():
    """Main function to compute statistics from director frames"""
    # Configuration
    FRAMES_DIR = "Frames/Avatar"
    OUTPUT_DIR = "Palettes"
    DIRECTOR_NAME = "Avatar"
    
    # Create analyzer
    analyzer = ColorStatisticsAnalyzer()
    
    # Compute statistics
    print(f"\n{'='*60}")
    print(f"COMPUTING COLOR STATISTICS (lab space) FOR: {DIRECTOR_NAME}")
    print(f"{'='*60}\n")
    
    stats = analyzer.compute_director_statistics(
        frames_dir=FRAMES_DIR,
        output_dir=OUTPUT_DIR,
        director_name=DIRECTOR_NAME
    )
    
    # Print results
    analyzer.print_statistics(stats)


if __name__ == "__main__":
    main()
