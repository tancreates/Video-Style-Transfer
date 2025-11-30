"""
Color Transfer - Reinhard et al. "Color Transfer between Images" Implementation
Proper lαβ color space with correct conversion and statistics transfer
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path


class ColorTransfer:
    """
    Color transfer using Reinhard et al. (2001) algorithm
    Implements proper lαβ color space conversion and statistical matching
    """
    
    def __init__(self, target_stats=None):
        """
        Initialize color transfer
        
        Args:
            target_stats: Dictionary with target color statistics (lαβ space)
        """
        self.target_stats = target_stats
    
    def load_director_stats(self, stats_path):
        """Load director statistics from JSON file"""
        with open(stats_path, 'r') as f:
            self.target_stats = json.load(f)
        return self.target_stats
    
    def bgr_to_lab(self, bgr_image):
        """
        Convert BGR to lαβ color space (Ruderman et al.)
        This is the proper decorrelated color space from the Reinhard paper
        
        Args:
            bgr_image: Input BGR image (uint8)
            
        Returns:
            lab_image: Image in lαβ space (float32)
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Avoid log(0)
        rgb = np.where(rgb == 0, 1e-10, rgb)
        
        # Convert RGB to LMS (long, medium, short cone responses)
        # Using the transformation matrix from Reinhard et al.
        lms_matrix = np.array([
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444]
        ])
        
        # Reshape for matrix multiplication
        h, w, c = rgb.shape
        rgb_reshaped = rgb.reshape(-1, 3).T  # (3, h*w)
        
        # RGB to LMS
        lms = lms_matrix @ rgb_reshaped
        lms = lms.T.reshape(h, w, 3)
        
        # Take logarithm (working in log space for better distribution)
        lms = np.log10(lms + 1e-10)
        
        # Convert LMS to lαβ (decorrelated color space)
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
    
    def lab_to_bgr(self, lab_image):
        """
        Convert lαβ color space back to BGR
        
        Args:
            lab_image: Image in lαβ space (float32)
            
        Returns:
            bgr_image: BGR image (uint8)
        """
        h, w, c = lab_image.shape
        
        # lαβ to LMS
        lab_to_lms_matrix = np.array([
            [1, 1, 1],
            [1, 1, -1],
            [1, -2, 0]
        ]) @ np.array([
            [np.sqrt(3)/3, 0, 0],
            [0, np.sqrt(6)/6, 0],
            [0, 0, np.sqrt(2)/2]
        ])
        
        lab_reshaped = lab_image.reshape(-1, 3).T
        lms = lab_to_lms_matrix @ lab_reshaped
        lms = lms.T.reshape(h, w, 3)
        
        # Antilog
        lms = np.power(10, lms)
        
        # LMS to RGB
        lms_to_rgb_matrix = np.array([
            [4.4679, -3.5873, 0.1193],
            [-1.2186, 2.3809, -0.1624],
            [0.0497, -0.2439, 1.2045]
        ])
        
        lms_reshaped = lms.reshape(-1, 3).T
        rgb = lms_to_rgb_matrix @ lms_reshaped
        rgb = rgb.T.reshape(h, w, 3)
        
        # Clip and scale to [0, 255]
        rgb = np.clip(rgb, 0, 1) * 255.0
        
        # Convert RGB to BGR
        bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        return bgr
    
    def transfer_lab_statistics(self, source_frame, target_stats=None, strength=1.0):
        """
        Transfer color statistics using Reinhard et al. method
        Properly implements the algorithm from the paper
        
        Args:
            source_frame: Source image in BGR format
            target_stats: Target statistics dict (uses self.target_stats if None)
            strength: Transfer strength (0=no transfer, 1=full transfer)
            
        Returns:
            transferred_frame: Image with transferred colors (BGR)
        """
        if target_stats is None:
            target_stats = self.target_stats
        
        if target_stats is None:
            raise ValueError("No target statistics provided")
        
        # Convert source to lαβ
        source_lab = self.bgr_to_lab(source_frame)
        
        # Compute source statistics
        source_mean = np.array([
            np.mean(source_lab[:, :, 0]),
            np.mean(source_lab[:, :, 1]),
            np.mean(source_lab[:, :, 2])
        ])
        
        source_std = np.array([
            np.std(source_lab[:, :, 0]),
            np.std(source_lab[:, :, 1]),
            np.std(source_lab[:, :, 2])
        ])
        
        # Target statistics
        target_mean = np.array([
            target_stats['mean_l'],
            target_stats['mean_alpha'],
            target_stats['mean_beta']
        ])
        
        target_std = np.array([
            target_stats['std_l'],
            target_stats['std_alpha'],
            target_stats['std_beta']
        ])
        
        # Avoid division by zero
        source_std = np.where(source_std < 1e-6, 1.0, source_std)
        
        # Apply color transfer for each channel
        result = np.zeros_like(source_lab)
        for i in range(3):
            # Step 1: Subtract source mean
            centered = source_lab[:, :, i] - source_mean[i]
            
            # Step 2: Scale by std ratio
            scaled = centered * (target_std[i] / source_std[i])
            
            # Step 3: Add target mean
            transferred = scaled + target_mean[i]
            
            # Blend with original based on strength
            result[:, :, i] = source_lab[:, :, i] * (1 - strength) + transferred * strength
        
        # Convert back to BGR
        result_bgr = self.lab_to_bgr(result)
        
        return result_bgr
    
    def transfer_selective_channels(self, source_frame, target_stats=None, 
                                    transfer_L=True, transfer_a=True, transfer_b=True,
                                    strength=1.0):
        """
        Selective channel transfer - choose which LAB channels to transfer
        
        Args:
            source_frame: Source image in BGR
            target_stats: Target statistics
            transfer_L: Whether to transfer lightness
            transfer_a: Whether to transfer a channel (green-red)
            transfer_b: Whether to transfer b channel (blue-yellow)
            strength: Transfer strength
            
        Returns:
            transferred_frame: BGR image
        """
        if target_stats is None:
            target_stats = self.target_stats
        
        # Convert to LAB
        source_lab = cv2.cvtColor(source_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        result = source_lab.copy()
        
        # Compute source stats
        source_mean = np.mean(source_lab, axis=(0, 1))
        source_std = np.std(source_lab, axis=(0, 1))
        source_std = np.where(source_std < 1e-6, 1.0, source_std)
        
        # Target stats
        target_mean = np.array([
            target_stats['mean_L'],
            target_stats['mean_a'],
            target_stats['mean_b']
        ])
        
        target_std = np.array([
            target_stats['std_L'],
            target_stats['std_a'],
            target_stats['std_b']
        ])
        
        # Transfer selected channels
        transfer_mask = [transfer_L, transfer_a, transfer_b]
        
        for i, should_transfer in enumerate(transfer_mask):
            if should_transfer:
                normalized = (source_lab[:, :, i] - source_mean[i]) / source_std[i]
                transferred = normalized * target_std[i] + target_mean[i]
                result[:, :, i] = source_lab[:, :, i] * (1 - strength) + transferred * strength
        
        # Clip values
        result[:, :, 0] = np.clip(result[:, :, 0], 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1], 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2], 0, 255)
        
        # Convert back
        result_bgr = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result_bgr
    
    def transfer_with_luminance_preservation(self, source_frame, target_stats=None, 
                                            strength=1.0, luminance_weight=0.5):
        """
        Transfer colors while preserving source luminance structure
        Useful for maintaining original lighting/contrast
        
        Args:
            source_frame: Source BGR image
            target_stats: Target statistics
            strength: Color transfer strength
            luminance_weight: How much to preserve original luminance (0-1)
            
        Returns:
            transferred_frame: BGR image
        """
        if target_stats is None:
            target_stats = self.target_stats
        
        # Convert to LAB
        source_lab = cv2.cvtColor(source_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        original_L = source_lab[:, :, 0].copy()
        
        # Transfer all channels
        transferred = self.transfer_lab_statistics(source_frame, target_stats, strength)
        transferred_lab = cv2.cvtColor(transferred, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Blend luminance
        transferred_lab[:, :, 0] = (
            original_L * luminance_weight + 
            transferred_lab[:, :, 0] * (1 - luminance_weight)
        )
        
        # Clip and convert
        transferred_lab[:, :, 0] = np.clip(transferred_lab[:, :, 0], 0, 255)
        result_bgr = cv2.cvtColor(transferred_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result_bgr
    
    def adaptive_transfer(self, source_frame, target_stats=None, 
                         strength=1.0, preserve_extremes=True):
        """
        Adaptive transfer that preserves extreme values (highlights/shadows)
        
        Args:
            source_frame: Source BGR image
            target_stats: Target statistics
            strength: Transfer strength
            preserve_extremes: Whether to preserve very bright/dark regions
            
        Returns:
            transferred_frame: BGR image
        """
        if target_stats is None:
            target_stats = self.target_stats
        
        # Convert to LAB
        source_lab = cv2.cvtColor(source_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Create mask for extreme values
        if preserve_extremes:
            # Preserve very dark (L < 20) and very bright (L > 235) regions
            extreme_mask = ((source_lab[:, :, 0] < 20) | (source_lab[:, :, 0] > 235)).astype(np.float32)
            # Smooth the mask
            extreme_mask = cv2.GaussianBlur(extreme_mask, (15, 15), 5)
            extreme_mask = np.stack([extreme_mask] * 3, axis=2)
        else:
            extreme_mask = np.zeros_like(source_lab)
        
        # Apply transfer
        transferred = self.transfer_lab_statistics(source_frame, target_stats, strength)
        transferred_lab = cv2.cvtColor(transferred, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Blend based on extreme mask
        result_lab = transferred_lab * (1 - extreme_mask) + source_lab * extreme_mask
        
        # Clip and convert
        result_lab = np.clip(result_lab, 0, 255)
        result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result_bgr


class HistogramTransfer:
    """
    Histogram-based color transfer techniques
    """
    
    def __init__(self):
        pass
    
    def match_histogram_1d(self, source, reference_hist_cdf):
        """
        Match 1D histogram using CDF matching
        
        Args:
            source: Source channel (2D array)
            reference_hist_cdf: Reference CDF
            
        Returns:
            matched: Matched channel
        """
        # Compute source histogram and CDF
        source_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
        source_cdf = source_hist.cumsum()
        source_cdf = source_cdf / source_cdf[-1]  # Normalize
        
        # Create lookup table
        lookup_table = np.zeros(256, dtype=np.uint8)
        
        for i in range(256):
            # Find closest value in reference CDF
            j = np.searchsorted(reference_hist_cdf, source_cdf[i])
            lookup_table[i] = min(j, 255)
        
        # Apply lookup
        matched = cv2.LUT(source.astype(np.uint8), lookup_table)
        
        return matched
    
    def match_histogram_3d(self, source_frame, reference_stats):
        """
        Match color histogram in LAB space
        
        Args:
            source_frame: Source BGR image
            reference_stats: Dictionary with reference histogram info
            
        Returns:
            matched_frame: BGR image with matched histogram
        """
        # Convert to LAB
        source_lab = cv2.cvtColor(source_frame, cv2.COLOR_BGR2LAB)
        result_lab = source_lab.copy()
        
        # Match each channel independently
        for i in range(3):
            # Compute reference CDF from normal distribution approximation
            # Using mean and std from reference stats
            channel_name = ['L', 'a', 'b'][i]
            ref_mean = reference_stats.get(f'mean_{channel_name}', 128)
            ref_std = reference_stats.get(f'std_{channel_name}', 30)
            
            # Create reference histogram (Gaussian approximation)
            ref_hist = np.zeros(256)
            x = np.arange(256)
            ref_hist = np.exp(-0.5 * ((x - ref_mean) / (ref_std + 1e-6)) ** 2)
            ref_hist = ref_hist / ref_hist.sum()
            ref_cdf = ref_hist.cumsum()
            
            # Match channel
            result_lab[:, :, i] = self.match_histogram_1d(source_lab[:, :, i], ref_cdf)
        
        # Convert back to BGR
        result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result_bgr
    
    def histogram_equalization_lab(self, source_frame, equalize_channels='L'):
        """
        Apply histogram equalization in LAB space
        
        Args:
            source_frame: Source BGR image
            equalize_channels: String with channels to equalize ('L', 'Lab', etc.)
            
        Returns:
            equalized_frame: BGR image
        """
        # Convert to LAB
        source_lab = cv2.cvtColor(source_frame, cv2.COLOR_BGR2LAB)
        result_lab = source_lab.copy()
        
        # Equalize specified channels
        if 'L' in equalize_channels:
            result_lab[:, :, 0] = cv2.equalizeHist(source_lab[:, :, 0])
        if 'a' in equalize_channels and len(equalize_channels) > 1:
            result_lab[:, :, 1] = cv2.equalizeHist(source_lab[:, :, 1])
        if 'b' in equalize_channels and len(equalize_channels) > 1:
            result_lab[:, :, 2] = cv2.equalizeHist(source_lab[:, :, 2])
        
        # Convert back
        result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result_bgr


class GradientPreservingTransfer:
    """
    Advanced transfer that preserves image gradients/edges
    Based on gradient domain processing
    """
    
    def __init__(self):
        pass
    
    def compute_gradients(self, image):
        """Compute image gradients using Sobel"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        return grad_x, grad_y
    
    def gradient_preserving_transfer(self, source_frame, target_stats, 
                                    color_transfer_func, edge_weight=0.3):
        """
        Apply color transfer while preserving edges
        
        Args:
            source_frame: Source BGR image
            target_stats: Target statistics
            color_transfer_func: Function to apply color transfer
            edge_weight: Weight for edge preservation (0-1)
            
        Returns:
            result_frame: BGR image
        """
        # Detect edges in original
        gray = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(float) / 255.0
        
        # Dilate edges slightly
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Apply color transfer
        transferred = color_transfer_func(source_frame, target_stats)
        
        # Blend: preserve original near edges
        edges_3ch = np.stack([edges] * 3, axis=2)
        result = (source_frame.astype(float) * edges_3ch * edge_weight + 
                 transferred.astype(float) * (1 - edges_3ch * edge_weight))
        
        return result.astype(np.uint8)


def test_color_transfer():
    """Test color transfer on a sample frame"""
    import matplotlib.pyplot as plt
    
    # Load a sample frame
    sample_frame_path = "Frames/Avatar/shot_000.jpg"
    if not os.path.exists(sample_frame_path):
        print(f"Sample frame not found: {sample_frame_path}")
        return
    
    source_frame = cv2.imread(sample_frame_path)
    
    # Load director statistics
    stats_path = "Palettes/Avatar_stats.json"
    with open(stats_path, 'r') as f:
        target_stats = json.load(f)
    
    print("Testing Color Transfer Methods...")
    print(f"Source frame: {sample_frame_path}")
    print(f"Target stats: {stats_path}")
    
    # Initialize color transfer
    ct = ColorTransfer(target_stats)
    ht = HistogramTransfer()
    gpt = GradientPreservingTransfer()
    
    # Apply different transfer methods
    results = {}
    
    print("\n1. Standard LAB transfer...")
    results['standard'] = ct.transfer_lab_statistics(source_frame, strength=0.8)
    
    print("2. Chrominance-only transfer...")
    results['chroma_only'] = ct.transfer_selective_channels(
        source_frame, transfer_L=False, transfer_a=True, transfer_b=True, strength=0.8
    )
    
    print("3. Luminance-preserving transfer...")
    results['lum_preserve'] = ct.transfer_with_luminance_preservation(
        source_frame, strength=0.8, luminance_weight=0.7
    )
    
    print("4. Adaptive transfer...")
    results['adaptive'] = ct.adaptive_transfer(
        source_frame, strength=0.8, preserve_extremes=True
    )
    
    print("5. Histogram matching...")
    results['histogram'] = ht.match_histogram_3d(source_frame, target_stats)
    
    print("6. Gradient-preserving transfer...")
    results['gradient'] = gpt.gradient_preserving_transfer(
        source_frame, target_stats, 
        lambda f, s: ct.transfer_lab_statistics(f, s, 0.8),
        edge_weight=0.4
    )
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')
    
    # Results
    titles = ['Standard LAB', 'Chroma Only', 'Luminance Preserve', 
              'Adaptive', 'Histogram Match', 'Gradient Preserve']
    
    for i, (key, title) in enumerate(zip(results.keys(), titles), 1):
        axes[i].imshow(cv2.cvtColor(results[key], cv2.COLOR_BGR2RGB))
        axes[i].set_title(title, fontweight='bold')
        axes[i].axis('off')
    
    # Hide last unused subplot
    axes[7].axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = "Palettes/color_transfer_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison saved to: {output_path}")
    
    # Save individual results
    output_dir = "Palettes/transfer_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, result in results.items():
        out_path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(out_path, result)
    
    print(f"✓ Individual results saved to: {output_dir}/")
    
    plt.close()
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("COLOR TRANSFER MODULE - Testing")
    print("="*60)
    
    test_color_transfer()
    
    print("\n" + "="*60)
    print("COLOR TRANSFER TEST COMPLETE!")
    print("="*60)
