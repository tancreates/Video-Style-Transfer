"""
Enhanced Color Transfer with Additional Image Processing Techniques
Implements multiple DIP techniques for better color style transfer
"""

import cv2
import numpy as np
from color_transfer import ColorTransfer
from scipy.linalg import sqrtm
import json


class EnhancedColorTransfer(ColorTransfer):
    """
    Extended color transfer with additional image processing techniques
    """
    
    def __init__(self, target_stats=None):
        super().__init__(target_stats)
    
    def histogram_specification(self, source_frame, target_frame):
        """
        Histogram specification - match source histogram to target histogram
        
        Args:
            source_frame: Source BGR image
            target_frame: Target BGR image (or representative frame)
            
        Returns:
            matched: Histogram-matched BGR image
        """
        # Convert to lab
        source_lab = self.bgr_to_lab(source_frame)
        target_lab = self.bgr_to_lab(target_frame)
        
        result = source_lab.copy()
        
        # Match histogram for each channel
        for i in range(3):
            # Flatten channels
            source_channel = source_lab[:, :, i].flatten()
            target_channel = target_lab[:, :, i].flatten()
            
            # Compute CDFs
            source_values, source_counts = np.unique(source_channel, return_counts=True)
            target_values, target_counts = np.unique(target_channel, return_counts=True)
            
            source_quantiles = np.cumsum(source_counts).astype(np.float64)
            source_quantiles /= source_quantiles[-1]
            
            target_quantiles = np.cumsum(target_counts).astype(np.float64)
            target_quantiles /= target_quantiles[-1]
            
            # Interpolate mapping
            interp_values = np.interp(source_quantiles, target_quantiles, target_values)
            
            # Map source values to target values
            result[:, :, i] = np.interp(source_lab[:, :, i].flatten(), 
                                         source_values, interp_values).reshape(source_lab[:, :, i].shape)
        
        # Convert back to BGR
        matched = self.lab_to_bgr(result)
        
        return matched
    
    def multiscale_transfer(self, source_frame, target_stats, scales=[1.0, 0.5, 0.25]):
        """
        Multi-scale color transfer for better detail preservation
        
        Args:
            source_frame: Source BGR image
            target_stats: Target statistics
            scales: List of scales to process
            
        Returns:
            result: Multi-scale transferred image
        """
        h, w = source_frame.shape[:2]
        
        # Process at multiple scales
        pyramid = []
        for scale in scales:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled = cv2.resize(source_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Transfer at this scale
            transferred = self.transfer_lab_statistics(scaled, target_stats, strength=1.0)
            
            # Resize back
            upscaled = cv2.resize(transferred, (w, h), interpolation=cv2.INTER_CUBIC)
            pyramid.append(upscaled)
        
        # Combine scales
        weights = [0.6, 0.3, 0.1]  # More weight to higher resolution
        result = np.zeros_like(pyramid[0], dtype=np.float32)
        
        for img, weight in zip(pyramid, weights):
            result += img.astype(np.float32) * weight
        
        return result.astype(np.uint8)
    
    def guided_filter_transfer(self, source_frame, target_stats, radius=8, eps=0.01):
        """
        Edge-preserving color transfer using guided filter
        
        Args:
            source_frame: Source BGR image
            target_stats: Target statistics
            radius: Guided filter radius
            eps: Regularization parameter
            
        Returns:
            result: Edge-preserved transferred image
        """
        # Basic transfer
        transferred = self.transfer_lab_statistics(source_frame, target_stats, strength=1.0)
        
        # Convert to grayscale for guidance
        guide = cv2.cvtColor(source_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Apply guided filter to each channel
        result = np.zeros_like(transferred, dtype=np.float32)
        transferred_float = transferred.astype(np.float32) / 255.0
        
        for i in range(3):
            result[:, :, i] = cv2.ximgproc.guidedFilter(
                guide, transferred_float[:, :, i], radius, eps
            )
        
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        
        return result
    
    def bilateral_color_transfer(self, source_frame, target_stats, d=5, sigma_color=30, sigma_space=30):
        """
        Color transfer with bilateral filtering for edge preservation
        
        Args:
            source_frame: Source BGR image
            target_stats: Target statistics
            d: Diameter of pixel neighborhood (default: 5, smaller = less blur)
            sigma_color: Color space sigma (default: 30, lower = sharper)
            sigma_space: Coordinate space sigma (default: 30, lower = sharper)
            
        Returns:
            result: Bilateral-filtered transferred image
        """
        # Apply basic transfer
        transferred = self.transfer_lab_statistics(source_frame, target_stats, strength=1.0)
        
        # Apply lighter bilateral filter to reduce noise while preserving edges
        result = cv2.bilateralFilter(transferred, d, sigma_color, sigma_space)
        
        return result
    
    def adaptive_manifold_transfer(self, source_frame, target_stats, sigma_s=60, sigma_r=0.4):
        """
        Color transfer with adaptive manifold filtering
        
        Args:
            source_frame: Source BGR image
            target_stats: Target statistics
            sigma_s: Spatial sigma
            sigma_r: Range sigma
            
        Returns:
            result: Adaptively filtered transferred image
        """
        # Basic transfer
        transferred = self.transfer_lab_statistics(source_frame, target_stats, strength=1.0)
        
        # Apply adaptive manifold filter
        joint = source_frame.astype(np.float32) / 255.0
        src = transferred.astype(np.float32) / 255.0
        
        result = cv2.ximgproc.amFilter(joint, src, sigma_s, sigma_r)
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        
        return result
    
    def detail_preserving_transfer(self, source_frame, target_stats, lambda_param=10.0):
        """
        Detail-preserving color transfer
        
        Args:
            source_frame: Source BGR image
            target_stats: Target statistics
            lambda_param: Edge preservation parameter
            
        Returns:
            result: Detail-preserved transferred image
        """
        # Basic transfer
        transferred = self.transfer_lab_statistics(source_frame, target_stats, strength=1.0)
        
        # Extract base and detail layers from source
        source_base = cv2.bilateralFilter(source_frame, 9, 75, 75)
        source_detail = cv2.subtract(source_frame, source_base)
        
        # Apply detail back to transferred
        result = cv2.add(transferred.astype(np.int16), source_detail.astype(np.int16))
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def chrominance_only_transfer(self, source_frame, target_stats, strength=1.0):
        """
        Transfer only chrominance (a and b), preserve lightness
        
        Args:
            source_frame: Source BGR image
            target_stats: Target statistics
            strength: Transfer strength
            
        Returns:
            result: Chrominance-transferred image
        """
        # Convert to lab
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
            source_mean[0],  # Keep original lightness mean
            target_stats['mean_alpha'],
            target_stats['mean_beta']
        ])
        
        target_std = np.array([
            source_std[0],  # Keep original lightness std
            target_stats['std_alpha'],
            target_stats['std_beta']
        ])
        
        # Avoid division by zero
        source_std = np.where(source_std < 1e-6, 1.0, source_std)
        
        # Apply transfer
        result = np.zeros_like(source_lab)
        for i in range(3):
            centered = source_lab[:, :, i] - source_mean[i]
            scaled = centered * (target_std[i] / source_std[i])
            transferred = scaled + target_mean[i]
            result[:, :, i] = source_lab[:, :, i] * (1 - strength) + transferred * strength
        
        # Convert back
        result_bgr = self.lab_to_bgr(result)
        
        return result_bgr
    
    def progressive_transfer(self, source_frame, target_stats, iterations=3, strength_schedule=None):
        """
        Progressive iterative color transfer for smoother results
        
        Args:
            source_frame: Source BGR image
            target_stats: Target statistics
            iterations: Number of iterations
            strength_schedule: List of strengths for each iteration
            
        Returns:
            result: Progressively transferred image
        """
        if strength_schedule is None:
            strength_schedule = [0.3, 0.5, 1.0][:iterations]
        
        result = source_frame.copy()
        
        for i, strength in enumerate(strength_schedule):
            result = self.transfer_lab_statistics(result, target_stats, strength=strength)
        
        return result
    
    def monge_kantorovitch_transfer(self, source_frame, target_stats, strength=1.0):
        """
        Linear Monge-Kantorovitch (MKL) color transfer
        
        Advanced method using covariance matching instead of simple std matching.
        Accounts for correlated color channels (e.g., warm colors move together).
        
        Mathematical formula:
        t(u) = Sigma_u^(-1/2) * (Sigma_u^(1/2) * Sigma_v * Sigma_u^(1/2))^(1/2) * Sigma_u^(-1/2) * (u - mu_u) + mu_v
        
        Args:
            source_frame: Input BGR image
            target_stats: Dictionary with 'mean' and 'covariance' arrays
            strength: Transfer strength (0.0-1.0)
        
        Returns:
            Transferred BGR image
        """
        # Convert to lab
        source_lab = self.bgr_to_lab(source_frame)
        h, w = source_lab.shape[:2]
        
        # Flatten to (N, 3) array
        pixels = source_lab.reshape(-1, 3)
        
        # Compute source statistics
        mu_source = np.mean(pixels, axis=0)  # (3,)
        pixels_centered = pixels - mu_source
        cov_source = np.cov(pixels_centered.T)  # (3, 3)
        
        # Get target statistics
        if 'mean' in target_stats and 'covariance' in target_stats:
            # New format with covariance
            mu_target = np.array(target_stats['mean'])
            cov_target = np.array(target_stats['covariance'])
        else:
            # Fallback: construct covariance from std (diagonal matrix)
            mu_target = np.array([
                target_stats['mean_l'],
                target_stats['mean_alpha'],
                target_stats['mean_beta']
            ])
            std_target = np.array([
                target_stats['std_l'],
                target_stats['std_alpha'],
                target_stats['std_beta']
            ])
            cov_target = np.diag(std_target ** 2)
        
        # Add regularization to avoid singular matrices
        eps = 1e-6
        cov_source = cov_source + eps * np.eye(3)
        cov_target = cov_target + eps * np.eye(3)
        
        # Compute transformation matrix using MKL formula
        # Step 1: Sigma_u^(1/2)
        cov_source_sqrt = sqrtm(cov_source).real
        
        # Step 2: Sigma_u^(-1/2)
        cov_source_inv_sqrt = sqrtm(np.linalg.inv(cov_source)).real
        
        # Step 3: (Sigma_u^(1/2) * Sigma_v * Sigma_u^(1/2))^(1/2)
        middle = cov_source_sqrt @ cov_target @ cov_source_sqrt
        middle_sqrt = sqrtm(middle).real
        
        # Step 4: Final transformation matrix
        T = cov_source_inv_sqrt @ middle_sqrt @ cov_source_inv_sqrt
        
        # Apply transformation
        pixels_centered = pixels - mu_source
        pixels_transformed = pixels_centered @ T.T
        pixels_transferred = pixels_transformed + mu_target
        
        # Blend with original based on strength
        pixels_final = strength * pixels_transferred + (1 - strength) * pixels
        
        # Reshape back to image
        result_lab = pixels_final.reshape(h, w, 3)
        
        # Convert back to BGR
        result_bgr = self.lab_to_bgr(result_lab)
        
        return result_bgr
    def protect_highlights_lab(self, bgr_image, threshold=220):
        """
        Desaturates highlights to ensure whites remain white.
        Uses CIE LAB space to isolate Luminance from Color.
        """
        # 1. Convert to CIE LAB (Standard OpenCV 8-bit)
        # Note: We use standard LAB here instead of Ruderman because 
        # thresholding (220) works predictably in standard 0-255 range.
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # 2. Create Desaturation Factor
        # If L < threshold, factor = 1.0 (Keep Color)
        # If L > threshold, factor fades to 0.0 (Remove Color)
        # We use a soft curve for smooth transition
        desat_factor = np.clip((255 - l_channel) / (255 - threshold + 1e-6), 0, 1)
        
        # Make the transition non-linear (smoother roll-off)
        desat_factor = desat_factor ** 2
        
        # 3. Apply to Color Channels (a and b)
        # In OpenCV 8-bit LAB, the neutral (grey) point is 128
        # We shift 'a' and 'b' towards 128 based on the factor
        a_channel = (a_channel - 128) * desat_factor + 128
        b_channel = (b_channel - 128) * desat_factor + 128
        
        # 4. Merge and Convert back to BGR
        lab_protected = cv2.merge([l_channel, a_channel, b_channel])
        result_bgr = cv2.cvtColor(lab_protected.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result_bgr
    
    def create_skin_mask_ycbcr(self, bgr_image):
        """
        Generate skin mask using YCbCr color space thresholding (DIP Method).
        Returns a soft alpha mask (0.0 - 1.0) where 1.0 is skin.
        """
        # 1. Convert to YCbCr
        ycbcr = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)
        
        # 2. Define skin color bounds (Empirical values)
        # Cb: 77-127, Cr: 133-173
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        # 3. Create binary mask
        mask = cv2.inRange(ycbcr, lower, upper)
        
        # 4. Clean up noise using Morphology (Open then Dilate)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # 5. Blur to create soft transitions (Alpha matte)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # 6. Normalize to 0-1 float and expand to 3 channels
        mask_float = mask.astype(np.float32) / 255.0
        return np.stack([mask_float] * 3, axis=2)

    def iterative_distribution_transfer(self, source_frame, target_pixels, 
                                       iterations=20, strength=1.0):
        """
        Iterative Distribution Transfer (IDT) using Sliced Optimal Transport
        
        Matches complex color distributions without Gaussian assumption.
        Can match arbitrary shapes (e.g., The Matrix's green spike + black spike).
        
        Algorithm:
        1. Pick random 3D direction (axis)
        2. Project both images onto this axis (3D -> 1D)
        3. Match the 1D distributions using CDF mapping
        4. Move pixels in 3D by the 1D displacement
        5. Repeat with different random directions
        
        Args:
            source_frame: Input BGR image
            target_pixels: Target image pixels (N, 3) array in lab space
            iterations: Number of random projections (default: 20)
            strength: Transfer strength (0.0-1.0)
        
        Returns:
            Transferred BGR image
        """
        # Convert to lab
        source_lab = self.bgr_to_lab(source_frame)
        h, w = source_lab.shape[:2]
        
        # Flatten to (N, 3)
        pixels = source_lab.reshape(-1, 3).copy()
        original_pixels = pixels.copy()
        
        # Iterative sliced transport
        for iteration in range(iterations):
            # Generate random normalized projection direction
            direction = np.random.randn(3)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # Project both source and target onto this direction
            proj_source = pixels @ direction
            proj_target = target_pixels @ direction
            
            # Sort to get CDFs
            sorted_idx_source = np.argsort(proj_source)
            sorted_idx_target = np.argsort(proj_target)
            
            # Match distributions
            n_source = len(proj_source)
            n_target = len(proj_target)
            
            # Create mapping from source to target
            source_percentiles = np.linspace(0, 1, n_source)
            target_percentiles = np.linspace(0, 1, n_target)
            
            # Inverse CDF mapping
            matched_values = np.interp(
                source_percentiles,
                target_percentiles,
                proj_target[sorted_idx_target]
            )
            
            # Compute displacement in 1D
            displacement_1d = np.zeros(n_source)
            displacement_1d[sorted_idx_source] = matched_values - proj_source[sorted_idx_source]
            
            # Move pixels in 3D along projection direction
            pixels = pixels + displacement_1d[:, np.newaxis] * direction[np.newaxis, :]
        
        # Blend with original based on strength
        pixels_final = strength * pixels + (1 - strength) * original_pixels
        
        # Reshape back to image
        result_lab = pixels_final.reshape(h, w, 3)
        
        # Convert back to BGR
        result_bgr = self.lab_to_bgr(result_lab)
        
        return result_bgr


def test_enhanced_methods():
    """Test enhanced color transfer methods"""
    import matplotlib.pyplot as plt
    import os
    
    print("="*70)
    print("TESTING ENHANCED COLOR TRANSFER METHODS")
    print("="*70)
    
    # Load source
    source_path = "Frames/Avatar/shot_050.jpg"
    source = cv2.imread(source_path)
    
    # Load stats
    with open("Palettes/Avatar_stats.json", 'r') as f:
        stats = json.load(f)
    
    # Initialize
    ct = EnhancedColorTransfer(stats)
    
    print("\nApplying enhanced methods...")
    
    results = {}
    
    # Try methods that don't require additional dependencies
    print("  1. Basic Reinhard transfer...")
    results['Basic'] = ct.transfer_lab_statistics(source, stats, strength=1.0)
    
    print("  2. Multi-scale transfer...")
    results['MultiScale'] = ct.multiscale_transfer(source, stats)
    
    print("  3. Bilateral transfer...")
    results['Bilateral'] = ct.bilateral_color_transfer(source, stats)
    
    print("  4. Detail-preserving transfer...")
    results['DetailPreserve'] = ct.detail_preserving_transfer(source, stats)
    
    print("  5. Chrominance-only transfer...")
    results['ChromaOnly'] = ct.chrominance_only_transfer(source, stats, strength=1.0)
    
    print("  6. Progressive transfer...")
    results['Progressive'] = ct.progressive_transfer(source, stats)
    
    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    axes[0].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')
    
    for i, (name, result) in enumerate(results.items(), 1):
        axes[i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[i].set_title(name, fontweight='bold')
        axes[i].axis('off')
    
    axes[7].axis('off')
    
    plt.tight_layout()
    
    output_path = "Palettes/enhanced_methods_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to: {output_path}")
    
    # Save individual results
    output_dir = "Palettes/enhanced_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, result in results.items():
        cv2.imwrite(os.path.join(output_dir, f"{name}.jpg"), result)
    
    print(f"Saved individual results to: {output_dir}/")
    
    plt.close()
    
    print("\n" + "="*70)
    print("ENHANCED METHODS TEST COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    test_enhanced_methods()
