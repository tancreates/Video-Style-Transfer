"""
Geometric Color Transfer Methods
Advanced techniques using Optimal Transport and Distribution Matching

Implements:
1. Linear Monge-Kantorovitch (MKL) - Covariance-based transfer
2. Iterative Distribution Transfer (IDT) - Sliced Optimal Transport
"""

import cv2
import numpy as np
from scipy.linalg import sqrtm
from tqdm import tqdm


class GeometricColorTransfer:
    """
    Advanced color transfer using geometric and probabilistic methods
    """
    
    def __init__(self, target_stats=None):
        """
        Initialize with target statistics
        
        Args:
            target_stats: Dictionary with mean and covariance from director's frames
        """
        self.target_stats = target_stats
    
    def bgr_to_lab(self, bgr_image):
        """
        Convert BGR to Ruderman's lab color space
        Same as other modules for consistency
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
    
    def lab_to_bgr(self, lab_image):
        """
        Convert lab back to BGR
        """
        # lab to LMS
        lab_matrix = np.array([
            [1/np.sqrt(3), 0, 0],
            [0, 1/np.sqrt(6), 0],
            [0, 0, 1/np.sqrt(2)]
        ]) @ np.array([
            [1, 1, 1],
            [1, 1, -2],
            [1, -1, 0]
        ])
        
        inv_lab_matrix = np.linalg.inv(lab_matrix)
        
        h, w, c = lab_image.shape
        lab_reshaped = lab_image.reshape(-1, 3).T
        lms = inv_lab_matrix @ lab_reshaped
        lms = lms.T.reshape(h, w, 3)
        
        # Inverse log
        lms = 10 ** lms
        
        # LMS to RGB
        lms_matrix = np.array([
            [0.3811, 0.5783, 0.0402],
            [0.1967, 0.7244, 0.0782],
            [0.0241, 0.1288, 0.8444]
        ])
        inv_lms_matrix = np.linalg.inv(lms_matrix)
        
        lms_reshaped = lms.reshape(-1, 3).T
        rgb = inv_lms_matrix @ lms_reshaped
        rgb = rgb.T.reshape(h, w, 3)
        
        # Clip and denormalize
        rgb = np.clip(rgb, 0, 1) * 255
        bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        
        return bgr
    
    def monge_kantorovitch_transfer(self, source_frame, target_stats, strength=1.0):
        """
        Linear Monge-Kantorovitch (MKL) color transfer
        
        Uses covariance matching instead of simple std matching.
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
        mu_target = np.array(target_stats['mean'])  # (3,)
        cov_target = np.array(target_stats['covariance'])  # (3, 3)
        
        # Add small regularization to avoid singular matrices
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
        # T = Sigma_u^(-1/2) * middle^(1/2) * Sigma_u^(-1/2)
        T = cov_source_inv_sqrt @ middle_sqrt @ cov_source_inv_sqrt
        
        # Apply transformation
        # Center source
        pixels_centered = pixels - mu_source
        
        # Apply rotation and scaling
        pixels_transformed = pixels_centered @ T.T
        
        # Add target mean
        pixels_transferred = pixels_transformed + mu_target
        
        # Blend with original based on strength
        pixels_final = strength * pixels_transferred + (1 - strength) * pixels
        
        # Reshape back to image
        result_lab = pixels_final.reshape(h, w, 3)
        
        # Convert back to BGR
        result_bgr = self.lab_to_bgr(result_lab)
        
        return result_bgr
    
    def iterative_distribution_transfer(self, source_frame, target_pixels, 
                                       iterations=20, strength=1.0):
        """
        Iterative Distribution Transfer (IDT) using Sliced Optimal Transport
        
        Matches complex color distributions by iteratively projecting onto
        random 1D axes and matching cumulative distributions.
        
        Does NOT assume Gaussian distributions - can match arbitrary shapes
        (e.g., The Matrix's green spike + black spike).
        
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
            # Step 1: Generate random normalized projection direction
            # Random vector on unit sphere
            direction = np.random.randn(3)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            
            # Step 2: Project both source and target onto this direction
            # This gives us 1D distributions
            proj_source = pixels @ direction  # (N,)
            proj_target = target_pixels @ direction  # (M,)
            
            # Step 3: Sort to get CDFs
            # Sorting creates the cumulative distribution
            sorted_idx_source = np.argsort(proj_source)
            sorted_idx_target = np.argsort(proj_target)
            
            # Step 4: Match distributions
            # Map source percentiles to target percentiles
            n_source = len(proj_source)
            n_target = len(proj_target)
            
            # Create mapping from source to target
            # Use linear interpolation to handle different sizes
            source_percentiles = np.linspace(0, 1, n_source)
            target_percentiles = np.linspace(0, 1, n_target)
            
            # Inverse CDF mapping
            matched_values = np.interp(
                source_percentiles,
                target_percentiles,
                proj_target[sorted_idx_target]
            )
            
            # Step 5: Compute displacement in 1D
            displacement_1d = np.zeros(n_source)
            displacement_1d[sorted_idx_source] = matched_values - proj_source[sorted_idx_source]
            
            # Step 6: Move pixels in 3D along the projection direction
            # displacement_3d = displacement_1d * direction (broadcast)
            pixels = pixels + displacement_1d[:, np.newaxis] * direction[np.newaxis, :]
        
        # Blend with original based on strength
        pixels_final = strength * pixels + (1 - strength) * original_pixels
        
        # Reshape back to image
        result_lab = pixels_final.reshape(h, w, 3)
        
        # Convert back to BGR
        result_bgr = self.lab_to_bgr(result_lab)
        
        return result_bgr
    
    def compute_covariance_stats(self, frames_dir):
        """
        Compute mean and covariance matrix from director frames
        Needed for MKL method
        
        Args:
            frames_dir: Directory with extracted frames
        
        Returns:
            Dictionary with 'mean' and 'covariance'
        """
        from glob import glob
        
        frame_paths = sorted(glob(f"{frames_dir}/*.jpg"))
        
        all_pixels = []
        
        print(f"Computing covariance statistics from {len(frame_paths)} frames...")
        
        for frame_path in tqdm(frame_paths):
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Convert to lab
            lab = self.bgr_to_lab(frame)
            
            # Flatten and sample (memory efficient)
            pixels = lab.reshape(-1, 3)
            
            # Sample 1000 pixels per frame
            if len(pixels) > 1000:
                indices = np.random.choice(len(pixels), 1000, replace=False)
                pixels = pixels[indices]
            
            all_pixels.append(pixels)
        
        # Concatenate all pixels
        all_pixels = np.vstack(all_pixels)
        
        # Compute statistics
        mean = np.mean(all_pixels, axis=0)
        cov = np.cov(all_pixels.T)
        
        stats = {
            'mean': mean.tolist(),
            'covariance': cov.tolist()
        }
        
        return stats, all_pixels


def test_geometric_methods():
    """
    Test the geometric color transfer methods
    """
    import matplotlib.pyplot as plt
    from glob import glob
    import os
    
    print("="*70)
    print("TESTING GEOMETRIC COLOR TRANSFER METHODS")
    print("="*70)
    
    # Setup
    director_name = "Avatar"
    frames_dir = f"Frames/{director_name}"
    
    # Find a sample frame
    sample_frames = glob(f"{frames_dir}/shot_*.jpg")
    if not sample_frames:
        print(f"Error: No frames found in {frames_dir}")
        return
    
    source_frame = cv2.imread(sample_frames[0])
    print(f"\nLoaded test frame: {sample_frames[0]}")
    print(f"Shape: {source_frame.shape}")
    
    # Initialize
    geo_transfer = GeometricColorTransfer()
    
    # Compute statistics
    print("\nComputing target statistics...")
    target_stats, target_pixels = geo_transfer.compute_covariance_stats(frames_dir)
    
    print(f"\nTarget mean: {np.array(target_stats['mean'])}")
    print(f"Target covariance shape: {np.array(target_stats['covariance']).shape}")
    print(f"Target pixels: {target_pixels.shape}")
    
    # Test MKL method
    print("\n1. Testing Monge-Kantorovitch (MKL)...")
    result_mkl = geo_transfer.monge_kantorovitch_transfer(
        source_frame, target_stats, strength=1.0
    )
    
    # Test IDT method
    print("\n2. Testing Iterative Distribution Transfer (IDT)...")
    result_idt = geo_transfer.iterative_distribution_transfer(
        source_frame, target_pixels, iterations=20, strength=1.0
    )
    
    # Create comparison
    print("\n3. Creating comparison visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    axes[0].imshow(cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # MKL result
    axes[1].imshow(cv2.cvtColor(result_mkl, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Monge-Kantorovitch (MKL)\nCovariance Matching", 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # IDT result
    axes[2].imshow(cv2.cvtColor(result_idt, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Iterative Distribution Transfer (IDT)\nSliced Optimal Transport", 
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_dir = "Comparisons/geometric_methods"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/{director_name}_geometric_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to: {output_path}")
    
    # Save individual results
    cv2.imwrite(f"{output_dir}/{director_name}_mkl.jpg", result_mkl)
    cv2.imwrite(f"{output_dir}/{director_name}_idt.jpg", result_idt)
    print(f"Saved individual results to: {output_dir}/")
    
    plt.close()
    
    print("\n" + "="*70)
    print("GEOMETRIC METHODS TEST COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    test_geometric_methods()
