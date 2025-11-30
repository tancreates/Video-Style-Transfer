"""
Evaluation Metrics for Color Transfer Quality Assessment

Based on Digital Image Processing best practices, this module implements
three categories of metrics:
1. Color Fidelity - How well did we match the target style?
2. Structure Preservation - Did we keep image details?
3. Temporal Stability - Did we avoid flickering in videos?

As noted in the literature: "there is no accepted objective benchmark" for
color transfer, so these metrics provide quantitative guidance for artistic assessment.
"""

import numpy as np
import cv2
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json


class ColorTransferEvaluator:
    """Comprehensive evaluation of color transfer quality"""
    
    def __init__(self, color_space='lab'):
        """
        Initialize evaluator
        
        Args:
            color_space: 'lab' for perceptual LAB space, 'rgb' for RGB
        """
        self.color_space = color_space
        self.history = []
        
    def evaluate_frame(self, 
                      source: np.ndarray,
                      output: np.ndarray, 
                      target: np.ndarray,
                      verbose: bool = True) -> Dict[str, float]:
        """
        Comprehensive evaluation of a single frame
        
        Args:
            source: Original input image (BGR)
            output: Color-transferred output (BGR)
            target: Target style image (BGR)
            verbose: Print results
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # 1. Color Fidelity Metrics
        color_metrics = self.evaluate_color_fidelity(output, target)
        metrics.update(color_metrics)
        
        # 2. Structure Preservation Metrics
        structure_metrics = self.evaluate_structure_preservation(source, output)
        metrics.update(structure_metrics)
        
        if verbose:
            self.print_metrics(metrics)
            
        self.history.append(metrics)
        return metrics
    
    # ========================================================================
    # 1. COLOR FIDELITY METRICS
    # ========================================================================
    
    def evaluate_color_fidelity(self, output: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """
        Measure how well output matches target color distribution
        
        Metrics:
        - EMD (Wasserstein Distance): Optimal Transport distance
        - KL Divergence: Information-theoretic difference
        - Histogram Intersection: Overlap between distributions
        - Mean Color Distance: Average color difference
        """
        metrics = {}
        
        # Convert to LAB for perceptual evaluation
        output_lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # A. Earth Mover's Distance (Wasserstein Distance)
        emd_l, emd_a, emd_b = self._compute_wasserstein_distance(output_lab, target_lab)
        metrics['EMD_L'] = emd_l
        metrics['EMD_A'] = emd_a
        metrics['EMD_B'] = emd_b
        metrics['EMD_Average'] = (emd_l + emd_a + emd_b) / 3.0
        
        # B. Kullback-Leibler Divergence
        kl_l, kl_a, kl_b = self._compute_kl_divergence(output_lab, target_lab)
        metrics['KL_L'] = kl_l
        metrics['KL_A'] = kl_a
        metrics['KL_B'] = kl_b
        metrics['KL_Average'] = (kl_l + kl_a + kl_b) / 3.0
        
        # C. Histogram Intersection (0-1, higher is better)
        hist_int = self._compute_histogram_intersection(output_lab, target_lab)
        metrics['Histogram_Intersection'] = hist_int
        
        # D. Mean Color Distance in LAB space
        mean_dist = self._compute_mean_color_distance(output_lab, target_lab)
        metrics['Mean_Color_Distance'] = mean_dist
        
        # E. Statistical Moments Difference (mean, std)
        for i, channel in enumerate(['L', 'A', 'B']):
            out_mean = np.mean(output_lab[:, :, i])
            tgt_mean = np.mean(target_lab[:, :, i])
            out_std = np.std(output_lab[:, :, i])
            tgt_std = np.std(target_lab[:, :, i])
            
            metrics[f'Mean_Diff_{channel}'] = abs(out_mean - tgt_mean)
            metrics[f'Std_Diff_{channel}'] = abs(out_std - tgt_std)
        
        return metrics
    
    def _compute_wasserstein_distance(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute Earth Mover's Distance (Wasserstein) for each channel
        
        This is the mathematically robust way to measure distribution difference
        as emphasized in the Optimal Transport literature.
        
        Lower values = better match
        """
        distances = []
        
        for channel in range(3):
            # Flatten channel to 1D distribution
            dist1 = img1[:, :, channel].flatten()
            dist2 = img2[:, :, channel].flatten()
            
            # Compute Wasserstein distance (1D Earth Mover's Distance)
            emd = wasserstein_distance(dist1, dist2)
            distances.append(emd)
        
        return tuple(distances)
    
    def _compute_kl_divergence(self, img1: np.ndarray, img2: np.ndarray, bins: int = 256) -> Tuple[float, float, float]:
        """
        Compute Kullback-Leibler Divergence for each channel
        
        Measures information loss when approximating target with output.
        Lower values = better match
        
        Note: KL divergence is not symmetric and can be infinite if distributions
        don't overlap. We add small epsilon for numerical stability.
        """
        divergences = []
        epsilon = 1e-10
        
        for channel in range(3):
            # Compute normalized histograms (probability distributions)
            hist1, _ = np.histogram(img1[:, :, channel], bins=bins, range=(0, 256), density=True)
            hist2, _ = np.histogram(img2[:, :, channel], bins=bins, range=(0, 256), density=True)
            
            # Add epsilon to avoid log(0)
            hist1 = hist1 + epsilon
            hist2 = hist2 + epsilon
            
            # Normalize to sum to 1
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            # Compute KL divergence: sum(p * log(p/q))
            kl = entropy(hist1, hist2)
            
            # Handle potential inf/nan
            if np.isnan(kl) or np.isinf(kl):
                kl = 100.0  # Large penalty for poor match
            
            divergences.append(kl)
        
        return tuple(divergences)
    
    def _compute_histogram_intersection(self, img1: np.ndarray, img2: np.ndarray, bins: int = 256) -> float:
        """
        Compute histogram intersection (overlap metric)
        
        Returns value in [0, 1] where 1 = perfect match
        Higher values = better match
        """
        intersections = []
        
        for channel in range(3):
            hist1, _ = np.histogram(img1[:, :, channel], bins=bins, range=(0, 256), density=True)
            hist2, _ = np.histogram(img2[:, :, channel], bins=bins, range=(0, 256), density=True)
            
            # Normalize
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            # Intersection = sum of minimum values
            intersection = np.sum(np.minimum(hist1, hist2))
            intersections.append(intersection)
        
        return np.mean(intersections)
    
    def _compute_mean_color_distance(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute average Euclidean distance between pixels in LAB space
        
        Lower values = better match
        """
        # Compute pixel-wise Euclidean distance
        diff = img1.astype(np.float32) - img2.astype(np.float32)
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        return np.mean(distances)
    
    # ========================================================================
    # 2. STRUCTURE PRESERVATION METRICS
    # ========================================================================
    
    def evaluate_structure_preservation(self, source: np.ndarray, output: np.ndarray) -> Dict[str, float]:
        """
        Measure how well output preserves source structure/details
        
        Metrics:
        - SSIM: Structural similarity (0-1, higher is better)
        - PSNR: Peak signal-to-noise ratio (dB, higher is better)
        - Edge Correlation: Preservation of edges
        - Gradient Magnitude Similarity: Detail preservation
        """
        metrics = {}
        
        # Convert to grayscale for structural comparison
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        
        # A. SSIM (Structural Similarity Index)
        # Range: [-1, 1], typically [0, 1] for images. 1 = perfect match
        ssim_value = ssim(source_gray, output_gray, data_range=255)
        metrics['SSIM'] = ssim_value
        
        # B. PSNR (Peak Signal-to-Noise Ratio)
        # Typically 20-50 dB. Higher = better. Infinite for identical images.
        try:
            psnr_value = psnr(source, output, data_range=255)
            if np.isinf(psnr_value):
                psnr_value = 100.0  # Cap at reasonable value
        except:
            psnr_value = 0.0
        metrics['PSNR'] = psnr_value
        
        # C. Edge Correlation
        edge_corr = self._compute_edge_correlation(source_gray, output_gray)
        metrics['Edge_Correlation'] = edge_corr
        
        # D. Gradient Magnitude Similarity
        grad_sim = self._compute_gradient_similarity(source_gray, output_gray)
        metrics['Gradient_Similarity'] = grad_sim
        
        return metrics
    
    def _compute_edge_correlation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute correlation between edge maps
        
        Higher values (closer to 1) = better edge preservation
        """
        # Sobel edge detection
        edges1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        edges1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        edges1 = np.sqrt(edges1_x**2 + edges1_y**2)
        
        edges2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
        edges2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
        edges2 = np.sqrt(edges2_x**2 + edges2_y**2)
        
        # Pearson correlation
        corr = np.corrcoef(edges1.flatten(), edges2.flatten())[0, 1]
        
        # Handle NaN
        if np.isnan(corr):
            corr = 0.0
        
        return corr
    
    def _compute_gradient_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute similarity of gradient magnitudes
        
        Higher values = better detail preservation
        """
        # Compute gradients
        grad1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
        grad1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
        mag1 = np.sqrt(grad1_x**2 + grad1_y**2)
        
        grad2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
        grad2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
        mag2 = np.sqrt(grad2_x**2 + grad2_y**2)
        
        # Compute similarity (1 - normalized difference)
        diff = np.abs(mag1 - mag2)
        max_mag = np.maximum(mag1, mag2) + 1e-10
        similarity = 1.0 - np.mean(diff / max_mag)
        
        return similarity
    
    # ========================================================================
    # 3. TEMPORAL STABILITY METRICS (for videos)
    # ========================================================================
    
    def evaluate_temporal_stability(self, 
                                   frames: List[np.ndarray],
                                   verbose: bool = True) -> Dict[str, float]:
        """
        Measure temporal consistency across video frames
        
        Metrics:
        - Temporal Coherence: Frame-to-frame color stability
        - Flicker Index: Brightness fluctuation measure
        - Motion-Compensated Difference: Stability accounting for motion
        """
        if len(frames) < 2:
            return {'error': 'Need at least 2 frames for temporal evaluation'}
        
        metrics = {}
        
        # A. Temporal Coherence (average inter-frame difference)
        coherence = self._compute_temporal_coherence(frames)
        metrics['Temporal_Coherence'] = coherence
        
        # B. Flicker Index (brightness fluctuation)
        flicker = self._compute_flicker_index(frames)
        metrics['Flicker_Index'] = flicker
        
        # C. Color Variance Over Time
        color_variance = self._compute_color_variance_over_time(frames)
        metrics['Color_Variance'] = color_variance
        
        if verbose:
            print("\n=== TEMPORAL STABILITY ===")
            print(f"Temporal Coherence: {coherence:.4f} (lower = more stable)")
            print(f"Flicker Index: {flicker:.4f} (lower = less flicker)")
            print(f"Color Variance: {color_variance:.4f} (lower = more stable)")
        
        return metrics
    
    def _compute_temporal_coherence(self, frames: List[np.ndarray]) -> float:
        """
        Compute average frame-to-frame difference
        
        Lower values = more temporally stable (less flickering)
        """
        differences = []
        
        for i in range(len(frames) - 1):
            frame1 = frames[i].astype(np.float32)
            frame2 = frames[i + 1].astype(np.float32)
            
            # Mean absolute difference
            diff = np.mean(np.abs(frame1 - frame2))
            differences.append(diff)
        
        return np.mean(differences)
    
    def _compute_flicker_index(self, frames: List[np.ndarray]) -> float:
        """
        Compute brightness flicker using luminance channel
        
        Lower values = less flicker
        """
        luminances = []
        
        for frame in frames:
            # Convert to LAB and extract L channel
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            mean_luminance = np.mean(lab[:, :, 0])
            luminances.append(mean_luminance)
        
        # Standard deviation of luminance over time
        flicker = np.std(luminances)
        
        return flicker
    
    def _compute_color_variance_over_time(self, frames: List[np.ndarray]) -> float:
        """
        Compute variance of mean colors across frames
        
        Lower values = more stable colors
        """
        mean_colors = []
        
        for frame in frames:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            mean_color = np.mean(lab, axis=(0, 1))
            mean_colors.append(mean_color)
        
        mean_colors = np.array(mean_colors)
        
        # Variance across time for each channel
        variance = np.mean(np.var(mean_colors, axis=0))
        
        return variance
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Pretty print metrics"""
        print("\n" + "="*60)
        print("COLOR TRANSFER EVALUATION METRICS")
        print("="*60)
        
        print("\n--- COLOR FIDELITY (Did we match the target?) ---")
        print(f"EMD Average: {metrics.get('EMD_Average', 0):.4f} (lower = better)")
        print(f"  L: {metrics.get('EMD_L', 0):.4f}, A: {metrics.get('EMD_A', 0):.4f}, B: {metrics.get('EMD_B', 0):.4f}")
        print(f"KL Divergence: {metrics.get('KL_Average', 0):.4f} (lower = better)")
        print(f"  L: {metrics.get('KL_L', 0):.4f}, A: {metrics.get('KL_A', 0):.4f}, B: {metrics.get('KL_B', 0):.4f}")
        print(f"Histogram Intersection: {metrics.get('Histogram_Intersection', 0):.4f} (higher = better, max=1.0)")
        print(f"Mean Color Distance: {metrics.get('Mean_Color_Distance', 0):.4f} (lower = better)")
        
        print("\n--- STRUCTURE PRESERVATION (Did we keep details?) ---")
        print(f"SSIM: {metrics.get('SSIM', 0):.4f} (higher = better, max=1.0)")
        print(f"PSNR: {metrics.get('PSNR', 0):.2f} dB (higher = better)")
        print(f"Edge Correlation: {metrics.get('Edge_Correlation', 0):.4f} (higher = better)")
        print(f"Gradient Similarity: {metrics.get('Gradient_Similarity', 0):.4f} (higher = better)")
        
        print("="*60 + "\n")
    
    def save_metrics(self, filepath: str):
        """Save evaluation history to JSON"""
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        serializable_history = convert_to_native(self.history)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """
        Plot key metrics for visual analysis
        """
        if not self.history:
            print("No metrics to plot")
            return
        
        # Check if single frame (image) or multiple frames (video)
        is_single_frame = len(self.history) == 1
        
        if is_single_frame:
            # For single image, create a bar chart instead of line plot
            self._plot_single_frame_metrics(save_path)
        else:
            # For video/multiple frames, create line plots
            self._plot_multi_frame_metrics(save_path)
    
    def _plot_single_frame_metrics(self, save_path: Optional[str] = None):
        """Plot metrics for single frame as bar chart"""
        metrics = self.history[0]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Color Transfer Quality Metrics (Single Image)', fontsize=16, fontweight='bold')
        
        # Plot 1: Color Fidelity Metrics
        color_metrics = {
            'EMD\nAverage': metrics.get('EMD_Average', 0),
            'KL\nDivergence': metrics.get('KL_Average', 0),
            'Histogram\nIntersection': metrics.get('Histogram_Intersection', 0),
            'Mean Color\nDistance': metrics.get('Mean_Color_Distance', 0) / 10  # Scale for visibility
        }
        
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(color_metrics)), list(color_metrics.values()), 
                        color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
        ax1.set_xticks(range(len(color_metrics)))
        ax1.set_xticklabels(list(color_metrics.keys()), fontsize=9)
        ax1.set_title('Color Fidelity Metrics', fontweight='bold')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Structure Preservation Metrics
        structure_metrics = {
            'SSIM': metrics.get('SSIM', 0),
            'PSNR\n(÷10)': metrics.get('PSNR', 0) / 10,  # Scale for visibility
            'Edge\nCorrelation': metrics.get('Edge_Correlation', 0),
            'Gradient\nSimilarity': metrics.get('Gradient_Similarity', 0)
        }
        
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(structure_metrics)), list(structure_metrics.values()),
                        color=['#9b59b6', '#1abc9c', '#34495e', '#e67e22'])
        ax2.set_xticks(range(len(structure_metrics)))
        ax2.set_xticklabels(list(structure_metrics.keys()), fontsize=9)
        ax2.set_title('Structure Preservation Metrics', fontweight='bold')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: EMD by Channel
        emd_channels = {
            'L Channel': metrics.get('EMD_L', 0),
            'A Channel': metrics.get('EMD_A', 0),
            'B Channel': metrics.get('EMD_B', 0)
        }
        
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(emd_channels)), list(emd_channels.values()),
                        color=['#3498db', '#e74c3c', '#f39c12'])
        ax3.set_xticks(range(len(emd_channels)))
        ax3.set_xticklabels(list(emd_channels.keys()))
        ax3.set_title('EMD per LAB Channel', fontweight='bold')
        ax3.set_ylabel('EMD (lower = better)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Quality Summary (Traffic Light Style)
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Quality assessment
        emd_avg = metrics.get('EMD_Average', 0)
        hist_int = metrics.get('Histogram_Intersection', 0)
        ssim_val = metrics.get('SSIM', 0)
        psnr_val = metrics.get('PSNR', 0)
        
        # Determine quality levels
        color_quality = 'Excellent' if emd_avg < 5 else 'Good' if emd_avg < 10 else 'Moderate' if emd_avg < 20 else 'Poor'
        color_color = '#2ecc71' if emd_avg < 5 else '#f39c12' if emd_avg < 10 else '#e67e22' if emd_avg < 20 else '#e74c3c'
        
        structure_quality = 'Excellent' if ssim_val > 0.9 else 'Good' if ssim_val > 0.8 else 'Moderate' if ssim_val > 0.7 else 'Poor'
        structure_color = '#2ecc71' if ssim_val > 0.9 else '#f39c12' if ssim_val > 0.8 else '#e67e22' if ssim_val > 0.7 else '#e74c3c'
        
        # Display summary
        summary_text = f"""
QUALITY ASSESSMENT

Color Fidelity: {color_quality}
  • EMD Average: {emd_avg:.2f}
  • Hist. Intersection: {hist_int:.2f}
  
Structure Preservation: {structure_quality}
  • SSIM: {ssim_val:.3f}
  • PSNR: {psnr_val:.1f} dB

Interpretation:
  • EMD < 10 = Good match
  • SSIM > 0.85 = Good details
  • Hist. Int. > 0.7 = Good overlap
"""
        
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
                fontsize=11, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Add color quality indicator
        ax4.add_patch(plt.Rectangle((0.05, 0.85), 0.1, 0.1, transform=ax4.transAxes,
                                    facecolor=color_color, edgecolor='black', linewidth=2))
        ax4.text(0.18, 0.9, 'Color Match', transform=ax4.transAxes,
                fontsize=10, verticalalignment='center', fontweight='bold')
        
        ax4.add_patch(plt.Rectangle((0.05, 0.70), 0.1, 0.1, transform=ax4.transAxes,
                                    facecolor=structure_color, edgecolor='black', linewidth=2))
        ax4.text(0.18, 0.75, 'Detail Preservation', transform=ax4.transAxes,
                fontsize=10, verticalalignment='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_multi_frame_metrics(self, save_path: Optional[str] = None):
        """Plot metrics for multiple frames as line plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Color Transfer Quality Metrics', fontsize=16, fontweight='bold')
        
        # Extract metrics over frames
        emd_avg = [m.get('EMD_Average', 0) for m in self.history]
        kl_avg = [m.get('KL_Average', 0) for m in self.history]
        ssim_vals = [m.get('SSIM', 0) for m in self.history]
        hist_int = [m.get('Histogram_Intersection', 0) for m in self.history]
        
        frames = range(len(self.history))
        
        # Plot 1: EMD
        axes[0, 0].plot(frames, emd_avg, 'b-', linewidth=2)
        axes[0, 0].set_title('Earth Mover\'s Distance (EMD)')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('EMD (lower = better)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: KL Divergence
        axes[0, 1].plot(frames, kl_avg, 'r-', linewidth=2)
        axes[0, 1].set_title('KL Divergence')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('KL (lower = better)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: SSIM
        axes[1, 0].plot(frames, ssim_vals, 'g-', linewidth=2)
        axes[1, 0].set_title('Structural Similarity (SSIM)')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('SSIM (higher = better)')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Histogram Intersection
        axes[1, 1].plot(frames, hist_int, 'm-', linewidth=2)
        axes[1, 1].set_title('Histogram Intersection')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Intersection (higher = better)')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all evaluated frames
        """
        if not self.history:
            return {}
        
        summary = {}
        
        # Average each metric across all frames
        all_keys = self.history[0].keys()
        for key in all_keys:
            values = [m.get(key, 0) for m in self.history]
            # Convert to native Python float for JSON serialization
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values))
            summary[f'{key}_min'] = float(np.min(values))
            summary[f'{key}_max'] = float(np.max(values))
        
        return summary


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Evaluate color transfer on sample images
    """
    print("Color Transfer Evaluation Metrics - Example Usage\n")
    
    # Create evaluator
    evaluator = ColorTransferEvaluator()
    
    # Example with dummy images (replace with real images)
    source = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    output = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    target = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Evaluate single frame
    metrics = evaluator.evaluate_frame(source, output, target)
    
    # For video evaluation (example with 10 frames)
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
    temporal_metrics = evaluator.evaluate_temporal_stability(frames)
    
    # Get summary
    summary = evaluator.get_summary()
    print("\nSummary Statistics:")
    for key, value in summary.items():
        if 'mean' in key:
            print(f"{key}: {value:.4f}")
    
    # Save results
    evaluator.save_metrics('evaluation_results.json')
    evaluator.plot_metrics('evaluation_plot.png')
    
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("\nColor Fidelity (Target Matching):")
    print("  - EMD < 5: Excellent match")
    print("  - EMD 5-10: Good match")
    print("  - EMD > 10: Poor match")
    print("  - Histogram Intersection > 0.8: Excellent")
    print("\nStructure Preservation:")
    print("  - SSIM > 0.9: Excellent detail preservation")
    print("  - SSIM 0.7-0.9: Good")
    print("  - SSIM < 0.7: Poor (over-processing)")
    print("  - PSNR > 30 dB: Good quality")
    print("\nTemporal Stability:")
    print("  - Flicker Index < 2: Minimal flicker")
    print("  - Flicker Index 2-5: Noticeable")
    print("  - Flicker Index > 5: Severe flickering")
    print("="*60)
