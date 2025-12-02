"""
Palette Matching - Map frame colors to director palette using KNN
Implements direct color quantization and palette-based color grading
"""

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import json
import os


class PaletteMatcher:
    """
    Map source frame colors to director's palette using KNN
    """
    
    def __init__(self, director_palette=None, palette_info=None):
        """
        Initialize palette matcher
        
        Args:
            director_palette: Numpy array of shape (n_colors, 3) in LAB
            palette_info: Dictionary with palette information
        """
        self.director_palette = director_palette
        self.palette_info = palette_info
        self.knn_mapper = None
        
        if director_palette is not None:
            self._build_knn_mapper()
    
    def load_director_palette(self, palette_path, info_path=None):
        """
        Load director palette from file
        
        Args:
            palette_path: Path to .npy palette file
            info_path: Path to palette info JSON (optional)
        """
        self.director_palette = np.load(palette_path)
        
        if info_path and os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.palette_info = json.load(f)
        
        self._build_knn_mapper()
        
        return self.director_palette
    
    def _build_knn_mapper(self):
        """Build KNN model for color mapping"""
        if self.director_palette is None:
            raise ValueError("No director palette loaded")
        
        # Use KNN with distance weighting
        # k=1 for direct mapping to nearest color
        # k=3 for smoother mapping
        self.knn_mapper = KNeighborsClassifier(
            n_neighbors=3,
            weights='distance',
            metric='euclidean'
        )
        
        # Fit with palette colors (labels are just indices)
        n_colors = len(self.director_palette)
        labels = np.arange(n_colors)
        self.knn_mapper.fit(self.director_palette, labels)
    
    def map_frame_colors_hard(self, frame, color_space='LAB'):
        """
        Hard mapping: Replace each pixel with nearest palette color
        
        Args:
            frame: Input BGR image
            color_space: Color space for mapping ('LAB' or 'RGB')
            
        Returns:
            mapped_frame: BGR image with quantized colors
        """
        if self.knn_mapper is None:
            raise ValueError("KNN mapper not initialized. Load palette first.")
        
        # Convert to appropriate color space
        if color_space == 'LAB':
            frame_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        else:
            frame_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Reshape to pixel array
        h, w, c = frame_converted.shape
        pixels = frame_converted.reshape(-1, 3).astype(np.float32)
        
        # Find nearest palette colors
        nearest_indices = self.knn_mapper.predict(pixels)
        
        # Map to palette colors
        mapped_pixels = self.director_palette[nearest_indices]
        
        # Reshape back
        mapped_frame_converted = mapped_pixels.reshape(h, w, c).astype(np.uint8)
        
        # Convert back to BGR
        if color_space == 'LAB':
            mapped_frame = cv2.cvtColor(mapped_frame_converted, cv2.COLOR_LAB2BGR)
        else:
            mapped_frame = cv2.cvtColor(mapped_frame_converted, cv2.COLOR_RGB2BGR)
        
        return mapped_frame
    
    def map_frame_colors_soft(self, frame, blend_factor=0.5):
        """
        Soft mapping: Blend original with palette-mapped colors
        
        Args:
            frame: Input BGR image
            blend_factor: How much to blend with mapped colors (0-1)
            
        Returns:
            blended_frame: BGR image
        """
        # Get hard mapping
        mapped = self.map_frame_colors_hard(frame)
        
        # Blend with original
        blended = cv2.addWeighted(
            mapped, blend_factor,
            frame, 1 - blend_factor,
            0
        )
        
        return blended.astype(np.uint8)
    
    def map_frame_colors_weighted(self, frame, k=3):
        """
        Weighted mapping: Use weighted average of k nearest palette colors
        
        Args:
            frame: Input BGR image
            k: Number of nearest neighbors to consider
            
        Returns:
            mapped_frame: BGR image
        """
        if self.director_palette is None:
            raise ValueError("No palette loaded")
        
        # Convert to LAB
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Reshape
        h, w, c = frame_lab.shape
        pixels = frame_lab.reshape(-1, 3)
        
        # Build temporary KNN for this k
        knn_temp = KNeighborsClassifier(n_neighbors=k, weights='distance')
        labels = np.arange(len(self.director_palette))
        knn_temp.fit(self.director_palette, labels)
        
        # Get k nearest neighbors and their distances
        distances, indices = knn_temp.kneighbors(pixels)
        
        # Compute weights (inverse distance)
        weights = 1.0 / (distances + 1e-6)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Weighted average of palette colors
        mapped_pixels = np.zeros_like(pixels)
        for i in range(len(pixels)):
            neighbor_colors = self.director_palette[indices[i]]
            mapped_pixels[i] = np.average(neighbor_colors, axis=0, weights=weights[i])
        
        # Reshape and convert back
        mapped_lab = mapped_pixels.reshape(h, w, c).astype(np.uint8)
        mapped_bgr = cv2.cvtColor(mapped_lab, cv2.COLOR_LAB2BGR)
        
        return mapped_bgr
    
    def map_with_luminance_preservation(self, frame, mapping_strength=0.8):
        """
        Map colors while preserving original luminance
        
        Args:
            frame: Input BGR image
            mapping_strength: Strength of color mapping
            
        Returns:
            result_frame: BGR image
        """
        # Convert to LAB
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        original_L = frame_lab[:, :, 0].copy()
        
        # Map colors
        mapped = self.map_frame_colors_weighted(frame, k=3)
        mapped_lab = cv2.cvtColor(mapped, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Replace luminance with original
        result_lab = mapped_lab.copy()
        result_lab[:, :, 0] = original_L * (1 - mapping_strength) + mapped_lab[:, :, 0] * mapping_strength
        
        # Clip and convert
        result_lab = np.clip(result_lab, 0, 255)
        result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result_bgr
    
    def adaptive_palette_mapping(self, frame, local_mapping=True, window_size=15):
        """
        Adaptive mapping that considers local image context
        
        Args:
            frame: Input BGR image
            local_mapping: Whether to use local adaptive mapping
            window_size: Size of local window
            
        Returns:
            mapped_frame: BGR image
        """
        if not local_mapping:
            return self.map_frame_colors_weighted(frame, k=3)
        
        # Convert to LAB
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = frame_lab.shape[:2]
        result_lab = np.zeros_like(frame_lab)
        
        # Apply bilateral filter for edge-aware smoothing
        frame_lab_smooth = cv2.bilateralFilter(
            frame_lab.astype(np.uint8), 9, 75, 75
        ).astype(np.float32)
        
        # Map smoothed version
        mapped_smooth = self.map_frame_colors_weighted(
            cv2.cvtColor(frame_lab_smooth.astype(np.uint8), cv2.COLOR_LAB2BGR), k=3
        )
        mapped_smooth_lab = cv2.cvtColor(mapped_smooth, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Preserve high-frequency details from original
        detail_L = frame_lab[:, :, 0] - frame_lab_smooth[:, :, 0]
        
        result_lab = mapped_smooth_lab.copy()
        result_lab[:, :, 0] += detail_L * 0.5  # Add back some detail
        
        # Clip and convert
        result_lab = np.clip(result_lab, 0, 255)
        result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result_bgr


class RegionAwarePaletteMapping:
    """
    Advanced palette mapping with region awareness
    Different regions may map to different palette colors
    """
    
    def __init__(self, director_palette):
        self.director_palette = director_palette
    
    def segment_image(self, frame, n_segments=5):
        """
        Segment image into regions using KMeans in LAB space
        
        Args:
            frame: Input BGR image
            n_segments: Number of segments
            
        Returns:
            segments: Segmentation mask
            segment_colors: Average color of each segment
        """
        # Convert to LAB
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        h, w = frame_lab.shape[:2]
        
        # Reshape
        pixels = frame_lab.reshape(-1, 3)
        
        # Cluster
        kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        segments = kmeans.fit_predict(pixels)
        segments = segments.reshape(h, w)
        
        segment_colors = kmeans.cluster_centers_
        
        return segments, segment_colors
    
    def map_by_region(self, frame, n_segments=5):
        """
        Map each region to its optimal palette color
        
        Args:
            frame: Input BGR image
            n_segments: Number of regions
            
        Returns:
            mapped_frame: BGR image
        """
        # Segment image
        segments, segment_colors = self.segment_image(frame, n_segments)
        
        # Find best palette color for each segment
        segment_palette_mapping = []
        for seg_color in segment_colors:
            # Find nearest palette color
            distances = np.linalg.norm(self.director_palette - seg_color, axis=1)
            best_palette_idx = np.argmin(distances)
            segment_palette_mapping.append(best_palette_idx)
        
        # Create mapped image
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        result_lab = frame_lab.copy()
        
        # Map each segment
        for seg_id, palette_idx in enumerate(segment_palette_mapping):
            mask = segments == seg_id
            target_color = self.director_palette[palette_idx]
            
            # Shift colors toward palette color
            for c in range(3):
                shift = target_color[c] - segment_colors[seg_id][c]
                result_lab[:, :, c][mask] += shift * 0.6
        
        # Clip and convert
        result_lab = np.clip(result_lab, 0, 255)
        result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result_bgr


def test_palette_matching():
    """Test palette matching methods"""
    import matplotlib.pyplot as plt
    
    print("Testing Palette Matching Methods...")
    
    # Load sample frame
    sample_frame_path = "Frames/Avatar/shot_000.jpg"
    if not os.path.exists(sample_frame_path):
        print(f"Sample frame not found: {sample_frame_path}")
        return
    
    source_frame = cv2.imread(sample_frame_path)
    
    # Load director palette
    palette_path = "Palettes/Avatar_palette.npy"
    info_path = "Palettes/Avatar_palette_info.json"
    
    # Initialize matcher
    matcher = PaletteMatcher()
    matcher.load_director_palette(palette_path, info_path)
    
    print(f"Loaded palette with {len(matcher.director_palette)} colors")
    
    # Test different mapping methods
    results = {}
    
    print("\n1. Hard mapping (nearest color)...")
    results['hard'] = matcher.map_frame_colors_hard(source_frame)
    
    print("2. Soft mapping (50% blend)...")
    results['soft'] = matcher.map_frame_colors_soft(source_frame, blend_factor=0.5)
    
    print("3. Weighted mapping (k=3)...")
    results['weighted'] = matcher.map_frame_colors_weighted(source_frame, k=3)
    
    print("4. Luminance-preserving mapping...")
    results['lum_preserve'] = matcher.map_with_luminance_preservation(source_frame, 0.7)
    
    print("5. Adaptive mapping...")
    results['adaptive'] = matcher.adaptive_palette_mapping(source_frame, local_mapping=True)
    
    print("6. Region-aware mapping...")
    region_matcher = RegionAwarePaletteMapping(matcher.director_palette)
    results['region'] = region_matcher.map_by_region(source_frame, n_segments=5)
    
    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')
    
    # Results
    titles = ['Hard Mapping', 'Soft Mapping', 'Weighted (k=3)', 
              'Luminance Preserve', 'Adaptive', 'Region-Aware']
    
    for i, (key, title) in enumerate(zip(results.keys(), titles), 1):
        axes[i].imshow(cv2.cvtColor(results[key], cv2.COLOR_BGR2RGB))
        axes[i].set_title(title, fontweight='bold')
        axes[i].axis('off')
    
    # Hide last unused subplot
    axes[7].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = "Palettes/palette_matching_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison saved to: {output_path}")
    
    # Save individual results
    output_dir = "Palettes/palette_matching_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, result in results.items():
        out_path = os.path.join(output_dir, f"{name}.jpg")
        cv2.imwrite(out_path, result)
    
    print(f"✓ Individual results saved to: {output_dir}/")
    
    plt.close()
    
    return results


if __name__ == "__main__":
    print("="*60)
    print("PALETTE MATCHING MODULE - Testing")
    print("="*60)
    
    test_palette_matching()
    
    print("\n" + "="*60)
    print("PALETTE MATCHING TEST COMPLETE!")
    print("="*60)
