"""
Palette Extractor - Extract dominant color palettes from director's frames
Uses KMeans clustering in LAB color space for perceptually uniform results
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from glob import glob
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path


class PaletteExtractor:
    """Extract and aggregate color palettes from movie frames"""
    
    def __init__(self, n_colors=10, downsample_factor=4):
        """
        Initialize palette extractor
        
        Args:
            n_colors: Number of dominant colors to extract
            downsample_factor: Factor to downsample images for faster processing
        """
        self.n_colors = n_colors
        self.downsample_factor = downsample_factor
        
    def extract_frame_palette(self, frame, n_colors=None):
        """
        Extract dominant colors from a single frame using KMeans
        
        Args:
            frame: BGR image (numpy array)
            n_colors: Number of colors to extract (overrides self.n_colors if provided)
            
        Returns:
            colors: Array of shape (n_colors, 3) in LAB color space
            proportions: Array of shape (n_colors,) with color proportions
        """
        if n_colors is None:
            n_colors = self.n_colors
            
        # Convert to LAB color space (perceptually uniform)
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Downsample for faster processing
        h, w = lab_frame.shape[:2]
        small_frame = cv2.resize(lab_frame, 
                                (w // self.downsample_factor, 
                                 h // self.downsample_factor))
        
        # Reshape to pixel array
        pixels = small_frame.reshape(-1, 3)
        
        # Remove very dark pixels (shadows) - optional filtering
        # brightness_mask = pixels[:, 0] > 10  # L channel > 10
        # pixels = pixels[brightness_mask]
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get color centers and their proportions
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Calculate color proportions
        counts = np.bincount(labels, minlength=n_colors)
        proportions = counts / len(labels)
        
        return colors, proportions
    
    def create_director_palette(self, frames_dir, output_dir=None, 
                                director_name="Director", visualize=True):
        """
        Aggregate palettes from all director frames to create master palette
        
        Args:
            frames_dir: Directory containing extracted frames
            output_dir: Directory to save palette files
            director_name: Name of director for saving files
            visualize: Whether to create visualization
            
        Returns:
            master_palette: Array of shape (n_colors, 3) in LAB color space
            palette_info: Dictionary with additional palette information
        """
        if output_dir is None:
            output_dir = "Palettes"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect all frame paths
        frame_paths = sorted(glob(os.path.join(frames_dir, "*.jpg")))
        
        if len(frame_paths) == 0:
            raise ValueError(f"No frames found in {frames_dir}")
        
        print(f"Processing {len(frame_paths)} frames for {director_name}...")
        
        # Collect colors and weights from all frames
        all_colors = []
        all_weights = []
        
        for frame_path in tqdm(frame_paths, desc="Extracting frame palettes"):
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
                
            # Extract palette from this frame
            colors, proportions = self.extract_frame_palette(frame, n_colors=self.n_colors)
            
            all_colors.append(colors)
            all_weights.append(proportions)
        
        # Flatten all colors and weights
        all_colors = np.vstack(all_colors)
        all_weights = np.concatenate(all_weights)
        
        print(f"Collected {len(all_colors)} colors from all frames")
        print("Creating master palette with weighted clustering...")
        
        # Re-cluster to get final director palette (weighted by proportions)
        final_kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init=20)
        final_kmeans.fit(all_colors, sample_weight=all_weights)
        
        master_palette = final_kmeans.cluster_centers_
        
        # Calculate final proportions by assigning all colors to master palette
        final_labels = final_kmeans.predict(all_colors)
        final_counts = np.bincount(final_labels, weights=all_weights, minlength=self.n_colors)
        final_proportions = final_counts / final_counts.sum()
        
        # Sort by proportion (most dominant first)
        sort_idx = np.argsort(final_proportions)[::-1]
        master_palette = master_palette[sort_idx]
        final_proportions = final_proportions[sort_idx]
        
        # Create palette info dictionary
        palette_info = {
            'director': director_name,
            'n_frames': len(frame_paths),
            'n_colors': self.n_colors,
            'proportions': final_proportions.tolist(),
            'palette_lab': master_palette.tolist(),
            'palette_rgb': self._lab_to_rgb_palette(master_palette).tolist()
        }
        
        # Save palette
        palette_path = os.path.join(output_dir, f"{director_name}_palette.npy")
        np.save(palette_path, master_palette)
        print(f"Saved palette to {palette_path}")
        
        # Save palette info as JSON
        info_path = os.path.join(output_dir, f"{director_name}_palette_info.json")
        with open(info_path, 'w') as f:
            json.dump(palette_info, f, indent=2)
        print(f"Saved palette info to {info_path}")
        
        # Visualize
        if visualize:
            self.visualize_palette(master_palette, final_proportions, 
                                  director_name, output_dir)
        
        return master_palette, palette_info
    
    def _lab_to_rgb_palette(self, lab_palette):
        """Convert LAB palette to RGB for visualization"""
        # Create a small image with palette colors
        lab_img = lab_palette.reshape(1, -1, 3).astype(np.uint8)
        bgr_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return rgb_img.reshape(-1, 3)
    
    def visualize_palette(self, palette, proportions, director_name, output_dir):
        """
        Create visualization of the color palette
        
        Args:
            palette: LAB color palette
            proportions: Color proportions
            director_name: Name for the title
            output_dir: Directory to save visualization
        """
        # Convert to RGB for visualization
        rgb_palette = self._lab_to_rgb_palette(palette)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
        
        # Plot 1: Color swatches with proportions
        y_pos = 0
        for i, (color, prop) in enumerate(zip(rgb_palette, proportions)):
            height = prop * 10  # Scale for visibility
            ax1.barh(y_pos, 1, height=height, color=color/255.0, 
                    edgecolor='black', linewidth=0.5)
            ax1.text(1.02, y_pos, f'{prop*100:.1f}%', 
                    va='center', fontsize=9)
            y_pos += height
        
        ax1.set_xlim(0, 1.2)
        ax1.set_ylim(0, y_pos)
        ax1.axis('off')
        ax1.set_title(f'{director_name} Color Palette\n(Weighted by Proportion)', 
                     fontsize=12, fontweight='bold')
        
        # Plot 2: Equal-sized swatches
        swatch_height = 1
        for i, color in enumerate(rgb_palette):
            ax2.add_patch(plt.Rectangle((0, i*swatch_height), 1, swatch_height,
                                       color=color/255.0, edgecolor='black', linewidth=0.5))
            # Add LAB values
            L, a, b = palette[i]
            ax2.text(1.1, i*swatch_height + swatch_height/2, 
                    f'L:{L:.0f} a:{a:.0f} b:{b:.0f}',
                    va='center', fontsize=8, family='monospace')
        
        ax2.set_xlim(0, 2.5)
        ax2.set_ylim(0, len(rgb_palette)*swatch_height)
        ax2.axis('off')
        ax2.set_title('Color Swatches\n(LAB Values)', 
                     fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        viz_path = os.path.join(output_dir, f"{director_name}_palette_viz.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {viz_path}")
        plt.close()
        
        # Also create a simple swatch strip
        self._create_swatch_strip(rgb_palette, proportions, director_name, output_dir)
    
    def _create_swatch_strip(self, rgb_palette, proportions, director_name, output_dir):
        """Create a simple horizontal color strip"""
        # Create image with proportional widths
        strip_height = 100
        strip_width = 800
        strip = np.zeros((strip_height, strip_width, 3), dtype=np.uint8)
        
        x_pos = 0
        for color, prop in zip(rgb_palette, proportions):
            width = int(strip_width * prop)
            if width > 0:
                strip[:, x_pos:x_pos+width] = color
                x_pos += width
        
        # Convert RGB to BGR for cv2
        strip_bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
        
        # Save
        strip_path = os.path.join(output_dir, f"{director_name}_palette_strip.png")
        cv2.imwrite(strip_path, strip_bgr)
        print(f"Saved palette strip to {strip_path}")


def main():
    """Main function to extract palette from director frames"""
    # Configuration
    FRAMES_DIR = "Frames/Avatar"
    OUTPUT_DIR = "Palettes"
    DIRECTOR_NAME = "Avatar"
    N_COLORS = 10
    
    # Create extractor
    extractor = PaletteExtractor(n_colors=N_COLORS, downsample_factor=4)
    
    # Extract palette
    print(f"\n{'='*60}")
    print(f"PALETTE EXTRACTION FOR: {DIRECTOR_NAME}")
    print(f"{'='*60}\n")
    
    master_palette, palette_info = extractor.create_director_palette(
        frames_dir=FRAMES_DIR,
        output_dir=OUTPUT_DIR,
        director_name=DIRECTOR_NAME,
        visualize=True
    )
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nMaster Palette Shape: {master_palette.shape}")
    print(f"Total Frames Processed: {palette_info['n_frames']}")
    print(f"\nTop 5 Dominant Colors (LAB):")
    for i in range(min(5, len(master_palette))):
        L, a, b = master_palette[i]
        prop = palette_info['proportions'][i]
        print(f"  {i+1}. L={L:.1f}, a={a:.1f}, b={b:.1f} ({prop*100:.1f}%)")
    

if __name__ == "__main__":
    main()
