"""
Batch Palette Generation - Extract palettes for all directors
Processes all directors and standalone movies in Frames/ directory
"""

import os
from pathlib import Path
from palette_extractor import PaletteExtractor
from color_statistics import ColorStatisticsAnalyzer


def generate_all_director_palettes(frames_base_dir="Frames", output_dir="Palettes", n_colors=10):
    """
    Generate palettes for all directors/movies in Frames directory
    
    Args:
        frames_base_dir: Base directory containing director/movie frame folders
        output_dir: Output directory for palette files
        n_colors: Number of dominant colors to extract per palette
    """
    print("="*70)
    print("BATCH PALETTE GENERATION FOR ALL DIRECTORS")
    print("="*70)
    
    if not os.path.exists(frames_base_dir):
        print(f"❌ Frames directory not found: {frames_base_dir}")
        return
    
    # Get all subdirectories (directors/movies)
    subdirs = [d for d in os.listdir(frames_base_dir) 
               if os.path.isdir(os.path.join(frames_base_dir, d))]
    
    if not subdirs:
        print(f"❌ No subdirectories found in {frames_base_dir}")
        return
    
    print(f"\nFound {len(subdirs)} directors/movies to process:")
    for i, name in enumerate(subdirs, 1):
        frame_count = len(list(Path(frames_base_dir, name).glob("*.jpg")))
        print(f"  {i}. {name} ({frame_count} frames)")
    
    print(f"\n{'='*70}")
    print("Starting palette extraction...")
    print(f"{'='*70}\n")
    
    # Initialize extractors
    palette_extractor = PaletteExtractor(n_colors=n_colors, downsample_factor=4)
    stats_analyzer = ColorStatisticsAnalyzer()
    
    success_count = 0
    failed = []
    
    for idx, director_name in enumerate(subdirs, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(subdirs)}] Processing: {director_name}")
        print(f"{'='*70}")
        
        frames_dir = os.path.join(frames_base_dir, director_name)
        
        try:
            # 1. Extract color palette
            print("\n1. Extracting dominant color palette...")
            master_palette, palette_info = palette_extractor.create_director_palette(
                frames_dir=frames_dir,
                output_dir=output_dir,
                director_name=director_name,
                visualize=True
            )
            
            # 2. Compute LAB statistics for Reinhard transfer
            print("\n2. Computing LAB statistics for transfer...")
            stats = stats_analyzer.compute_director_statistics(
                frames_dir=frames_dir,
                output_dir=output_dir,
                director_name=director_name
            )
            
            print(f"\n✓ Successfully processed {director_name}")
            print(f"  - Palette: {n_colors} colors extracted")
            print(f"  - Stats: mean_l={stats['mean_l']:.2f}, std_l={stats['std_l']:.2f}")
            print(f"  - Files saved to: {output_dir}/{director_name}_*")
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {director_name}: {e}")
            failed.append((director_name, str(e)))
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Successfully processed: {success_count}/{len(subdirs)}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for name, error in failed:
            print(f"  - {name}: {error}")
    
    print(f"\nAll palette files saved to: {output_dir}/")
    print("\nGenerated files per director:")
    print("  - <name>_palette.npy (palette array)")
    print("  - <name>_palette_info.json (palette metadata)")
    print("  - <name>_stats.json (LAB statistics for Reinhard)")
    print("  - <name>_palette_viz.png (visualization)")
    print("  - <name>_palette_strip.png (color strip)")


if __name__ == "__main__":
    generate_all_director_palettes(
        frames_base_dir="Frames",
        output_dir="Palettes",
        n_colors=10
    )
