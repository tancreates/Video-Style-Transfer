"""
Generate Statistics Only - For directors that already have palettes
Quickly generates LAB statistics without re-extracting palettes
"""

import os
from pathlib import Path
from color_statistics import ColorStatisticsAnalyzer


def generate_missing_statistics(frames_base_dir="Frames", output_dir="Palettes"):
    """
    Generate statistics for directors that have frames but no stats
    
    Args:
        frames_base_dir: Base directory containing director/movie frame folders
        output_dir: Output directory for statistics files
    """
    print("="*70)
    print("GENERATING MISSING LAB STATISTICS")
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
    
    # Check which directors already have stats
    existing_stats = []
    if os.path.exists(output_dir):
        existing_stats = [f.stem.replace('_stats', '') 
                         for f in Path(output_dir).glob('*_stats.json')]
    
    # Find directors missing stats
    missing_stats = [d for d in subdirs if d not in existing_stats]
    
    print(f"\nTotal directors: {len(subdirs)}")
    print(f"Already have stats: {len(existing_stats)}")
    print(f"Missing stats: {len(missing_stats)}")
    
    if not missing_stats:
        print("\n✓ All directors already have statistics!")
        return
    
    print(f"\nDirectors to process:")
    for i, name in enumerate(missing_stats, 1):
        frame_count = len(list(Path(frames_base_dir, name).glob("*.jpg")))
        print(f"  {i}. {name} ({frame_count} frames)")
    
    print(f"\n{'='*70}")
    print("Starting statistics computation...")
    print(f"{'='*70}\n")
    
    # Initialize analyzer
    stats_analyzer = ColorStatisticsAnalyzer()
    
    success_count = 0
    failed = []
    
    for idx, director_name in enumerate(missing_stats, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(missing_stats)}] Processing: {director_name}")
        print(f"{'='*70}")
        
        frames_dir = os.path.join(frames_base_dir, director_name)
        
        try:
            stats = stats_analyzer.compute_director_statistics(
                frames_dir=frames_dir,
                output_dir=output_dir,
                director_name=director_name
            )
            
            print(f"\n✓ Successfully generated statistics for {director_name}")
            print(f"  - mean_l={stats['mean_l']:.2f}, std_l={stats['std_l']:.2f}")
            print(f"  - File: {output_dir}/{director_name}_stats.json")
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {director_name}: {e}")
            failed.append((director_name, str(e)))
            continue
    
    # Summary
    print(f"\n{'='*70}")
    print("STATISTICS GENERATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Successfully processed: {success_count}/{len(missing_stats)}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for name, error in failed:
            print(f"  - {name}: {error}")
    
    print(f"\nAll statistics saved to: {output_dir}/*_stats.json")


if __name__ == "__main__":
    generate_missing_statistics(
        frames_base_dir="Frames",
        output_dir="Palettes"
    )
