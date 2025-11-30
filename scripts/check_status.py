"""
Check Director Setup Status
Shows which directors have frames and palettes ready
"""

import os
from pathlib import Path
from colorama import init, Fore, Style

# Initialize colorama for Windows
init(autoreset=True)


def check_status():
    """Check which directors are ready to use"""
    
    print("="*70)
    print("DIRECTOR PIPELINE STATUS CHECK")
    print("="*70)
    
    frames_dir = Path("Frames")
    palettes_dir = Path("Palettes")
    
    # Get all directors with extracted frames
    if not frames_dir.exists():
        print(f"{Fore.RED} Frames directory not found!{Style.RESET_ALL}")
        print("Run: python shot_frames.py")
        return
    
    directors_with_frames = sorted([d.name for d in frames_dir.iterdir() if d.is_dir()])
    
    if not directors_with_frames:
        print(f"{Fore.RED} No directors found in Frames directory!{Style.RESET_ALL}")
        print("Run: python shot_frames.py")
        return
    
    # Get directors with palettes
    directors_with_palettes = []
    if palettes_dir.exists():
        stats_files = list(palettes_dir.glob("*_stats.json"))
        directors_with_palettes = sorted([f.stem.replace('_stats', '') for f in stats_files])
    
    print(f"\n{Fore.CYAN}Directors with Extracted Frames:{Style.RESET_ALL} {len(directors_with_frames)}")
    print(f"{Fore.CYAN}Directors with Generated Palettes:{Style.RESET_ALL} {len(directors_with_palettes)}\n")
    
    # Status table
    print(f"{'Director':<25} {'Frames':<10} {'Palette':<10} {'Status':<15}")
    print("-"*70)
    
    ready_count = 0
    missing_count = 0
    
    for director in directors_with_frames:
        frame_count = len(list((frames_dir / director).glob("*.jpg")))
        has_frames = "✓" if frame_count > 0 else "✗"
        has_palette = "✓" if director in directors_with_palettes else "✗"
        
        if has_frames == "✓" and has_palette == "✓":
            status = f"{Fore.GREEN}READY{Style.RESET_ALL}"
            ready_count += 1
        else:
            status = f"{Fore.YELLOW}MISSING PALETTE{Style.RESET_ALL}"
            missing_count += 1
        
        frames_col = f"{Fore.GREEN}{has_frames}{Style.RESET_ALL} ({frame_count})"
        palette_col = f"{Fore.GREEN}{has_palette}{Style.RESET_ALL}" if has_palette == "✓" else f"{Fore.RED}{has_palette}{Style.RESET_ALL}"
        
        print(f"{director:<25} {frames_col:<20} {palette_col:<20} {status}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{Fore.GREEN} Ready to use:{Style.RESET_ALL} {ready_count} directors")
    print(f"{Fore.YELLOW} Need palette generation:{Style.RESET_ALL} {missing_count} directors")
    
    if missing_count > 0:
        print(f"\n{Fore.CYAN}To generate missing palettes, run:{Style.RESET_ALL}")
        print(f"  {Fore.WHITE}python generate_all_palettes.py{Style.RESET_ALL}")
    
    if ready_count > 0:
        print(f"\n{Fore.CYAN}Example usage:{Style.RESET_ALL}")
        ready_director = directors_with_palettes[0] if directors_with_palettes else None
        if ready_director:
            print(f"  {Fore.WHITE}python final_pipeline.py -i video.mp4 -o output.mp4 -d {ready_director} -t{Style.RESET_ALL}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        check_status()
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
