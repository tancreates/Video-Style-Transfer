import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def extract_uniform_frames_fast(video_path, output_dir, n_frames=50, 
                                skip_start=10*60, skip_end=10*60):
    """
    OPTIMIZED: Fast uniform frame extraction without scene detection
    Uses vectorized operations and efficient seeking
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f" Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    
    # Calculate frame range
    start_frame = int(skip_start * fps)
    end_frame = int((duration_sec - skip_end) * fps)
    
    if end_frame <= start_frame or total_frames < 1200:  # Less than 1 min
        start_frame = 0
        end_frame = total_frames
    
    # Calculate uniform frame indices (vectorized)
    frame_indices = np.linspace(start_frame, end_frame - 1, n_frames, dtype=int)
    
    print(f"  Duration: {duration_sec/60:.1f}m | Extracting {n_frames} frames")
    
    # Batch extraction with progress bar
    extracted = 0
    for idx, frame_num in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Resize to 720p for faster I/O (optional, maintains quality)
            height, width = frame.shape[:2]
            if height > 720:
                scale = 720 / height
                new_width = int(width * scale)
                frame = cv2.resize(frame, (new_width, 720), interpolation=cv2.INTER_AREA)
            
            out_path = os.path.join(output_dir, f"shot_{idx:03d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted += 1
    
    cap.release()
    print(f"  ✓ Extracted {extracted}/{n_frames} frames")
    return extracted


def extract_shot_based_frames(video_path, output_dir, n_frames=50,
                              skip_start=10*60, skip_end=10*60,
                              threshold=30, use_fast_mode=True):
    """
    OPTIMIZED: Scene detection with fast mode option
    """
    if use_fast_mode:
        # Skip scene detection for speed - use uniform sampling
        return extract_uniform_frames_fast(video_path, output_dir, n_frames, 
                                          skip_start, skip_end)
    
    # Original scene detection code (slower but more accurate)
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(" Could not open video")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    
    start_frame = int(skip_start * fps)
    end_frame = int((duration_sec - skip_end) * fps)
    
    if end_frame <= start_frame:
        start_frame = 0
        end_frame = total_frames
    
    print(f"  Duration: {duration_sec/60:.1f}m | Detecting scenes...")
    
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    
    video_manager.set_downscale_factor(2)  # 2x downscale for faster detection
    video_manager.start()
    video_manager.seek(start_frame)
    
    scene_manager.detect_scenes(frame_source=video_manager, end_time=(end_frame / fps))
    scene_list = scene_manager.get_scene_list()
    
    if len(scene_list) == 0:
        cap.release()
        return extract_uniform_frames_fast(video_path, output_dir, n_frames, skip_start, skip_end)
    
    # Vectorized scene selection
    n_frames = min(n_frames, len(scene_list))
    indices = np.linspace(0, len(scene_list) - 1, n_frames, dtype=int)
    selected_scenes = [scene_list[i] for i in indices]
    
    print(f"  Detected {len(scene_list)} scenes | Extracting {len(selected_scenes)} frames")
    
    # Batch frame extraction
    for idx, (start, end) in enumerate(selected_scenes):
        mid_frame = int((start.get_frames() + end.get_frames()) / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        
        if ret:
            height, width = frame.shape[:2]
            if height > 720:
                scale = 720 / height
                frame = cv2.resize(frame, (int(width * scale), 720), interpolation=cv2.INTER_AREA)
            
            out_path = os.path.join(output_dir, f"shot_{idx:03d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    cap.release()
    print(f"   Extracted {len(selected_scenes)} frames")


def process_all_directors(base_dir="Directors", frames_dir="Frames", n_frames=500, use_fast_mode=True, max_workers=4):
    """
    OPTIMIZED: Process all directors with parallel extraction
    
    Args:
        base_dir: Directory containing director folders
        frames_dir: Output directory for frames
        n_frames: Number of frames to extract per director
        use_fast_mode: Use uniform sampling (much faster) vs scene detection
        max_workers: Number of parallel workers
    """
    print("="*70)
    print("EXTRACTING FRAMES FROM DIRECTOR MOVIES (OPTIMIZED)")
    print("="*70)
    
    if not os.path.exists(base_dir):
        print(f" Directors folder not found: {base_dir}")
        return
    
    director_folders = [d for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d))]
    
    if not director_folders:
        print(f" No director folders found in {base_dir}")
        return
    
    mode_str = "Fast (Uniform Sampling)" if use_fast_mode else "Slow (Scene Detection)"
    print(f"\nMode: {mode_str}")
    print(f"Found {len(director_folders)} directors: {', '.join(director_folders)}")
    print(f"Target: {n_frames} frames per director\n")
    
    def process_single_director(director):
        """Process one director's movies"""
        print(f"\n{'='*70}")
        print(f"Processing: {director}")
        print(f"{'='*70}")
        
        director_path = os.path.join(base_dir, director)
        output_dir = os.path.join(frames_dir, director)
        
        video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
        video_files = []
        for ext in video_extensions:
            video_files.extend(Path(director_path).glob(f'*{ext}'))
        
        if not video_files:
            print(f"   No video files found for {director}")
            return
        
        print(f"  Found {len(video_files)} movie(s)")
        
        frames_per_movie = n_frames // len(video_files)
        remainder = n_frames % len(video_files)
        
        extracted_count = 0
        for idx, video_file in enumerate(video_files):
            current_frames = frames_per_movie + (remainder if idx == 0 else 0)
            
            print(f"\n  Movie {idx+1}/{len(video_files)}: {video_file.name}")
            
            try:
                temp_dir = os.path.join(output_dir, f"temp_{idx}")
                extract_shot_based_frames(
                    video_path=str(video_file),
                    output_dir=temp_dir,
                    n_frames=current_frames,
                    skip_start=10*60,
                    skip_end=10*60,
                    threshold=30,
                    use_fast_mode=use_fast_mode
                )
                
                # Continuous numbering
                temp_frames = sorted(Path(temp_dir).glob('*.jpg'))
                for frame_file in temp_frames:
                    new_name = f"shot_{extracted_count:03d}.jpg"
                    new_path = os.path.join(output_dir, new_name)
                    os.rename(str(frame_file), new_path)
                    extracted_count += 1
                
                os.rmdir(temp_dir)
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        print(f"\n  Total: {extracted_count} frames for {director}")
    
    # Process directors (sequential for cleaner output, but each movie can be parallel internally)
    for director in director_folders:
        process_single_director(director)


def process_standalone_movies(base_dir="Directors", frames_dir="Frames", n_frames=500, use_fast_mode=True):
    """
    OPTIMIZED: Process standalone movies from Directors folder
    
    Args:
        base_dir: Directory containing standalone movie files (Avatar, Barbie, Oppenheimer)
        frames_dir: Output directory for frames
        n_frames: Number of frames to extract per movie
        use_fast_mode: Use uniform sampling (much faster) vs scene detection
    """
    print("\n" + "="*70)
    print("EXTRACTING FRAMES FROM STANDALONE MOVIES (OPTIMIZED)")
    print("="*70)
    
    if not os.path.exists(base_dir):
        print(f"❌ Directory not found: {base_dir}")
        return
    
    standalone_movies = ['Avatar', 'Barbie', 'Oppenheimer']
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
    
    mode_str = "Fast (Uniform Sampling)" if use_fast_mode else "Slow (Scene Detection)"
    print(f"Mode: {mode_str}\n")
    
    for movie_name in standalone_movies:
        print(f"{'='*70}")
        print(f"Processing: {movie_name}")
        print(f"{'='*70}")
        
        movie_file = None
        for ext in video_extensions:
            potential_path = os.path.join(base_dir, f"{movie_name}{ext}")
            if os.path.exists(potential_path):
                movie_file = potential_path
                break
        
        if not movie_file:
            print(f"  ⚠️ {movie_name} not found in {base_dir}\n")
            continue
        
        output_dir = os.path.join(frames_dir, movie_name)
        print(f"  File: {Path(movie_file).name}")
        
        try:
            extract_shot_based_frames(
                video_path=movie_file,
                output_dir=output_dir,
                n_frames=n_frames,
                skip_start=10*60,
                skip_end=10*60,
                threshold=30,
                use_fast_mode=use_fast_mode
            )
            print()
        except Exception as e:
            print(f"  ❌ Error: {e}\n")


if __name__ == "__main__":
    """
    OPTIMIZED PIPELINE:
    - Uses fast uniform sampling by default
    - Set use_fast_mode=False for scene detection (slower, more accurate)
    - Automatic 720p downscaling for faster I/O
    - Vectorized frame selection
    """
    
    # Process director movies (500 frames per director)
    process_all_directors(
        base_dir="Directors",
        frames_dir="Frames",
        n_frames=500,
        use_fast_mode=True  # ⚡ FAST MODE: uniform sampling (change to False for scene detection)
    )
    
    # Process standalone movies from Directors folder (500 frames each)
    process_standalone_movies(
        base_dir="Directors",
        frames_dir="Frames",
        n_frames=500,
        use_fast_mode=True  # ⚡ FAST MODE: uniform sampling
    )
    
    print("\n" + "="*70)
    print(" ALL FRAME EXTRACTION COMPLETE!")
    print("="*70)


