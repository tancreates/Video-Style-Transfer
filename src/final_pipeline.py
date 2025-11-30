"""
Final Integrated Pipeline - Director Color Style Transfer
Proper Reinhard implementation + Enhanced DIP techniques
"""

import cv2
import numpy as np
import json
import os
from tqdm import tqdm
import argparse
from pathlib import Path

from enhanced_color_transfer import EnhancedColorTransfer
from color_statistics import ColorStatisticsAnalyzer


class DirectorStyleTransfer:
    """
    Main pipeline for transferring director's color style using proper Reinhard method
    """
    
    def __init__(self, director_name=None, palettes_dir="Palettes", reference_path=None):
        """
        Initialize the pipeline
        
        Args:
            director_name: Name of director (e.g., 'Avatar') - optional if reference_path provided
            palettes_dir: Directory containing statistics files
            reference_path: Path to reference image/video for custom style transfer
        """
        self.director_name = director_name
        self.palettes_dir = palettes_dir
        self.reference_path = reference_path
        
        # Load or compute statistics
        if reference_path:
            self._compute_reference_stats()
        elif director_name:
            self._load_director_data()
        else:
            raise ValueError("Must provide either director_name or reference_path")
        
        # Initialize transfer module
        self.color_transfer = EnhancedColorTransfer(self.stats)
        
        if reference_path:
            print(f"Initialized pipeline with custom reference: {reference_path}")
        else:
            print(f"Initialized pipeline for {director_name}")
        print(f"  - Statistics (lab space):")
        print(f"    Mean: ({self.stats['mean_l']:.4f}, {self.stats['mean_alpha']:.4f}, {self.stats['mean_beta']:.4f})")
        print(f"    Std:  ({self.stats['std_l']:.4f}, {self.stats['std_alpha']:.4f}, {self.stats['std_beta']:.4f})")
    
    def _load_director_data(self):
        """Load director's statistics"""
        stats_path = os.path.join(self.palettes_dir, f"{self.director_name}_stats.json")
        
        if not os.path.exists(stats_path):
            # List available directors
            available = self._get_available_directors()
            error_msg = f"Statistics not found for '{self.director_name}': {stats_path}\n"
            if available:
                error_msg += f"\nAvailable directors: {', '.join(available)}\n"
                error_msg += f"\nTo generate palettes for all directors, run:\n"
                error_msg += f"  python generate_all_palettes.py"
            else:
                error_msg += f"\nNo director palettes found in '{self.palettes_dir}'.\n"
                error_msg += f"Run: python generate_all_palettes.py"
            raise FileNotFoundError(error_msg)
        
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
    
    def _get_available_directors(self):
        """Get list of directors with generated palettes"""
        if not os.path.exists(self.palettes_dir):
            return []
        
        stats_files = Path(self.palettes_dir).glob("*_stats.json")
        directors = [f.stem.replace('_stats', '') for f in stats_files]
        return sorted(directors)
    
    def _compute_reference_stats(self):
        """Compute statistics from reference image or video"""
        print(f"\nComputing statistics from reference: {self.reference_path}")
        
        if not os.path.exists(self.reference_path):
            raise FileNotFoundError(f"Reference not found: {self.reference_path}")
        
        # Check if video or image
        ext = Path(self.reference_path).suffix.lower()
        is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
        
        analyzer = ColorStatisticsAnalyzer()
        
        if is_video:
            # Sample frames from video
            cap = cv2.VideoCapture(self.reference_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample up to 100 frames evenly distributed
            num_samples = min(100, total_frames)
            frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
            
            print(f"  Sampling {num_samples} frames from video...")
            
            all_stats = []
            for idx in tqdm(frame_indices, desc="Analyzing frames"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    stats = analyzer.compute_frame_statistics(frame)
                    all_stats.append(stats)
            
            cap.release()
            
            # Average statistics across all frames
            self.stats = {
                'mean_l': np.mean([s['mean_l'] for s in all_stats]),
                'mean_alpha': np.mean([s['mean_alpha'] for s in all_stats]),
                'mean_beta': np.mean([s['mean_beta'] for s in all_stats]),
                'std_l': np.mean([s['std_l'] for s in all_stats]),
                'std_alpha': np.mean([s['std_alpha'] for s in all_stats]),
                'std_beta': np.mean([s['std_beta'] for s in all_stats])
            }
        else:
            # Single image
            reference_img = cv2.imread(self.reference_path)
            if reference_img is None:
                raise ValueError(f"Cannot read reference image: {self.reference_path}")
            
            print(f"  Analyzing image...")
            self.stats = analyzer.compute_frame_statistics(reference_img)
        print(f"Statistics computed from reference")
    
    def _load_target_pixels_for_idt(self):
        """
        Load and sample pixels from director frames for IDT method
        Needed for Iterative Distribution Transfer
        """
        from glob import glob
        
        frames_dir = f"Frames/{self.director_name}"
        frame_paths = sorted(glob(f"{frames_dir}/*.jpg"))
        
        if not frame_paths:
            raise ValueError(f"No frames found in {frames_dir} for IDT method")
        
        all_pixels = []
        
        # Sample pixels from multiple frames
        sample_size = min(20, len(frame_paths))
        sampled_paths = np.random.choice(frame_paths, sample_size, replace=False)
        
        for frame_path in sampled_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Convert to lab
            lab = self.color_transfer.bgr_to_lab(frame)
            pixels = lab.reshape(-1, 3)
            
            # Sample 500 pixels per frame
            if len(pixels) > 500:
                indices = np.random.choice(len(pixels), 500, replace=False)
                pixels = pixels[indices]
            
            all_pixels.append(pixels)
        
        # Concatenate
        target_pixels = np.vstack(all_pixels)
        
        return target_pixels
    
    def process_frame(self, frame, method='reinhard', **kwargs):
        """
        Process a single frame
        
        Args:
            frame: Input BGR image
            method: Transfer method to use
            **kwargs: Additional parameters
            
        Returns:
            processed_frame: BGR image with director's style
        """
        if method == 'reinhard':
            strength = kwargs.get('strength', 1.0)
            return self.color_transfer.transfer_lab_statistics(frame, self.stats, strength)
        
        elif method == 'multiscale':
            return self.color_transfer.multiscale_transfer(frame, self.stats)
        
        elif method == 'bilateral':
            return self.color_transfer.bilateral_color_transfer(frame, self.stats)
        
        elif method == 'detail_preserve':
            return self.color_transfer.detail_preserving_transfer(frame, self.stats)
        
        elif method == 'chroma_only':
            strength = kwargs.get('strength', 1.0)
            return self.color_transfer.chrominance_only_transfer(frame, self.stats, strength)
        
        elif method == 'progressive':
            iterations = kwargs.get('iterations', 3)
            return self.color_transfer.progressive_transfer(frame, self.stats, iterations)
        
        elif method == 'mkl' or method == 'monge_kantorovitch':
            strength = kwargs.get('strength', 1.0)
            return self.color_transfer.monge_kantorovitch_transfer(frame, self.stats, strength)
        
        elif method == 'idt' or method == 'iterative_distribution':
            # Load target pixels if not already cached
            if not hasattr(self, '_idt_target_pixels'):
                self._idt_target_pixels = self._load_target_pixels_for_idt()
            
            iterations = kwargs.get('iterations', 20)
            strength = kwargs.get('strength', 1.0)
            return self.color_transfer.iterative_distribution_transfer(
                frame, self._idt_target_pixels, iterations, strength
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def process_video(self, input_video_path, output_video_path, 
                     method='reinhard', temporal_smooth=True, **kwargs):
        """
        Process entire video
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to save output video
            method: Transfer method
            temporal_smooth: Apply temporal smoothing
            **kwargs: Additional parameters
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING VIDEO: {input_video_path}")
        print(f"{'='*70}")
        print(f"Method: {method}")
        print(f"Temporal smoothing: {temporal_smooth}")
        
        # Open video
        cap = cv2.VideoCapture(input_video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / original_fps if original_fps > 0 else 0
        
        # Smart resolution downscaling for faster processing
        max_dimension = max(width, height)
        if max_dimension > 1080:
            scale_factor = 1080 / max_dimension
            process_width = int(width * scale_factor)
            process_height = int(height * scale_factor)
            downscale = True
            print(f"Downscaling: {width}x{height} -> {process_width}x{process_height} (for faster processing)")
        else:
            process_width = width
            process_height = height
            downscale = False
        
        # Smart FPS adjustment based on video length
        if duration_sec > 600:  # > 10 minutes
            target_fps = 20
            frame_skip = max(1, int(original_fps / target_fps))
            print(f"Long video ({duration_sec/60:.1f} min) - reducing to {target_fps}fps (every {frame_skip} frame)")
        elif duration_sec > 180:  # > 3 minutes
            target_fps = 24
            frame_skip = max(1, int(original_fps / target_fps))
            print(f"Medium video ({duration_sec/60:.1f} min) - reducing to {target_fps}fps (every {frame_skip} frame)")
        else:
            target_fps = original_fps
            frame_skip = 1
        
        fps = int(target_fps)
        estimated_frames = total_frames // frame_skip
        
        print(f"Original: {width}x{height} @ {original_fps:.0f}fps, {total_frames} frames ({duration_sec/60:.1f} min)")
        print(f"Output: {width}x{height} @ {fps}fps")
        print(f"Frames to process: {estimated_frames} (every {frame_skip} frame)")
        # Estimate based on 3-6 frames/sec processing speed
        est_time_min = estimated_frames / 4.5 / 60
        print(f"Estimated time: ~{est_time_min:.1f} minutes")
        
        # Special handling for IDT method to prevent flickering
        use_fixed_idt_mapping = False
        idt_mapping_frames = None
        
        if method in ['idt', 'iterative_distribution']:
            print("\nIDT Method Detected - Using anti-flicker strategy")
            print("Computing fixed color mapping from representative frames...")
            
            # Sample representative frames for creating stable mapping
            sample_indices = np.linspace(0, total_frames - 1, min(20, total_frames), dtype=int)
            sample_frames = []
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    if downscale:
                        frame = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)
                    sample_frames.append(frame)
            
            # Reset video capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            print(f"Sampled {len(sample_frames)} representative frames")
            print("Pre-computing IDT color mapping (this may take a minute)...")
            
            # Compute IDT mapping on aggregated sample
            # This creates a fixed transformation that won't change per frame
            use_fixed_idt_mapping = True
            
            # Store sample frames for creating consistent mapping
            idt_mapping_frames = sample_frames
        
        # Create temporary output (no audio)
        temp_output = output_video_path.replace('.mp4', '_temp_no_audio.mp4')
        
        # Create video writer with target fps (original resolution)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
        
        # Process frames
        prev_frame = None
        prev_processed_lab = None  # For IDT temporal smoothing in lab space
        alpha = 0.8  # Temporal smoothing factor (0.8 = 80% current, 20% previous)
        idt_alpha = 0.7  # Stronger smoothing for IDT (70% current, 30% previous)
        frame_count = 0
        
        with tqdm(total=estimated_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skipping frames if needed
                if frame_count % frame_skip != 0:
                    continue
                
                # Downscaling for processing if needed
                if downscale:
                    frame_small = cv2.resize(frame, (process_width, process_height), interpolation=cv2.INTER_AREA)
                else:
                    frame_small = frame
                
                # Special processing for IDT to reduce flicker
                if use_fixed_idt_mapping and method in ['idt', 'iterative_distribution']:
                    # Use cached target pixels and apply with temporal smoothing
                    if not hasattr(self, '_idt_target_pixels_video'):
                        # Create target pixels once from sample frames
                        print("Creating fixed IDT target distribution...")
                        all_sample_pixels = []
                        for sample_frame in idt_mapping_frames:
                            lab = self.color_transfer.bgr_to_lab(sample_frame)
                            pixels = lab.reshape(-1, 3)
                            # Sample 200 pixels per frame
                            if len(pixels) > 200:
                                indices = np.random.choice(len(pixels), 200, replace=False)
                                pixels = pixels[indices]
                            all_sample_pixels.append(pixels)
                        self._idt_target_pixels_video = np.vstack(all_sample_pixels)
                        print(f"Created fixed target distribution with {len(self._idt_target_pixels_video)} pixels")
                    
                    # Processing with fixed target and fewer iterations for consistency
                    iterations = kwargs.get('iterations', 10)  # Fewer iterations for video
                    strength = kwargs.get('strength', 1.0)
                    
                    # Convert to lab
                    frame_lab = self.color_transfer.bgr_to_lab(frame_small)
                    
                    # Apply IDT with fixed random seed for reproducibility
                    np.random.seed(42)  # Fixed seed for consistent random projections
                    processed_lab = self.color_transfer.iterative_distribution_transfer(
                        frame_small, 
                        self._idt_target_pixels_video, 
                        iterations=iterations, 
                        strength=strength
                    )
                    
                    # Temporal smoothing in lab space (reduces flicker significantly)
                    if prev_processed_lab is not None:
                        processed_lab_smooth = self.color_transfer.bgr_to_lab(processed_lab)
                        
                        # Blend in lab space
                        blended_lab = (
                            idt_alpha * processed_lab_smooth + 
                            (1 - idt_alpha) * prev_processed_lab
                        )
                        
                        # Convert back to BGR
                        processed = self.color_transfer.lab_to_bgr(blended_lab)
                        prev_processed_lab = blended_lab.copy()
                    else:
                        prev_processed_lab = self.color_transfer.bgr_to_lab(processed_lab)
                        processed = processed_lab
                else:
                    # Standard processing for other methods
                    processed = self.process_frame(frame_small, method=method, **kwargs)
                
                # Upscaling if needed
                if downscale:
                    processed = cv2.resize(processed, (width, height), interpolation=cv2.INTER_LANCZOS4)
                
                # Standard temporal smoothing (for non-IDT methods)
                if temporal_smooth and prev_frame is not None and method not in ['idt', 'iterative_distribution']:
                    processed = cv2.addWeighted(
                        processed, alpha,
                        prev_frame, 1 - alpha,
                        0
                    ).astype(np.uint8)
                
                prev_frame = processed.copy()
                
                # Write frame
                out.write(processed)
                
                pbar.update(1)
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"\nVideo processing complete")
        
        # Merge audio from original video using ffmpeg
        print(f"Merging audio from original video...")
        import subprocess
        
        try:
            # Checking if ffmpeg is available
            subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, check=True)
            
            # Merge video with audio
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', temp_output,  # Processed video (no audio)
                '-i', input_video_path,  # Original video (with audio)
                '-c:v', 'copy',  # Copy video stream
                '-c:a', 'aac',  # Encode audio as AAC
                '-map', '0:v:0',  # Video from first input
                '-map', '1:a:0?',  # Audio from second input (optional)
                '-shortest',  # Match shortest stream duration
                output_video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Audio merged successfully")
                # Remove temporary file
                os.remove(temp_output)
            else:
                print(f"Warning: Could not merge audio (ffmpeg error)")
                print(f"  Video saved without audio: {temp_output}")
                print(f"  You can manually merge audio using ffmpeg")
                # Renaming temp file to output
                if os.path.exists(temp_output):
                    os.rename(temp_output, output_video_path)
        
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Warning: ffmpeg not found")
            print(f"  Video saved without audio: {temp_output}")
            print(f"  Install ffmpeg to preserve audio:")
            print(f"    Windows: choco install ffmpeg  OR  download from ffmpeg.org")
            print(f"    You can manually merge audio later using:")
            print(f"    ffmpeg -i {temp_output} -i {input_video_path} -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 {output_video_path}")
            # Rename temp file to output
            if os.path.exists(temp_output):
                os.rename(temp_output, output_video_path)
        
        print(f"\nFinal video saved to: {output_video_path}")
        print(f"{'='*70}\n")
    
    def process_image(self, input_image_path, output_image_path, 
                     method='reinhard', **kwargs):
        """
        Process a single image
        
        Args:
            input_image_path: Path to input image
            output_image_path: Path to save output image
            method: Transfer method
            **kwargs: Additional parameters
        """
        # Reading image
        image = cv2.imread(input_image_path)
        
        if image is None:
            raise ValueError(f"Cannot read image: {input_image_path}")
        
        print(f"Processing image: {input_image_path}")
        print(f"Method: {method}")
        
        # Process
        processed = self.process_frame(image, method=method, **kwargs)
        
        # Save
        cv2.imwrite(output_image_path, processed)
        
        print(f"Saved to: {output_image_path}")
        
        return processed
    
    def compare_methods(self, input_image_path, output_dir="Comparisons"):
        """
        Compare all transfer methods on a single image
        
        Args:
            input_image_path: Path to input image
            output_dir: Directory to save comparison
        """
        import matplotlib.pyplot as plt
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Reading image
        image = cv2.imread(input_image_path)
        
        if image is None:
            raise ValueError(f"Cannot read image: {input_image_path}")
        
        print(f"\n{'='*70}")
        print(f"COMPARING METHODS ON: {input_image_path}")
        print(f"{'='*70}\n")
        
        # Processing with different methods
        results = {
            'Original': image,
        }
        
        methods = [
            ('reinhard', 'Reinhard (s=0.7)', {'strength': 0.7}),
            ('reinhard', 'Reinhard (s=1.0)', {'strength': 1.0}),
            ('multiscale', 'Multi-Scale', {}),
            ('bilateral', 'Bilateral Filter', {}),
            ('detail_preserve', 'Detail Preserve', {}),
            ('chroma_only', 'Chroma Only', {'strength': 1.0}),
            ('progressive', 'Progressive', {'iterations': 3}),
            ('mkl', 'Monge-Kantorovitch', {'strength': 1.0}),
            ('idt', 'Iterative Distribution', {'iterations': 15, 'strength': 1.0}),
        ]
        
        for method, name, params in methods:
            print(f"Processing: {name}...")
            try:
                results[name] = self.process_frame(image, method=method, **params)
            except Exception as e:
                print(f"  Warning: {name} failed - {e}")
        
        # Creating comparison grid (3x3 for 9 methods)
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, (name, result) in enumerate(results.items()):
            if i < len(axes):
                axes[i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                axes[i].set_title(name, fontsize=12, fontweight='bold')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(results), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        #saving comparison
        comparison_path = os.path.join(output_dir, f"{self.director_name}_final_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison saved to: {comparison_path}")
        
        # Save individual results
        for name, result in results.items():
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            result_path = os.path.join(output_dir, f"{safe_name}.jpg")
            cv2.imwrite(result_path, result)
        
        plt.close()
        
        print(f"{'='*70}\n")
        
        return results


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='Director Color Style Transfer - Proper Reinhard Implementation'
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input video or image path')
    parser.add_argument('--output', '-o', 
                       help='Output path (required unless --compare is used)')
    parser.add_argument('--reference', '-r',
                       help='Reference image or video for custom style transfer (optional, overrides --director)')
    parser.add_argument('--director', '-d', default='Avatar',
                       help='Director style to apply (e.g., Avatar, WesAnderson, ChristopherNolan, etc. - ignored if --reference provided)')
    parser.add_argument('--method', '-m', 
                       choices=['reinhard', 'multiscale', 'bilateral', 'detail_preserve', 
                               'chroma_only', 'progressive', 'mkl', 'monge_kantorovitch',
                               'idt', 'iterative_distribution'],
                       default='reinhard',
                       help='Transfer method (default: reinhard). New: mkl (Monge-Kantorovitch), idt (Iterative Distribution)')
    parser.add_argument('--strength', '-s', type=float, default=1.0,
                       help='Transfer strength 0-1 (default: 1.0)')
    parser.add_argument('--iterations', type=int, default=20,
                       help='Number of iterations for IDT method (default: 20)')
    parser.add_argument('--temporal', '-t', action='store_true',
                       help='Apply temporal smoothing for videos')
    parser.add_argument('--compare', '-c', action='store_true',
                       help='Compare all methods (image only)')
    
    args = parser.parse_args()
    
    # Initializing pipeline with reference or director
    if args.reference:
        pipeline = DirectorStyleTransfer(reference_path=args.reference)
    else:
        pipeline = DirectorStyleTransfer(director_name=args.director)
    
    # Checking if input is video or image
    ext = Path(args.input).suffix.lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
    
    if args.compare:
        if is_video:
            print("Warning: --compare only works with images")
            if args.output:
                pipeline.process_video(args.input, args.output, 
                                      method=args.method, 
                                      temporal_smooth=args.temporal,
                                      strength=args.strength)
        else:
            pipeline.compare_methods(args.input)
    else:
        if not args.output:
            parser.error("--output is required unless --compare is used")
        
        if is_video:
            pipeline.process_video(args.input, args.output, 
                                  method=args.method,
                                  temporal_smooth=args.temporal,
                                  strength=args.strength)
        else:
            pipeline.process_image(args.input, args.output, 
                                  method=args.method,
                                  strength=args.strength)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Demo mode
        print("="*70)
        print("DIRECTOR COLOR STYLE TRANSFER - PROPER REINHARD IMPLEMENTATION")
        print("="*70)
        
        pipeline = DirectorStyleTransfer('Avatar')
        
        sample_frame = "Frames/Avatar/shot_100.jpg"
        if os.path.exists(sample_frame):
            print(f"\nRunning comparison on: {sample_frame}\n")
            pipeline.compare_methods(sample_frame, output_dir="Comparisons")
            
            print("\n" + "="*70)
            print("DEMO COMPLETE! Check the 'Comparisons' folder.")
            print("="*70)
            print("\nTo process your own video/image:")
            print("  python final_pipeline.py -i input.mp4 -o output.mp4 -m reinhard -t")
        else:
            print(f"\nSample frame not found: {sample_frame}")
            print("Usage: python final_pipeline.py -i your_video.mp4 -o output.mp4")
    else:
        main()
