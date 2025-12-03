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
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

from enhanced_color_transfer import EnhancedColorTransfer
from palette_matching import PaletteMatcher
from color_statistics import ColorStatisticsAnalyzer
from evaluation_metrics import ColorTransferEvaluator


class DirectorStyleTransfer:
    """
    Main pipeline for transferring director's color style using proper Reinhard method
    """
    def __init__(self, director_name, palettes_dir=None):
        """
        Initialize the pipeline with absolute path handling
        """
        self.director_name = director_name
        
        # 1. RESOLVE ABSOLUTE PATHS
        # If palettes_dir is not provided, find it relative to this script file
        if palettes_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.palettes_dir = os.path.join(base_dir, "Palettes")
        else:
            self.palettes_dir = palettes_dir
            
        print(f"Looking for palettes in: {self.palettes_dir}")  # Debug print
        
        # Initialize placeholders
        self.stats = None
        self.palette_matcher = None
        self._idt_target_pixels = None
        self._cached_target_image = None
        
        # Always load director data
        self._load_director_data()
        
        # Initialize transfer module
        self.color_transfer = EnhancedColorTransfer(self.stats)

        # Initialize MediaPipe Segmenter if available
        if HAS_MEDIAPIPE:
            self.mp_selfie = mp.solutions.selfie_segmentation
            self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)
        else:
            self.segmenter = None
        
        print(f"Initialized pipeline for {director_name}")

    def _load_director_data(self):
        """Load director's statistics AND palette"""
        # 1. Load Statistics
        stats_path = os.path.join(self.palettes_dir, f"{self.director_name}_stats.json")
        
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Statistics file not found: {stats_path}")
        
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)

        # 2. Load Palette (Required for Palette Matching)
        palette_path = os.path.join(self.palettes_dir, f"{self.director_name}_palette.npy")
        
        # Debugging: Print exactly what it's looking for
        print(f"Attempting to load palette from: {palette_path}")
        
        if os.path.exists(palette_path):
            try:
                palette_data = np.load(palette_path)
                # Initialize the matcher
                self.palette_matcher = PaletteMatcher(director_palette=palette_data)
                print(f"  - Palette loaded successfully: {len(palette_data)} colors")
            except Exception as e:
                print(f"   Error loading palette file: {e}")
                self.palette_matcher = None
        else:
            print(f"   Palette file missing at: {palette_path}")
            print(f"  (Palette-based methods like 'palette_hard' will fail)")
            self.palette_matcher = None   

    
    def _get_available_directors(self):
        """Get list of directors with generated palettes"""
        if not os.path.exists(self.palettes_dir):
            return []
        
        stats_files = Path(os.path.normpath(self.palettes_dir)).glob("*_stats.json")
        directors = [f.stem.replace('_stats', '') for f in stats_files]
        return sorted(directors)
    
    def _load_target_pixels_for_idt(self):
        """
        Load and sample pixels from director frames for IDT method
        Needed for Iterative Distribution Transfer
        """
        from glob import glob
        
        frames_dir = os.path.normpath(f"Frames/{self.director_name}")
        frame_paths = sorted(glob(os.path.join(frames_dir, "*.jpg")))
        
        if not frame_paths:
            raise ValueError(f"No frames found in {frames_dir} for IDT method")
        
        all_pixels = []
        
        # Sample pixels from multiple frames
        sample_size = min(20, len(frame_paths))
        sampled_paths = np.random.choice(frame_paths, sample_size, replace=False)
        
        for frame_path in sampled_paths:
            frame = cv2.imread(os.path.normpath(frame_path))
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
     

    def process_frame(self, frame, method='reinhard', preserve_white=True,preserve_skin=False, **kwargs):
        """
        Process a single frame
        
        Args:
            frame: Input BGR image
            method: Transfer method to use
            preserve_white: Whether to preserve pure whites (highlight roll-off)
            **kwargs: Additional parameters
            
        Returns:
            processed_frame: BGR image with director's style
        """
        # 1. Generate the stylized frame
        processed = None

        if method == 'histogram_spec' or method == 'histogram_matching':
            # We need a target IMAGE, not just stats.
            # Check if we have one cached, otherwise load it.
            if not hasattr(self, '_cached_target_image') or self._cached_target_image is None:
                print("Loading sample target image for histogram matching...")
                self._cached_target_image = self._load_sample_target_image()
                
            if self._cached_target_image is None:
                raise ValueError(f"Could not find any frames in Frames/{self.director_name} to use as a reference.")
            
            # Resize target to match source for histogram calc (optional but faster)
            target_resized = cv2.resize(self._cached_target_image, (frame.shape[1], frame.shape[0]))
            
            # Call the method from EnhancedColorTransfer
            processed = self.color_transfer.histogram_specification(frame, target_resized)

        elif method == 'palette_hard':
            if self.palette_matcher is None:
                raise ValueError(f"Palette not found for {self.director_name}. Run palette extraction first.")
            processed = self.palette_matcher.map_frame_colors_hard(frame)

        elif method == 'palette_soft':
            if self.palette_matcher is None:
                raise ValueError(f"Palette not found for {self.director_name}.")
            # Map 'strength' kwarg to blend_factor (default 0.5)
            blend = kwargs.get('strength', 0.5)
            processed = self.palette_matcher.map_frame_colors_soft(frame, blend_factor=blend)

        elif method == 'palette_weighted':
            if self.palette_matcher is None:
                raise ValueError(f"Palette not found for {self.director_name}.")
            # We can use k=3 as standard
            processed = self.palette_matcher.map_frame_colors_weighted(frame, k=3)

        elif method == 'reinhard':
            strength = kwargs.get('strength', 1.0)
            processed = self.color_transfer.transfer_lab_statistics(frame, self.stats, strength)
        
        elif method == 'multiscale':
            processed = self.color_transfer.multiscale_transfer(frame, self.stats)
        
        elif method == 'bilateral':
            processed = self.color_transfer.bilateral_color_transfer(frame, self.stats)
        
        elif method == 'detail_preserve':
            processed = self.color_transfer.detail_preserving_transfer(frame, self.stats)
        
        elif method == 'chroma_only':
            strength = kwargs.get('strength', 1.0)
            processed = self.color_transfer.chrominance_only_transfer(frame, self.stats, strength)
        
        elif method == 'progressive':
            iterations = kwargs.get('iterations', 3)
            processed = self.color_transfer.progressive_transfer(frame, self.stats, iterations)
        
        elif method == 'mkl' or method == 'monge_kantorovitch':
            strength = kwargs.get('strength', 1.0)
            processed = self.color_transfer.monge_kantorovitch_transfer(frame, self.stats, strength)
        
        elif method == 'idt' or method == 'iterative_distribution':
            # Load target pixels if not already cached
            if not hasattr(self, '_idt_target_pixels'):
                self._idt_target_pixels = self._load_target_pixels_for_idt()
            
            iterations = kwargs.get('iterations', 20)
            strength = kwargs.get('strength', 1.0)
            processed = self.color_transfer.iterative_distribution_transfer(
                frame, self._idt_target_pixels, iterations, strength
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # --- WHITE PRESERVATION ---
        if preserve_white and processed is not None:
            processed = self.color_transfer.protect_highlights_lab(processed, threshold=220)

        #SKIN PRESERVATION (AI FIRST, CLASSICAL FALLBACK)
        if preserve_skin and processed is not None:
            mask = None
            
            # AI Method (MediaPipe)
            if self.segmenter is not None:
                # _get_dl_skin_mask is defined in THIS class (DirectorStyleTransfer)
                mask = self._get_dl_skin_mask(frame)
            
            # Fallback to Classical Method (YCbCr)
            if mask is None:
                # create_skin_mask_ycbcr is in the color_transfer object
                mask = self.color_transfer.create_skin_mask_ycbcr(frame)
            
            # Apply Blending
            if mask is not None:
                # Convert images to float for blending
                original_float = frame.astype(np.float32)
                styled_float = processed.astype(np.float32)
                
                # Formula: Result = (Original * Mask) + (Styled * (1 - Mask))
                # Where Mask is 1.0 for skin, 0.0 for background
                result_float = (original_float * mask) + (styled_float * (1.0 - mask))
                
                processed = np.clip(result_float, 0, 255).astype(np.uint8)

        return processed

 
    def process_video(self, input_video_path, output_video_path, 
                      method='reinhard', temporal_smooth=True, 
                      evaluate=False, eval_output_dir='evaluation_results', 
                      eval_sample_frames=30, **kwargs):
        """
        Process entire video
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to save output video
            method: Transfer method
            temporal_smooth: Apply temporal smoothing
            evaluate: Whether to compute quality metrics
            eval_output_dir: Directory to save evaluation results
            eval_sample_frames: Number of frames to evaluate (for performance)
            **kwargs: Additional parameters
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING VIDEO: {input_video_path}")
        print(f"{'='*70}")
        print(f"Method: {method}")
        print(f"Temporal smoothing: {temporal_smooth}")
        if evaluate:
            print(f"Evaluation: Enabled (sampling {eval_sample_frames} frames)")
        
        # Initialize evaluator if needed
        evaluator = None
        eval_frame_indices = []
        source_frames_for_eval = []
        output_frames_for_eval = []
        
        if evaluate:
            evaluator = ColorTransferEvaluator()
            os.makedirs(eval_output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(os.path.normpath(input_video_path))
        
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

        if method in ['histogram_spec', 'histogram_matching']:
            print("Pre-loading representative target frame for Histogram Matching...")
            self._cached_target_image = self._load_sample_target_image()
            if self._cached_target_image is None:
                 raise ValueError(f"No frames found in Frames/{self.director_name}")
                 
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
        out = cv2.VideoWriter(os.path.normpath(temp_output), fourcc, fps, (width, height))
        
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
                
                # Skip frames if needed
                if frame_count % frame_skip != 0:
                    continue
                
                # Downscale for processing if needed
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
                    
                    # Process with fixed target and fewer iterations for consistency
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


                if use_fixed_idt_mapping and kwargs.get('preserve_white', True):
                     processed = self.color_transfer.protect_highlights_lab(processed, threshold=220)

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
                
                # Collect frames for evaluation if enabled
                if evaluate and evaluator is not None:
                    # Sample frames uniformly for evaluation
                    if frame_count in eval_frame_indices or len(eval_frame_indices) == 0:
                        if len(source_frames_for_eval) < eval_sample_frames:
                            source_frames_for_eval.append(frame_small.copy() if downscale else frame.copy())
                            output_frames_for_eval.append(processed.copy())
                    
                    # Set evaluation indices if not already set
                    if len(eval_frame_indices) == 0 and frame_count == 1:
                        eval_frame_indices = np.linspace(0, estimated_frames - 1, 
                                                        min(eval_sample_frames, estimated_frames), 
                                                        dtype=int).tolist()
                
                # Write frame
                out.write(processed)
                
                pbar.update(1)
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"\nVideo processing complete")
        
        # Evaluate quality metrics if enabled
        if evaluate and evaluator is not None and len(output_frames_for_eval) > 0:
            print("\n" + "="*70)
            print(f"EVALUATING COLOR TRANSFER QUALITY ({len(output_frames_for_eval)} frames)")
            print("="*70)
            
            # Load target frames
            target_frames = []
            frames_dir = f"Frames/{self.director_name}"
            
            if os.path.exists(frames_dir):
                all_target_frames = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]
                # Sample target frames
                sample_indices = np.linspace(0, len(all_target_frames) - 1, 
                                            min(len(output_frames_for_eval), len(all_target_frames)), 
                                            dtype=int)
                for idx in sample_indices:
                    target_path = os.path.normpath(os.path.join(frames_dir, all_target_frames[idx]))
                    target = cv2.imread(target_path)
                    if target is not None:
                        # Resize to match output
                        target = cv2.resize(target, 
                                          (output_frames_for_eval[0].shape[1], 
                                           output_frames_for_eval[0].shape[0]))
                        target_frames.append(target)
            
            if len(target_frames) > 0:
                # Evaluate each frame
                print(f"\nEvaluating {min(len(output_frames_for_eval), len(target_frames))} sample frames...")
                
                for i in range(min(len(output_frames_for_eval), len(target_frames))):
                    metrics = evaluator.evaluate_frame(
                        source_frames_for_eval[i],
                        output_frames_for_eval[i],
                        target_frames[i],
                        verbose=False  # Don't print each frame
                    )
                
                # Compute temporal stability
                if len(output_frames_for_eval) > 1:
                    evaluator.evaluate_temporal_stability(
                        output_frames_for_eval, 
                        verbose=True
                    )
                
                # Get summary
                summary = evaluator.get_summary()
                
                # Print key metrics
                print("\n" + "="*70)
                print("EVALUATION SUMMARY")
                print("="*70)
                print(f"\nColor Fidelity (Target Matching):")
                print(f"  EMD Average: {summary.get('EMD_Average_mean', 0):.4f} ± {summary.get('EMD_Average_std', 0):.4f}")
                print(f"  Histogram Intersection: {summary.get('Histogram_Intersection_mean', 0):.4f} ± {summary.get('Histogram_Intersection_std', 0):.4f}")
                print(f"\nStructure Preservation:")
                print(f"  SSIM: {summary.get('SSIM_mean', 0):.4f} ± {summary.get('SSIM_std', 0):.4f}")
                print(f"  PSNR: {summary.get('PSNR_mean', 0):.2f} ± {summary.get('PSNR_std', 0):.2f} dB")
                
                # Save results
                video_name = Path(output_video_path).stem
                evaluator.save_metrics(f"{eval_output_dir}/{video_name}_metrics.json")
                evaluator.plot_metrics(f"{eval_output_dir}/{video_name}_metrics.png")
                
                # Save summary
                with open(f"{eval_output_dir}/{video_name}_summary.json", 'w') as f:
                    json.dump(summary, f, indent=2)
                
                print(f"\nEvaluation results saved to {eval_output_dir}/")
                print("="*70)
            else:
                print("Warning: Could not load target frames for evaluation")
        
        # Merge audio from original video using ffmpeg
        print(f"\nMerging audio from original video...")
        import subprocess
        
        try:
            # Check if ffmpeg is available
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
                # Rename temp file to output
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
                     method='reinhard', evaluate=False, eval_output_dir='evaluation_results', **kwargs):
        """
        Process a single image
        
        Args:
            input_image_path: Path to input image
            output_image_path: Path to save output image
            method: Transfer method
            evaluate: Whether to compute quality metrics
            eval_output_dir: Directory to save evaluation results
            **kwargs: Additional parameters
        """
        # Read image
        image = cv2.imread(os.path.normpath(input_image_path))
        
        if image is None:
            raise ValueError(f"Cannot read image: {input_image_path}")
        
        print(f"Processing image: {input_image_path}")
        print(f"Method: {method}")
        
        # Process
        processed = self.process_frame(image, method=method, **kwargs)
        
        # Save
        cv2.imwrite(os.path.normpath(output_image_path), processed)
        
        print(f"Saved to: {output_image_path}")
        
        # Evaluate if requested
        if evaluate:
            print("\n" + "="*70)
            print("EVALUATING COLOR TRANSFER QUALITY")
            print("="*70)
            
            # Load target image (sample from director frames)
            target = self._load_sample_target_image()
            
            if target is not None:
                # Create evaluator
                evaluator = ColorTransferEvaluator()
                
                # Resize target to match processed size
                target_resized = cv2.resize(target, (processed.shape[1], processed.shape[0]))
                
                # Evaluate
                metrics = evaluator.evaluate_frame(image, processed, target_resized, verbose=True)
                
                # Save results
                os.makedirs(eval_output_dir, exist_ok=True)
                output_base = Path(output_image_path).stem
                
                evaluator.save_metrics(os.path.normpath(f"{eval_output_dir}/{output_base}_metrics.json"))
                evaluator.plot_metrics(os.path.normpath(f"{eval_output_dir}/{output_base}_metrics.png"))
                
                print(f"\nEvaluation results saved to {eval_output_dir}/")
            else:
                print("Warning: Could not load target image for evaluation")
        
        return processed

    def _get_dl_skin_mask(self, frame):
        """Get skin/person mask using MediaPipe (DL Method)"""
        if not HAS_MEDIAPIPE or self.segmenter is None:
            return None
            
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(rgb)
        mask = results.segmentation_mask
        
        # Expand to 3 channels
        return np.stack([mask] * 3, axis=2)
        
    def _load_sample_target_image(self):
        """Load a sample target image from director frames"""
        # Try to load from director frames
        frames_dir = os.path.normpath(f"Frames/{self.director_name}")
        if os.path.exists(frames_dir):
            frames = [f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))]
            if frames:
                # Middle frame
                sample_path = os.path.normpath(os.path.join(frames_dir, frames[len(frames)//2]))
                return cv2.imread(sample_path)
        return None
    
    def compare_methods(self, input_image_path, output_dir="Comparisons"):
        """
        Compare all transfer methods on a single image
        
        Args:
            input_image_path: Path to input image
            output_dir: Directory to save comparison
        """
        import matplotlib.pyplot as plt
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Read image
        image = cv2.imread(os.path.normpath(input_image_path))
        
        if image is None:
            raise ValueError(f"Cannot read image: {input_image_path}")
        
        print(f"\n{'='*70}")
        print(f"COMPARING METHODS ON: {input_image_path}")
        print(f"{'='*70}\n")
        
        # Process with different methods
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
        
        # Create comparison grid (3x3 for 9 methods)
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
        
        # Save comparison
        comparison_path = os.path.normpath(os.path.join(output_dir, f"{self.director_name}_final_comparison.png"))
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison saved to: {comparison_path}")
        
        # Save individual results
        for name, result in results.items():
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
            result_path = os.path.normpath(os.path.join(output_dir, f"{safe_name}.jpg"))
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
    parser.add_argument('--director', '-d', default='Avatar',
                       help='Director style to apply (e.g., Avatar, WesAnderson, ChristopherNolan, etc.)')
    parser.add_argument('--method', '-m', 
                       choices=['reinhard', 'multiscale', 'bilateral', 'detail_preserve', 
                               'chroma_only', 'progressive', 'mkl', 'monge_kantorovitch',
                               'idt', 'iterative_distribution', 'histogram_spec',
                               'palette_hard', 'palette_soft', 'palette_weighted'],
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
    parser.add_argument('--evaluate', '-e', action='store_true',
                       help='Evaluate quality metrics (EMD, KL, SSIM, etc.)')
    parser.add_argument('--eval-output', type=str, default='evaluation_results',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Initialize pipeline - simplified to only use director name
    pipeline = DirectorStyleTransfer(director_name=args.director)
    
    # Check if input is video or image
    input_path = os.path.normpath(args.input)
    ext = Path(input_path).suffix.lower()
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
    
    if args.compare:
        if is_video:
            print("Warning: --compare only works with images")
            if args.output:
                pipeline.process_video(input_path, os.path.normpath(args.output), 
                                      method=args.method, 
                                      temporal_smooth=args.temporal,
                                      strength=args.strength)
        else:
            pipeline.compare_methods(input_path)
    else:
        if not args.output:
            parser.error("--output is required unless --compare is used")
        output_path = os.path.normpath(args.output)
        if is_video:
            pipeline.process_video(input_path, output_path, 
                                  method=args.method,
                                  temporal_smooth=args.temporal,
                                  strength=args.strength,
                                  evaluate=args.evaluate,
                                  eval_output_dir=os.path.normpath(args.eval_output))
        else:
            pipeline.process_image(input_path, output_path, 
                                  method=args.method,
                                  strength=args.strength,
                                  evaluate=args.evaluate,
                                  eval_output_dir=os.path.normpath(args.eval_output))


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
