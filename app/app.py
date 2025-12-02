"""
Streamlit Web App for Director Color Style Transfer
Interactive web interface for applying director color grading to videos and images
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import json
from PIL import Image
import io

from final_pipeline import DirectorStyleTransfer
from color_statistics import ColorStatisticsAnalyzer

# Page configuration
st.set_page_config(
    page_title="Director Color Style Transfer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .director-card {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎬 Director Color Style Transfer</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Transform your videos and images with the cinematic color grading of famous directors</p>', unsafe_allow_html=True)

# Initialize session state
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Get available directors
@st.cache_data
def get_available_directors():
    """Get list of directors with generated palettes"""
    palettes_dir = "Palettes"
    if not os.path.exists(palettes_dir):
        return []
    
    stats_files = Path(palettes_dir).glob("*_stats.json")
    directors = sorted([f.stem.replace('_stats', '') for f in stats_files])
    return directors

# Load director info
@st.cache_data
def load_director_info():
    """Load director statistics and descriptions"""
    directors_info = {
        'Avatar': {
            'description': 'Vibrant blues and greens, fantasy aesthetic',
            'style': 'Sci-Fi Fantasy',
            'colors': ['#0080FF', '#00D4AA', '#2ECC71']
        },
        'Barbie': {
            'description': 'Saturated pinks, playful and bright',
            'style': 'Pop Art',
            'colors': ['#FF69B4', '#FFB6C1', '#FF1493']
        },
        'Oppenheimer': {
            'description': 'Desaturated, dramatic, historical epic',
            'style': 'Historical Drama',
            'colors': ['#8B7355', '#A0826D', '#C9B037']
        },
        'WesAnderson': {
            'description': 'Pastel colors, symmetrical, whimsical',
            'style': 'Quirky Indie',
            'colors': ['#F4A896', '#F9D5A7', '#9DC3C1']
        },
        'ChristopherNolan': {
            'description': 'Cool tones, desaturated, epic scale',
            'style': 'Epic Thriller',
            'colors': ['#5D8AA8', '#8B9DC3', '#A7C7E7']
        },
        'DavidFincher': {
            'description': 'Dark, teal and orange, neo-noir',
            'style': 'Dark Thriller',
            'colors': ['#008080', '#FF8C00', '#2F4F4F']
        },
        'QuentinTarentino': {
            'description': 'Saturated, retro, bold colors',
            'style': 'Pulp Fiction',
            'colors': ['#DC143C', '#FFD700', '#FF4500']
        },
        'BongJoonHo': {
            'description': 'High contrast, dramatic, social commentary',
            'style': 'Social Drama',
            'colors': ['#1C1C1C', '#B22222', '#F0E68C']
        },
        'MartinScorcesse': {
            'description': 'Warm tones, gritty, classic cinema',
            'style': 'Classic Crime',
            'colors': ['#8B4513', '#CD853F', '#DAA520']
        }
    }
    return directors_info

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/video.png", width=80)
    st.markdown("## 🎨 Configuration")
    
    # File upload
    st.markdown("### 📁 Upload Media")
    
    # File type selector
    input_type = st.radio(
        "Input type",
        ["Image", "Video"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if input_type == "Image":
        accepted_types = ['jpg', 'jpeg', 'png', 'bmp', 'webp']
        help_text = "Supported: JPG, PNG, BMP, WEBP (Max 200MB)"
    else:
        accepted_types = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv']
        help_text = "Supported: MP4, AVI, MOV, MKV, WebM, FLV (Max 2GB)"
    
    uploaded_file = st.file_uploader(
        "Drop your file here or click to browse",
        type=accepted_types,
        help=help_text,
        accept_multiple_files=False,
        key="main_upload"
    )
    
    # Show file info if uploaded
    if uploaded_file:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb < 1:
            size_display = f"{uploaded_file.size / 1024:.1f} KB"
        else:
            size_display = f"{file_size_mb:.1f} MB"
        
        st.success(f"Loaded: {uploaded_file.name}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"Size: {size_display}")
        with col_b:
            st.caption(f"Type: {uploaded_file.type}")
    
    # Director selection
    st.markdown("### 🎬 Director Style")
    available_directors = get_available_directors()
    
    if not available_directors:
        st.error("No director palettes found! Run `python generate_stats_only.py` first.")
        st.stop()
    
    selected_director = st.selectbox(
        "Select director style",
        available_directors,
        format_func=lambda x: f"{x}",
        help="Choose a director's cinematic color grading style"
    )
    
    # Quick preview of selected director
    if selected_director:
        directors_info = load_director_info()
        if selected_director in directors_info:
            info = directors_info[selected_director]
            with st.expander(f"ℹ️ About {selected_director}", expanded=False):
                st.write(f"**Style:** {info['style']}")
                st.write(f"**Description:** {info['description']}")
                palette_html = "".join([
                    f'<div style="display:inline-block; width:40px; height:40px; background-color:{c}; margin:4px; border-radius:8px; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.2);"></div>' 
                    for c in info['colors']
                ])
                st.markdown("**Color Palette:**", unsafe_allow_html=True)
                st.markdown(palette_html, unsafe_allow_html=True)

    # Transfer method
    st.markdown("### 🎨 Transfer Method")
    transfer_method = st.selectbox(
        "Method",
        ['reinhard', 'chroma_only', 'multiscale', 'bilateral', 'detail_preserve', 
         'progressive', 'mkl', 'idt','histogram_spec',
          'palette_hard', 'palette_soft', 'palette_weighted'],
        format_func=lambda x: {
            'reinhard': 'Reinhard - Fast & Natural (Recommended)',
            'chroma_only': 'Chroma Only - Preserve Brightness',
            'multiscale': 'Multi-Scale - Enhanced Details',
            'bilateral': 'Bilateral - Edge Preserving',
            'detail_preserve': 'Detail Preserve - Best Quality',
            'progressive': 'Progressive - Iterative Refinement',
            'mkl': 'Monge-Kantorovitch - Geometric (Advanced)',
            'idt': 'Iterative Distribution - Optimal Transport (Advanced)',
            'histogram_spec': 'Histogram Matching - Exact Contrast Copy',
            'palette_hard': 'Palette Hard - Quantized/Posterized',
            'palette_soft': 'Palette Soft - Tinted Blend',
            'palette_weighted': 'Palette Weighted - Smooth Quantization'
        }.get(x, x),
        help="""
        Standard Methods:
        - Reinhard: Fast, natural (recommended for most)
        - Chroma Only: Color only, preserves brightness (good for Barbie style)
        - Multiscale: Enhanced details, 3-scale pyramid
        - Bilateral: Edge-preserving, no halos
        - Detail Preserve: Best quality (slower)
        - Progressive: Iterative refinement
        - Histogram Matching: Forces exact texture match (uses sample frame)
        
        Advanced Geometric Methods:
        - MKL: Matches covariance (correlated colors)
        - IDT: Matches full distribution shape (no Gaussian assumption)
        """
    )
    
    # Strength slider
    st.markdown("### 💪 Transfer Strength")
    strength = st.slider(
        "Strength",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower values = more subtle, Higher values = more dramatic"
    )
    
    # Advanced options for geometric methods
    if transfer_method in ['idt', 'iterative_distribution']:
        with st.expander("IDT Options"):
            idt_iterations = st.slider(
                "Iterations",
                min_value=5,
                max_value=50,
                value=20,
                step=5,
                help="More iterations = better convergence (slower)"
            )
    else:
        idt_iterations = 20
    
    # Video-specific options
    st.markdown("### ⚙️ Video Options")
    temporal_smooth = st.checkbox(
        "Temporal smoothing",
        value=True,
        help="Reduces flickering in videos (recommended)"
    )
    preserve_white = st.checkbox(
        "Preserve Whites",
        value=False,
        help="Prevents highlights from being tinted (Highlight Roll-off)"
    )
    
    preserve_skin = st.checkbox(
        "Preserve Skin Tones",
        value=False,
        help="Protects faces/skin from aggressive color grading (Uses AI or YCbCr)"
    )
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        max_resolution = st.selectbox(
            "Max processing resolution",
            [720, 1080, 1440, 2160],
            index=1,
            help="Higher resolution = better quality but slower"
        )
        
        output_fps = st.selectbox(
            "Output FPS (for videos)",
            [20, 24, 30, 60],
            index=1,
            help="Lower FPS = faster processing"
        )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Input Preview")
    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == 'image':
            # Display input image
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Input: {uploaded_file.name}")
            
            # Image info
            with st.expander("📊 Image Details"):
                st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} px")
                st.write(f"**Format:** {image.format}")
                st.write(f"**Mode:** {image.mode}")
                st.write(f"**Size:** {uploaded_file.size / (1024*1024):.2f} MB")
        else:
            # Display video info
            st.info(f"Video: {uploaded_file.name}")
            
            # Try to get video metadata
            temp_video_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
            with open(temp_video_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            cap = cv2.VideoCapture(temp_video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                
                # Display video stats
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.metric("Resolution", f"{width}x{height}")
                    st.metric("Duration", f"{duration/60:.1f} min")
                with col_v2:
                    st.metric("FPS", f"{fps:.0f}")
                    st.metric("Frames", f"{frame_count:,}")
                
                # Show first frame as preview
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="First Frame Preview")
                
                cap.release()
            
            # Clean up temp file
            uploaded_file.seek(0)  # Reset file pointer for later use
            try:
                os.remove(temp_video_path)
            except:
                pass
    else:
        st.markdown("""
        <div class="upload-section">
            <h3>👆 Upload a file to get started</h3>
            <p style="margin-top: 1rem;">📸 <strong>Images:</strong> JPG, PNG, BMP, WEBP</p>
            <p>🎬 <strong>Videos:</strong> MP4, AVI, MOV, MKV, WebM, FLV</p>
            <p style="margin-top: 1rem; color: #888;">Maximum size: 2GB</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### 📥 Output Preview")
    if st.session_state.processed_file:
        if st.session_state.processed_file.endswith(('.jpg', '.png')):
            output_image = Image.open(st.session_state.processed_file)
            st.image(output_image, caption="Processed Result")
            
            # Image comparison
            if uploaded_file and uploaded_file.type.split('/')[0] == 'image':
                with st.expander("🔍 Before/After Comparison"):
                    comp_col1, comp_col2 = st.columns(2)
                    with comp_col1:
                        original_img = Image.open(uploaded_file)
                        st.image(original_img, caption="Before")
                    with comp_col2:
                        st.image(output_image, caption="After")
            
            # Download button
            with open(st.session_state.processed_file, 'rb') as f:
                file_ext = st.session_state.processed_file.split('.')[-1]
                st.download_button(
                    label="Download Processed Image",
                    data=f,
                    file_name=f"styled_{selected_director}.{file_ext}",
                    mime=f"image/{file_ext}"
                )
        else:
            st.success("Video processed successfully!")
            
            # Video info
            if os.path.exists(st.session_state.processed_file):
                file_size = os.path.getsize(st.session_state.processed_file) / (1024*1024)
                st.info(f"Output: {os.path.basename(st.session_state.processed_file)} ({file_size:.1f} MB)")
                
                # Download button
                with open(st.session_state.processed_file, 'rb') as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f,
                        file_name=f"styled_{selected_director}.mp4",
                        mime="video/mp4"
                    )
                
                st.caption("Tip: Right-click the download button and 'Save link as...' for large files")
    else:
        st.markdown("""
        <div class="upload-section">
            <h3>⏳ Waiting for processing</h3>
            <p style="margin-top: 1rem;">1️⃣ Upload your media</p>
            <p>2️⃣ Select a director style</p>
            <p>3️⃣ Adjust settings</p>
            <p>4️⃣ Click "Apply Style Transfer"</p>
            <p style="margin-top: 1rem; color: #888;">Your processed result will appear here</p>
        </div>
        """, unsafe_allow_html=True)

# Director info display
if selected_director:
    st.markdown("---")
    st.markdown("### 🎬 Selected Director Style")
    
    directors_info = load_director_info()
    if selected_director in directors_info:
        info = directors_info[selected_director]
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"**{selected_director}**")
            st.write(info['description'])
        
        with col2:
            st.markdown("**Style**")
            st.write(info['style'])
        
        with col3:
            st.markdown("**Color Palette**")
            palette_html = "".join([f'<div style="display:inline-block; width:30px; height:30px; background-color:{c}; margin:2px; border-radius:5px;"></div>' for c in info['colors']])
            st.markdown(palette_html, unsafe_allow_html=True)

# Process button
st.markdown("---")
process_button = st.button(
    "Apply Style Transfer",
    type="primary",
    disabled=(uploaded_file is None)
)

if process_button and uploaded_file is not None:
    st.session_state.processing = True
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Save uploaded file
        file_type = uploaded_file.type.split('/')[0]
        file_ext = uploaded_file.name.split('.')[-1]
        input_path = os.path.join(temp_dir, f"input.{file_ext}")
        
        with open(input_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        # Set reference path to None, as it's no longer used
        reference_path = None
        
        # Initialize pipeline
        with st.spinner("Initializing pipeline..."):
            pipeline = DirectorStyleTransfer(director_name=selected_director)
        
        # Process
        # Process
        if file_type == 'image':
            # Process image
            output_path = os.path.join(temp_dir, "output.jpg")
            
            with st.spinner("Processing image..."):
                progress_bar = st.progress(0)
                
                # --- BETTER WAY TO READ IMAGE ---
                # 1. Reset pointer
                uploaded_file.seek(0)
                # 2. Read bytes
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                # 3. Decode directly from memory (Avoids path issues)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                # --------------------------------
                
                if image is None:
                    st.error("Error decoding image. Please try a different file.")
                    st.stop()

                kwargs = {'strength': strength}
                if transfer_method in ['idt', 'iterative_distribution']:
                    kwargs['iterations'] = idt_iterations
                
                processed = pipeline.process_frame(
                    image,
                    method=transfer_method,
                    preserve_white=preserve_white, # New parameter
                    preserve_skin=preserve_skin,   # New parameter
                    **kwargs
                )
                
                # Save
                cv2.imwrite(output_path, processed)
                progress_bar.progress(100)
            
            st.success("Image processed successfully!")
            st.session_state.processed_file = output_path
            
        else:
            # Process video
            output_path = os.path.join(temp_dir, "output.mp4")
            
            st.markdown("### Processing Video")
            
            # Create progress containers
            progress_container = st.container()
            
            with progress_container:
                status_text = st.empty()
                progress_bar = st.progress(0)
                time_text = st.empty()
                
                # Get video info
                cap = cv2.VideoCapture(input_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                status_text.write(f"Total frames: {total_frames} @ {fps:.0f} fps")
                
                # Process video (we'll capture output in a different way)
                import subprocess
                import sys

                script_content = f"""
import sys
sys.path.insert(0, r'{os.getcwd()}')
from final_pipeline import DirectorStyleTransfer

pipeline = DirectorStyleTransfer(director_name='{selected_director}')
pipeline.process_video(
    r'{input_path}',
    r'{output_path}',
    method='{transfer_method}',
    temporal_smooth={temporal_smooth},
    strength={strength},
    preserve_white={preserve_white},  # Passed from Streamlit
    preserve_skin={preserve_skin}     # Passed from Streamlit
)
""" 
#                 # Create a Python script to run the processing
#                 script_content = f"""
# import sys
# sys.path.insert(0, r'{os.getcwd()}')
# from final_pipeline import DirectorStyleTransfer

# pipeline = DirectorStyleTransfer(director_name='{selected_director}')
# pipeline.process_video(
#     r'{input_path}',
#     r'{output_path}',
#     method='{transfer_method}',
#     temporal_smooth={temporal_smooth},
#     strength={strength}
# )
# """
                script_path = os.path.join(temp_dir, 'process.py')
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                # Run processing
                process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                # Monitor progress
                for line in process.stdout:
                    if '%|' in line:
                        # Parse tqdm progress
                        try:
                            parts = line.split('%|')[0].strip().split()
                            if parts and parts[-1].replace('%', '').isdigit():
                                progress = int(parts[-1].replace('%', ''))
                                progress_bar.progress(progress)
                                status_text.write(f"Processing: {progress}%")
                        except:
                            pass
                    
                    time_text.write(line.strip())
                
                process.wait()
                
                if process.returncode == 0 and os.path.exists(output_path):
                    progress_bar.progress(100)
                    st.success("Video processed successfully!")
                    st.session_state.processed_file = output_path
                    st.balloons()
                else:
                    st.error("Processing failed. Check the error messages above.")
        
        st.session_state.processing = False
        st.rerun()
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.session_state.processing = False

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🎬 Director Color Style Transfer | Built with Streamlit & OpenCV</p>
    <p style="font-size: 0.9rem;">Transform your videos with cinematic color grading from famous directors</p>
</div>
""", unsafe_allow_html=True)

# Show examples
with st.expander("📚 Examples & Tips"):
    st.markdown("""
    ### 💡 Tips for Best Results
    
    **For Highly Saturated Styles (Barbie, Tarantino):**
    - Use strength 0.5-0.7
    - Try `chroma_only` method
    - Lower values for more natural results
    
    **For Desaturated Styles (Nolan, Oppenheimer):**
    - Use strength 0.7-1.0
    - `reinhard` method works best
    - Higher values for dramatic effect
    
    **For Best Performance:**
    - Use 24 FPS for long videos
    - Keep resolution at 1080p
    - Enable temporal smoothing for videos
    
    **Method Recommendations:**
    - **Fast & Natural:** `reinhard` (default)
    - **Color Only:** `chroma_only` (preserves brightness)
    - **Best Quality:** `detail_preserve` (slower)
    - **Edge Preservation:** `bilateral`
    """)

# Statistics display
if st.sidebar.checkbox("📊 Show Processing Stats", value=False):
    st.markdown("---")
    st.markdown("### 📊 System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Available Directors", len(available_directors))
    
    with col2:
        st.metric("Transfer Methods", 6)
    
    with col3:
        if selected_director:
            stats_file = f"Palettes/{selected_director}_stats.json"
            if os.path.exists(stats_file):
                with open(stats_file) as f:
                    stats = json.load(f)
                st.metric("Frames Analyzed", stats.get('n_frames', 'N/A'))
