"""
COLMAP Process Helper Functions with Progress Bars

This module provides optimized COLMAP execution functions with proper timeout handling,
batch processing, enhanced GPU utilization, and progress bars using tqdm.
"""
# src/preprocessing/colmap_helpers.py
import os
import subprocess
import logging
import time
import threading
from pathlib import Path
import shutil
import pandas as pd
import math
import numpy as np
from tqdm import tqdm
import cv2

# Set up logging
logger = logging.getLogger(__name__)

# Constants
COLMAP_TIMEOUT = 1800  # 30 minutes timeout for COLMAP processes
MAX_IMAGES_PER_BATCH = 150  # Maximum images to process in a batch

def select_camera_model(image_metadata):
    """
    Select appropriate camera model based on image metadata
    
    Args:
        image_metadata: Dictionary with image metadata
        
    Returns:
        str: COLMAP camera model name
    """
    # Check for thermal images (IRX)
    if 'Filename' in image_metadata and 'IRX' in image_metadata['Filename']:
        # Thermal cameras often have different lens characteristics
        return 'SIMPLE_RADIAL'
    
    # Check camera model
    camera_model = image_metadata.get('CameraModel', '')
    if 'Autel' in camera_model and 'EVO' in camera_model:
        if 'ImageWidth' in image_metadata and image_metadata['ImageWidth'] > 3000:
            # High-res RGB camera
            return 'RADIAL'
        else:
            # Lower-res camera
            return 'SIMPLE_PINHOLE'
    
    # Check image dimensions
    if 'ImageWidth' in image_metadata and 'ImageHeight' in image_metadata:
        width = image_metadata['ImageWidth']
        height = image_metadata['ImageHeight']
        
        # For 4K or larger images
        if width >= 3840 or height >= 3840:
            return 'RADIAL'
        # For very wide aspect ratios (panoramic)
        elif width / height > 2.0 or height / width > 2.0:
            return 'RADIAL'
    
    # Default for most cases
    return 'SIMPLE_PINHOLE'

    
def has_colmap():
    """Check if COLMAP is available on the system with proper timeout handling"""
    try:
        process = subprocess.run(['colmap', '--help'], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL,
                               timeout=5)
        return process.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return False

def downsample_images_for_sfm(flight_group, max_images=MAX_IMAGES_PER_BATCH):
    """
    Downsample images for SfM processing to improve performance
    
    Args:
        flight_group: DataFrame with flight group images
        max_images: Maximum number of images to use
        
    Returns:
        DataFrame: Downsampled images for SfM processing
    """
    # Only use RGB images for SfM if we have that column
    if 'IsRGB' in flight_group.columns:
        rgb_images = flight_group[flight_group['IsRGB'] == True]
    else:
        # Otherwise, we'll just use all images (and rely on filename filtering later)
        rgb_images = flight_group
    
    if len(rgb_images) <= max_images:
        return rgb_images
    
    logger.info(f"Downsampling from {len(rgb_images)} to {max_images} images for SfM")
    
    # Try to preserve images with existing orientation data
    has_orientation = (
        rgb_images['Phi (Pitch)'].notna() & 
        rgb_images['Kappa (Yaw)'].notna() & 
        rgb_images['Omega (Roll)'].notna()
    )
    
    known_images = rgb_images[has_orientation]
    unknown_images = rgb_images[~has_orientation]
    
    if len(known_images) >= max_images:
        # If we have more known images than our limit, sample from them
        return known_images.sample(max_images)
    
    # Keep all known images, sample from unknown
    num_to_sample = max_images - len(known_images)
    
    # For orbit patterns, try to evenly sample the orbit
    if 'Timestamp' in unknown_images.columns and unknown_images['Timestamp'].notna().any():
        unknown_images_sorted = unknown_images.sort_values('Timestamp')
        step = max(1, len(unknown_images_sorted) // num_to_sample)
        sampled_unknown = unknown_images_sorted.iloc[::step][:num_to_sample]
    else:
        # Random sampling as fallback
        sampled_unknown = unknown_images.sample(min(num_to_sample, len(unknown_images)))
    
    # Combine known and sampled unknown images
    return pd.concat([known_images, sampled_unknown])

def create_colmap_config(config_path, use_gpu=True):
    """
    Create a COLMAP configuration file with optimized settings
    
    Args:
        config_path: Path to save the config file
        use_gpu: Whether to use GPU acceleration
    """
    config = {
        "SiftExtraction": {
            "use_gpu": 1 if use_gpu else 0,
            "gpu_index": "0",
            "estimate_affine_shape": 0,  # Faster extraction
            "domain_size_pooling": 1,
            "max_image_size": 1600,  # Limit image size for faster processing
            "max_num_features": 8192,  # Reasonable limit
            "first_octave": -1,  # Start at lower resolution for better coarse features
            "num_octaves": 4,
            "peak_threshold": 0.0066667 if stability_mode else 0.01,  # More stringent peak detection
            "edge_threshold": 10.0  # Default is 10

        },
        "SiftMatching": {
            "use_gpu": 1 if use_gpu else 0,
            "gpu_index": "0",
            "max_ratio": 0.8,  # Good balance between speed and match quality
            "max_distance": 0.7,
            "cross_check": 1,
            "multiple_models": 0 if stability_mode else 1,  # Disable multiple models for stability
            "guided_matching": 1 if stability_mode else 0,  # Use guided matching for better precision
        },
        "SequentialMatching": {
            "overlap": 20 if stability_mode else 10,  # Consider more neighboring images
            "quadratic_overlap": 0,  # Disable quadratic overlap
            "loop_detection": 1,  # Enable loop detection
            "vocab_tree_path": ""  # Leave empty unless you have a vocabulary tree
        },
        "VocabTreeMatching": {
            "num_images": 50,  # Match to more images for better connectivity
            "num_verifications": 30,
            "max_num_features": 8192
        },        
        "Mapper": {
            "ba_local_max_num_iterations": 50 if stability_mode else 25,
            "ba_local_max_refinements": 3,
            "ba_global_max_num_iterations": 100 if stability_mode else 50,
            "ba_global_max_refinements": 5 if stability_mode else 3,
            "ba_local_function_tolerance": 0.001,  # More relaxed tolerance
            "ba_local_gradient_tolerance": 0.001,  # More relaxed tolerance
            "ba_global_function_tolerance": 0.001,  # More relaxed tolerance
            "ba_global_gradient_tolerance": 0.001,  # More relaxed tolerance
            "ba_global_images_ratio": 1.1,  # Only run global BA at end
            "ba_global_points_ratio": 1.1,  # Only run global BA at end
            "ba_global_points_freq": 250000,
            "ba_local_max_num_constraints": 1000000,  # Increased from default
            "init_min_num_inliers": 15 if stability_mode else 30,  # Require fewer inliers
            "abs_pose_min_num_inliers": 15 if stability_mode else 30,
            "abs_pose_min_inlier_ratio": 0.25 if stability_mode else 0.5,  # More permissive ratio
            "filter_max_reproj_error": 8.0 if stability_mode else 4.0,  # More permissive error threshold
            "filter_min_tri_angle": 1.0,  # Minimum triangulation angle in degrees
            "max_reg_trials": 3,
            "min_num_matches": 10 if stability_mode else 15,  # Require fewer matches
            "init_max_error": 3.0 if stability_mode else 2.0  # More permissive initial error
        },
        "Triangulation": {
            "max_transitivity": 10,  # Include more transitively observed points
            "create_max_angle_error": 5.0,  # Maximum angular error in degrees
            "continue_max_angle_error": 5.0,
            "min_angle": 1.0,  # Minimum triangulation angle in degrees
            "ignore_two_view_tracks": 0  # Include two-view tracks
        }

        # "Mapper": {
        #     "ba_local_max_num_iterations": 25,  # Reduce iterations for speed
        #     "ba_global_max_num_iterations": 30,  # Reduce iterations for speed
        #     "min_num_matches": 10,  # Require fewer matches to consider images
        #     "max_num_models": 50,  # Reasonable limit\
        #     "ba_global_function_tolerance":0.01,
        #     "ba_global_max_refinements":5,
        #     "max_register_trials": 3,  # Reduce for speed
        #     "ba_global_images_ratio": 1.1,  # Only run global BA at the end
        #     "ba_global_points_ratio": 1.1,  # Only run global BA at the end
        #     "ba_local_max_refinements": 2  # Reduce for speed
        # }
    }
    
    with open(config_path, 'w') as f:
        for section, params in config.items():
            f.write(f"[{section}]\n")
            for key, value in params.items():
                f.write(f"{key}={value}\n")
            f.write("\n")

def run_colmap_process(cmd, timeout=COLMAP_TIMEOUT, desc=None):
    """
    Run a COLMAP process with proper timeout and monitoring with progress bar
    
    Args:
        cmd: COLMAP command to run
        timeout: Timeout in seconds
        desc: Description for the progress bar
        
    Returns:
        bool: Success status
    """
    logger.info(f"Running COLMAP command: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Use tqdm to show progress
        pbar = tqdm(total=100, desc=desc or "COLMAP process", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        last_progress = 0
        progress_patterns = {
            'Feature extraction': ['Extracted features for image', 'in'],
            'Matcher': ['Matched images', 'against'],
            'Mapper': ['Registering image', 'successful']
        }
        
        def log_output(stream, log_func, pbar, process_type):
            nonlocal last_progress
            for line in iter(stream.readline, ''):
                log_func(line.strip())
                
                # Try to extract progress information for the progress bar
                if process_type in progress_patterns:
                    patterns = progress_patterns[process_type]
                    if all(p in line for p in patterns):
                        try:
                            if 'Feature extraction' in process_type:
                                # Extract current image number and total
                                parts = line.split()
                                current = int(parts[parts.index('image') + 1])
                                total = int(parts[parts.index('of') + 1])
                                progress = min(int(current / total * 100), 100)
                            elif 'Matched images' in line:
                                # Extract matched pair count
                                parts = line.split()
                                current = int(parts[parts.index('Matched') + 2])
                                progress = min(last_progress + 1, 100)  # Increment by 1
                            elif 'Registering image' in line:
                                # Extract registered image count
                                parts = line.split()
                                if 'successful' in line:
                                    progress = min(last_progress + 2, 100)  # Increment by 2
                                else:
                                    progress = last_progress
                            
                            # Update progress bar
                            if progress > last_progress:
                                pbar.update(progress - last_progress)
                                last_progress = progress
                        except (ValueError, IndexError):
                            pass
                
        # Determine process type for progress monitoring
        process_type = "Unknown"
        if 'feature_extractor' in cmd:
            process_type = 'Feature extraction'
        elif 'matcher' in cmd:
            process_type = 'Matcher'
        elif 'mapper' in cmd:
            process_type = 'Mapper'
        
        # Start threads to log output in real-time
        stdout_thread = threading.Thread(
            target=log_output, 
            args=(process.stdout, logger.debug, pbar, process_type)
        )
        stderr_thread = threading.Thread(
            target=log_output, 
            args=(process.stderr, logger.warning, pbar, process_type)
        )
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete with timeout
        start_time = time.time()
        while process.poll() is None:
            if time.time() - start_time > timeout:
                logger.error(f"COLMAP process timed out after {timeout}s, terminating")
                process.terminate()
                try:
                    process.wait(timeout=30)  # Wait for graceful termination
                except subprocess.TimeoutExpired:
                    logger.error("Force killing COLMAP process")
                    process.kill()
                pbar.close()
                return False
            time.sleep(0.5)  # Check every half second
            
            # Update progress bar if we've received no updates (indeterminate progress)
            if time.time() - start_time > 10 and last_progress < 2:
                last_progress = 2
                pbar.update(2)  # Show some progress so it doesn't look stuck
        
        # Process completed
        pbar.update(100 - last_progress)  # Complete the progress bar
        pbar.close()
        
        if process.returncode != 0:
            logger.error(f"COLMAP process failed with code {process.returncode}")
            return False
            
        return True
    
    except Exception as e:
        logger.error(f"Error running COLMAP process: {e}")
        if 'pbar' in locals():
            pbar.close()
        return False

def quaternion_to_euler(qw, qx, qy, qz):
    """Convert quaternion to Euler angles (degrees)"""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    # Convert to degrees
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    roll = math.degrees(roll)
    
    # Map to expected format for drone orientation
    # Note: This mapping may need adjustment based on COLMAP vs. drone coordinate systems
    return pitch, yaw, roll

def run_colmap_sfm_with_enhanced_stability(flight_group, output_dir, use_gpu=True):
    """
    Enhanced COLMAP SfM implementation with numerical stability improvements
    and better fallback handling to address the Eigen failures
    
    Args:
        flight_group: DataFrame with flight group images
        output_dir: Directory for COLMAP output
        use_gpu: Whether to use GPU for COLMAP processing
        
    Returns:
        DataFrame: Updated with orientation data from COLMAP
    """
    # Check if COLMAP is available
    if not has_colmap():
        logger.error("COLMAP not found on this system. Please install COLMAP.")
        return flight_group
    
    logger.info(f"Running enhanced COLMAP SfM with stability improvements, outputting to {output_dir}")
    
    # Create directories
    colmap_dir = Path(output_dir)
    os.makedirs(colmap_dir, exist_ok=True)
    images_dir = colmap_dir / 'images'
    sparse_dir = colmap_dir / 'sparse'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Step 1: Image Selection and Preprocessing
    # --------------------------------------
    
    # Downsample images for SfM if needed
    sfm_images = downsample_images_for_sfm(flight_group)
    logger.info(f"Using {len(sfm_images)} images for SfM processing")
    
    # Enhanced image preprocessing
    preprocessed_dir = colmap_dir / 'preprocessed'
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Copy and enhance images for better feature detection
    copied_images = []
    
    with tqdm(total=len(sfm_images), desc="Preprocessing images", unit="img") as pbar:
        for idx, row in sfm_images.iterrows():
            # Only use RGB images for SfM
            if 'IsRGB' in flight_group.columns and not row['IsRGB']:
                pbar.update(1)
                continue
                
            img_source = row['SourceFile']
            
            # Skip missing files
            if not os.path.exists(img_source):
                logger.warning(f"Image file not found: {img_source}")
                pbar.update(1)
                continue
                
            filename = os.path.basename(img_source)
            img_dest = images_dir / filename
            
            try:
                # Apply image enhancement for better feature detection
                img = cv2.imread(img_source)
                if img is None:
                    logger.warning(f"Could not read image: {img_source}")
                    pbar.update(1)
                    continue
                
                # Resize image to limit memory usage
                max_dimension = 1600  # Reduced from 2000 to save memory
                h, w = img.shape[:2]
                if max(h, w) > max_dimension:
                    scale = max_dimension / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                
                # Apply CLAHE contrast enhancement to improve feature detection
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                
                # Save preprocessed image
                prep_path = preprocessed_dir / filename
                cv2.imwrite(str(prep_path), enhanced)
                
                # Create symbolic link to preprocessed image
                if os.path.exists(img_dest):
                    os.remove(img_dest)
                os.symlink(os.path.abspath(prep_path), img_dest)
                
                copied_images.append((idx, filename))
            except Exception as e:
                logger.error(f"Error preprocessing image {img_source}: {str(e)}")
                
                # Fallback to simple copy if preprocessing fails
                try:
                    if os.path.exists(img_dest):
                        os.remove(img_dest)
                    os.symlink(os.path.abspath(img_source), img_dest)
                    copied_images.append((idx, filename))
                except Exception as e2:
                    logger.error(f"Error creating symlink for {img_source}: {str(e2)}")
                
            pbar.update(1)
    
    logger.info(f"Preprocessed and copied {len(copied_images)} images for SfM processing")
    
    if len(copied_images) < 3:
        logger.warning("Not enough images for SfM. Need at least 3 images.")
        return flight_group
    
    # Step 2: Configure COLMAP with Stability Enhancements
    # --------------------------------------
    database_path = colmap_dir / 'database.db'
    config_path = colmap_dir / 'colmap_config.ini'
    
    # Create enhanced config with stability improvements
    with open(config_path, 'w') as f:
        # SiftExtraction section - more robust features but with memory constraints
        f.write("[SiftExtraction]\n")
        f.write(f"use_gpu={1 if use_gpu else 0}\n")
        f.write("gpu_index=0\n")
        f.write("estimate_affine_shape=0\n")
        f.write("domain_size_pooling=1\n")  # Improves feature robustness
        f.write("max_image_size=1600\n")  # Reduced to save memory
        f.write("max_num_features=4096\n")  # Reduced to save memory
        f.write("first_octave=-1\n")  # Start at lower resolution
        f.write("num_octaves=4\n")
        f.write("peak_threshold=0.0066667\n")  # More conservative threshold
        f.write("num_threads=4\n")  # Limit number of threads to avoid OOM
        f.write("\n")
        
        # SiftMatching section - improved matching
        f.write("[SiftMatching]\n")
        f.write(f"use_gpu={1 if use_gpu else 0}\n")
        f.write("gpu_index=0\n")
        f.write("max_ratio=0.8\n")
        f.write("max_distance=0.7\n")
        f.write("cross_check=1\n")
        f.write("guided_matching=1\n")  # More precise matches
        f.write("num_threads=4\n")  # Limit number of threads to avoid OOM
        f.write("\n")
        
        # SequentialMatching section - better connectivity
        f.write("[SequentialMatching]\n")
        f.write("overlap=15\n")  # Consider more neighbors
        f.write("quadratic_overlap=0\n")
        f.write("loop_detection=1\n")  # Enable loop closure
        f.write("\n")
        
        # Mapper section - improved stability
        f.write("[Mapper]\n")
        f.write("ba_global_max_num_iterations=75\n")  # More iterations
        f.write("ba_local_max_num_iterations=30\n")  # More iterations
        f.write("ba_global_max_refinements=5\n")  # More refinements
        f.write("min_num_matches=10\n")  # Fewer required matches
        f.write("ba_global_function_tolerance=0.001\n")  # Relaxed tolerance
        f.write("filter_max_reproj_error=8.0\n")  # More permissive threshold
        f.write("init_min_num_inliers=15\n")  # Fewer required inliers
        f.write("abs_pose_min_num_inliers=15\n")  # Fewer required inliers
        f.write("abs_pose_min_inlier_ratio=0.25\n")  # More permissive ratio
        # f.write("triangulation_method=1\n")  # DLT with checks
        f.write("num_threads=4\n")  # Limit number of threads to avoid OOM
        f.write("\n")
    
    # Clear any existing database
    if database_path.exists():
        os.remove(database_path)
    
    # Step 3: Run COLMAP Pipeline with Enhanced Error Handling
    # --------------------------------------
    
    # 1. Feature extraction with memory-conscious parameters
    logger.info("Running COLMAP feature extraction with enhanced stability parameters...")
    
    feature_extractor_cmd = [
        'colmap', 'feature_extractor',
        '--database_path', str(database_path),
        '--image_path', str(images_dir),
        '--SiftExtraction.use_gpu', '1' if use_gpu else '0',
        '--SiftExtraction.domain_size_pooling', '1',
        '--SiftExtraction.first_octave', '-1',
        '--SiftExtraction.peak_threshold', '0.0066667',
        '--SiftExtraction.max_image_size', '1600',  # Reduced to save memory
        '--SiftExtraction.max_num_features', '4096',  # Reduced to save memory
        '--SiftExtraction.num_threads', '4'  # Limit threads to avoid OOM
    ]
    
    if not run_colmap_process(feature_extractor_cmd, desc="Feature extraction"):
        logger.error("Feature extraction failed, aborting SfM")
        return flight_group
    
    # 2. Feature matching with multiple strategies
    logger.info("Running COLMAP feature matching...")
    
    # Try different matching strategies based on dataset size
    if len(copied_images) < 30:
        # For small datasets, use exhaustive matching
        logger.info("Using exhaustive matching for small dataset...")
        matcher_cmd = [
            'colmap', 'exhaustive_matcher',
            '--database_path', str(database_path),
            '--SiftMatching.use_gpu', '1' if use_gpu else '0',
            '--SiftMatching.guided_matching', '1',
            '--SiftMatching.num_threads', '4'  # Limit threads to avoid OOM
        ]
        
        if not run_colmap_process(matcher_cmd, desc="Exhaustive matching"):
            logger.warning("Exhaustive matching failed, falling back to sequential matching...")
            matcher_cmd = [
                'colmap', 'sequential_matcher',
                '--database_path', str(database_path),
                '--SiftMatching.use_gpu', '1' if use_gpu else '0',
                '--SiftMatching.guided_matching', '1',
                '--SequentialMatching.overlap', '15',
                '--SiftMatching.num_threads', '4'  # Limit threads to avoid OOM
            ]
            
            if not run_colmap_process(matcher_cmd, desc="Sequential matching"):
                logger.error("All matching strategies failed, aborting SfM")
                return flight_group
    else:
        # For larger datasets, start with sequential matching
        matcher_cmd = [
            'colmap', 'sequential_matcher',
            '--database_path', str(database_path),
            '--SiftMatching.use_gpu', '1' if use_gpu else '0',
            '--SiftMatching.guided_matching', '1',
            '--SequentialMatching.overlap', '15',
            '--SiftMatching.num_threads', '4'  # Limit threads to avoid OOM
        ]
        
        if not run_colmap_process(matcher_cmd, desc="Sequential matching"):
            logger.warning("Sequential matching failed, falling back to spatial matching...")
            
            # Try spatial matching as fallback
            matcher_cmd = [
                'colmap', 'spatial_matcher',
                '--database_path', str(database_path),
                '--SiftMatching.use_gpu', '1' if use_gpu else '0',
                '--SiftMatching.guided_matching', '1',
                '--SpatialMatching.max_num_neighbors', '50',
                '--SiftMatching.num_threads', '4'  # Limit threads to avoid OOM
            ]
            
            if not run_colmap_process(matcher_cmd, desc="Spatial matching"):
                logger.error("All matching strategies failed, aborting SfM")
                return flight_group
    
    # 3. Incremental SfM with enhanced stability parameters
    logger.info("Running COLMAP mapper with enhanced stability parameters...")
    
    # Remove parameters that aren't supported by your COLMAP version
    mapper_cmd = [
        'colmap', 'mapper',
        '--database_path', str(database_path),
        '--image_path', str(images_dir),
        '--output_path', str(sparse_dir),
        '--Mapper.ba_global_function_tolerance', '0.001',
        '--Mapper.ba_global_max_num_iterations', '75',
        '--Mapper.ba_local_max_num_iterations', '30',
        '--Mapper.ba_global_max_refinements', '5',
        '--Mapper.min_num_matches', '10',
        '--Mapper.filter_max_reproj_error', '8.0',
        '--Mapper.init_min_num_inliers', '15',
        '--Mapper.abs_pose_min_num_inliers', '15',
        '--Mapper.abs_pose_min_inlier_ratio', '0.25',
        # '--Mapper.triangulation_method', '1',
        '--Mapper.num_threads', '4'  # Limit threads to avoid OOM
    ]
    
    if not run_colmap_process(mapper_cmd, timeout=COLMAP_TIMEOUT * 2, desc="Mapping"):
        logger.warning("Mapper failed with enhanced parameters, trying with alternative configuration...")
        
        # Try with different initialization strategy
        mapper_cmd = [
            'colmap', 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(images_dir),
            '--output_path', str(sparse_dir),
            '--Mapper.init_min_num_inliers', '10',  # Even fewer inliers
            '--Mapper.abs_pose_min_num_inliers', '10',
            '--Mapper.abs_pose_min_inlier_ratio', '0.2',  # More permissive
            '--Mapper.ba_refine_focal_length', '0',  # Fix focal length
            '--Mapper.ba_refine_principal_point', '0',  # Fix principal point
            '--Mapper.ba_refine_extra_params', '0',  # Fix distortion
            '--Mapper.min_focal_length_ratio', '0.1',  # More permissive
            '--Mapper.max_focal_length_ratio', '10',  # More permissive
            '--Mapper.max_reg_trials', '5',  # More registration attempts
            '--Mapper.ba_global_images_ratio', '1.3',  # More frequent global BA
            '--Mapper.ba_global_points_ratio', '1.3',  # More frequent global BA
            '--Mapper.num_threads', '4'  # Limit threads to avoid OOM
        ]
        
        if not run_colmap_process(mapper_cmd, desc="Mapping (alternative)"):
            # If all else fails, try with most permissive settings
            logger.warning("Alternative mapping failed, trying with most permissive settings...")
            
            mapper_cmd = [
                'colmap', 'mapper',
                '--database_path', str(database_path),
                '--image_path', str(images_dir),
                '--output_path', str(sparse_dir),
                '--Mapper.init_min_num_inliers', '8',  # Bare minimum
                '--Mapper.abs_pose_min_num_inliers', '8',  # Bare minimum
                '--Mapper.abs_pose_min_inlier_ratio', '0.15',  # Very permissive
                '--Mapper.min_num_matches', '5',  # Minimum matches
                '--Mapper.filter_max_reproj_error', '12.0',  # Very permissive
                '--Mapper.num_threads', '4'  # Limit threads to avoid OOM
            ]
            
            if not run_colmap_process(mapper_cmd, desc="Mapping (permissive)"):
                logger.error("All mapping attempts failed, aborting SfM")
                return flight_group
    
    # Step 4: Extract and Process Results
    # --------------------------------------
    
    # Find reconstruction directories
    reconstruction_dirs = []
    for item in os.listdir(sparse_dir):
        if item.isdigit():
            recon_dir = sparse_dir / item
            if os.path.isdir(recon_dir):
                reconstruction_dirs.append(item)
        else:
            # Some COLMAP versions put files directly in sparse_dir
            if item == 'images.txt':
                reconstruction_dirs = ['']  # Use sparse_dir directly
                break

    if not reconstruction_dirs:
        logger.error("No COLMAP reconstruction found")
        return flight_group

    # Load the reconstruction
    if reconstruction_dirs[0] == '':
        # Images.txt is directly in sparse_dir
        images_file = sparse_dir / 'images.txt'
        logger.info("Using reconstruction in base sparse directory")
    else:
        # Use the largest reconstruction
        largest_recon = max(reconstruction_dirs, key=lambda d: os.path.getsize(sparse_dir / d))
        logger.info(f"Using reconstruction {largest_recon}")
        images_file = sparse_dir / largest_recon / 'images.txt'
    
    # Images.txt contains camera poses
    orientations = {}
    
    if not images_file.exists():
        logger.error(f"COLMAP images file not found: {images_file}")
        return flight_group
    
    # Parse images.txt to get orientations
    with open(images_file, 'r') as f:
        lines = f.readlines()
    
    for i in range(0, len(lines), 2):
        if i+1 >= len(lines) or lines[i].startswith('#'):
            continue
            
        # Parse image line
        parts = lines[i].strip().split()
        
        # Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        if len(parts) < 10:
            continue
            
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        image_name = parts[9]
        
        # Convert quaternion to Euler angles (pitch, yaw, roll)
        pitch, yaw, roll = quaternion_to_euler(qw, qx, qy, qz)
        
        # Store orientations
        orientations[image_name] = {
            'Phi (Pitch)': pitch,
            'Kappa (Yaw)': yaw,
            'Omega (Roll)': roll,
            'TX': tx,
            'TY': ty,
            'TZ': tz
        }
    
    # Update flight group with COLMAP orientations
    logger.info(f"Found orientation data for {len(orientations)} images")
    
    orientations_set = 0
    updated_group = flight_group.copy()
    
    with tqdm(total=len(copied_images), desc="Updating orientation data", unit="img") as pbar:
        for idx, filename in copied_images:
            if filename in orientations:
                # Update orientation data
                updated_group.at[idx, 'Phi (Pitch)'] = orientations[filename]['Phi (Pitch)']
                updated_group.at[idx, 'Kappa (Yaw)'] = orientations[filename]['Kappa (Yaw)']
                updated_group.at[idx, 'Omega (Roll)'] = orientations[filename]['Omega (Roll)']
                updated_group.at[idx, 'TX'] = orientations[filename]['TX']
                updated_group.at[idx, 'TY'] = orientations[filename]['TY']
                updated_group.at[idx, 'TZ'] = orientations[filename]['TZ']
                updated_group.at[idx, 'Orientation_Source'] = 'Enhanced_COLMAP_SfM'
                updated_group.at[idx, 'Orientation_Confidence'] = 'High'
                orientations_set += 1
            pbar.update(1)
    
    logger.info(f"Updated orientation data for {orientations_set} images")
    
    return updated_group


def run_colmap_sfm_optimized(flight_group, output_dir, use_gpu=True, enhanced_stability=True):
    """
    Run COLMAP SfM with optimized parameters and error handling
    
    This function is a drop-in replacement for the original run_colmap_sfm function
    with added timeout handling, better GPU utilization, and optimized parameters.
    
    Args:
        flight_group: DataFrame with flight group images
        output_dir: Directory for COLMAP output
        use_gpu: Whether to use GPU for COLMAP processing
        
    Returns:
        DataFrame: Updated with orientation data from COLMAP
    """
    if enhanced_stability:
        # Use the enhanced version with stability improvements
        return run_colmap_sfm_with_enhanced_stability(flight_group, output_dir, use_gpu)
    else:
        # Check if COLMAP is available
        if not has_colmap():
            logger.error("COLMAP not found on this system. Please install COLMAP.")
            return flight_group
        
        logger.info(f"Running optimized COLMAP SfM, outputting to {output_dir}")
        
        # Create directories
        colmap_dir = Path(output_dir)
        images_dir = colmap_dir / 'images'
        sparse_dir = colmap_dir / 'sparse'
        
        os.makedirs(colmap_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(sparse_dir, exist_ok=True)
        
        # Downsample images for SfM if needed
        sfm_images = downsample_images_for_sfm(flight_group)
        logger.info(f"Using {len(sfm_images)} images for SfM processing")
        
        # Copy images to COLMAP directory with progress bar
        copied_images = []
            
        with tqdm(total=len(sfm_images), desc="Copying images", unit="img") as pbar:
            logger.info(f"Checking if destination directory {images_dir} is writable: {os.access(images_dir, os.W_OK)}")
            for idx, row in sfm_images.iterrows():
                # Only use RGB images for SfM
                if 'IsRGB' in flight_group.columns and not row['IsRGB']:
                    pbar.update(1)
                    continue
                    
                img_source = row['SourceFile']
                
                # Skip missing files
                if not os.path.exists(img_source):
                    logger.warning(f"Image file not found: {img_source}")
                    pbar.update(1)
                    continue
                    
                # Create symlink to COLMAP images directory
                filename = os.path.basename(img_source)
                img_dest = images_dir / filename
                
                logger.info(f"Creating symlink for: {img_source}")
                
                try:
                    # Remove existing file/link if it exists
                    if os.path.exists(img_dest):
                        os.remove(img_dest)
                        
                    # Create symbolic link
                    os.symlink(os.path.abspath(img_source), img_dest)
                    copied_images.append((idx, filename))
                    logger.info(f"Symlink created successfully for {filename}")
                except Exception as e:
                    logger.error(f"Error creating symlink for {img_source}: {str(e)}")
                
                pbar.update(1)

            
        logger.info(f"Copied {len(copied_images)} images for SfM processing")
        
        if len(copied_images) < 3:
            logger.warning("Not enough images for SfM. Need at least 3 images.")
            return flight_group
        
        # COLMAP database path
        database_path = colmap_dir / 'database.db'
        
        # Create optimized COLMAP config
        config_path = colmap_dir / 'colmap_config.ini'
        create_colmap_config(config_path, use_gpu)
        
        # Clear any existing database
        if database_path.exists():
            os.remove(database_path)
        
        # 1. Feature extraction
        logger.info("Running COLMAP feature extraction...")
        
        feature_extractor_cmd = [
            'colmap', 'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', str(images_dir),
            '--SiftExtraction.use_gpu', '1' if use_gpu else '0',
            '--SiftExtraction.max_image_size', '1600',
            '--SiftExtraction.max_num_features', '8192'
        ]
        
        if not run_colmap_process(feature_extractor_cmd, desc="Feature extraction"):
            logger.error("Feature extraction failed, aborting SfM")
            return flight_group
        
        # 2. Feature matching
        logger.info("Running COLMAP sequential matching...")
        
        matcher_cmd = [
            'colmap', 'sequential_matcher',
            '--database_path', str(database_path),
            '--SiftMatching.use_gpu', '1' if use_gpu else '0',
            '--SequentialMatching.overlap', '10',
            '--SequentialMatching.quadratic_overlap', '0'
        ]
        
        if not run_colmap_process(matcher_cmd, desc="Feature matching"):
            logger.error("Feature matching failed, aborting SfM")
            return flight_group
        
        # 3. Incremental Structure-from-Motion
        logger.info("Running COLMAP mapper...")
        
        mapper_cmd = [
            'colmap', 'mapper',
            '--database_path', str(database_path),
            '--image_path', str(images_dir),
            '--output_path', str(sparse_dir),
            '--Mapper.ba_global_max_num_iterations', '50',
            '--Mapper.ba_local_max_num_iterations', '25',
            '--Mapper.min_num_matches', '10'
        ]
        
        if not run_colmap_process(mapper_cmd, timeout=COLMAP_TIMEOUT * 2, desc="Mapping"):  # Double timeout for mapper
            logger.error("Mapper failed, trying with more restrictive parameters")
            
            # Try again with more restrictive parameters
            mapper_cmd.extend([
                '--Mapper.ba_global_max_num_iterations', '25',
                '--Mapper.ba_local_max_num_iterations', '10',
                '--Mapper.max_num_models', '25',
                '--Mapper.min_model_size', '5'
            ])
            
            if not run_colmap_process(mapper_cmd, desc="Mapping (restrictive)"):
                logger.error("Mapper failed again, aborting SfM")
                return flight_group
        
        # 4. Extract camera orientations from COLMAP output
        logger.info("Extracting orientation data from COLMAP results...")
        
        # Find the largest reconstruction
        reconstruction_dirs = [d for d in os.listdir(sparse_dir) if d.isdigit()]
        
        if not reconstruction_dirs:
            logger.error("No COLMAP reconstruction found")
            return flight_group
        
        # Use the largest reconstruction
        largest_recon = max(reconstruction_dirs, key=lambda d: os.path.getsize(sparse_dir / d))
        logger.info(f"Using reconstruction {largest_recon}")
        
        # Parse COLMAP output files
        images_file = sparse_dir / largest_recon / 'images.txt'
        
        # Images.txt contains camera poses
        orientations = {}
        
        if not images_file.exists():
            logger.error(f"COLMAP images file not found: {images_file}")
            return flight_group
        
        # Parse images.txt to get orientations
        with open(images_file, 'r') as f:
            lines = f.readlines()
        
        for i in range(0, len(lines), 2):
            if i+1 >= len(lines) or lines[i].startswith('#'):
                continue
                
            # Parse image line
            parts = lines[i].strip().split()
            
            # Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            if len(parts) < 10:
                continue
                
            image_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            image_name = parts[9]
            
            # Convert quaternion to Euler angles (pitch, yaw, roll)
            pitch, yaw, roll = quaternion_to_euler(qw, qx, qy, qz)
            
            # Store orientations
            orientations[image_name] = {
                'Phi (Pitch)': pitch,
                'Kappa (Yaw)': yaw,
                'Omega (Roll)': roll,
                'TX': tx,
                'TY': ty,
                'TZ': tz
            }
        
        # Update flight group with COLMAP orientations
        logger.info(f"Found orientation data for {len(orientations)} images")
        
        orientations_set = 0
        updated_group = flight_group.copy()
        
        with tqdm(total=len(copied_images), desc="Updating orientation data", unit="img") as pbar:
            for idx, filename in copied_images:
                if filename in orientations:
                    # Update orientation data
                    updated_group.at[idx, 'Phi (Pitch)'] = orientations[filename]['Phi (Pitch)']
                    updated_group.at[idx, 'Kappa (Yaw)'] = orientations[filename]['Kappa (Yaw)']
                    updated_group.at[idx, 'Omega (Roll)'] = orientations[filename]['Omega (Roll)']
                    updated_group.at[idx, 'TX'] = orientations[filename]['TX']
                    updated_group.at[idx, 'TY'] = orientations[filename]['TY']
                    updated_group.at[idx, 'TZ'] = orientations[filename]['TZ']
                    updated_group.at[idx, 'Orientation_Source'] = 'COLMAP_SfM'
                    orientations_set += 1
                pbar.update(1)
        
        logger.info(f"Updated orientation data for {orientations_set} images")
        
        return updated_group

# def run_colmap_sfm_optimized(flight_group, output_dir, use_gpu=True):
#     """
#     Run COLMAP SfM with optimized parameters and error handling
    
#     This function is a drop-in replacement for the original run_colmap_sfm function
#     with added timeout handling, better GPU utilization, and optimized parameters.
    
#     Args:
#         flight_group: DataFrame with flight group images
#         output_dir: Directory for COLMAP output
#         use_gpu: Whether to use GPU for COLMAP processing
        
#     Returns:
#         DataFrame: Updated with orientation data from COLMAP
#     """
#     # Check if COLMAP is available
#     if not has_colmap():
#         logger.error("COLMAP not found on this system. Please install COLMAP.")
#         return flight_group
    
#     logger.info(f"Running optimized COLMAP SfM, outputting to {output_dir}")
    
#     # Create directories
#     colmap_dir = Path(output_dir)
#     os.makedirs(colmap_dir, exist_ok=True)
#     images_dir = colmap_dir / 'images'
#     sparse_dir = colmap_dir / 'sparse'
#     os.makedirs(images_dir, exist_ok=True)
#     os.makedirs(sparse_dir, exist_ok=True)
    
#     # Downsample images for SfM if needed
#     sfm_images = downsample_images_for_sfm(flight_group)
#     logger.info(f"Using {len(sfm_images)} images for SfM processing")
    
#     # Copy images to COLMAP directory with progress bar
#     copied_images = []
        
#     with tqdm(total=len(sfm_images), desc="Copying images", unit="img") as pbar:
#         logger.info(f"Checking if destination directory {images_dir} is writable: {os.access(images_dir, os.W_OK)}")
#         for idx, row in sfm_images.iterrows():
#             # Only use RGB images for SfM
#             if 'IsRGB' in flight_group.columns and not row['IsRGB']:
#                 pbar.update(1)
#                 continue
                
#             img_source = row['SourceFile']
            
#             # Skip missing files
#             if not os.path.exists(img_source):
#                 logger.warning(f"Image file not found: {img_source}")
#                 pbar.update(1)
#                 continue
                
#             # Create symlink to COLMAP images directory
#             filename = os.path.basename(img_source)
#             img_dest = images_dir / filename
            
#             logger.info(f"Creating symlink for: {img_source}")
            
#             try:
#                 # Remove existing file/link if it exists
#                 if os.path.exists(img_dest):
#                     os.remove(img_dest)
                    
#                 # Create symbolic link
#                 os.symlink(os.path.abspath(img_source), img_dest)
#                 copied_images.append((idx, filename))
#                 logger.info(f"Symlink created successfully for {filename}")
#             except Exception as e:
#                 logger.error(f"Error creating symlink for {img_source}: {str(e)}")
            
#             pbar.update(1)

        
#     logger.info(f"Copied {len(copied_images)} images for SfM processing")
    
#     if len(copied_images) < 3:
#         logger.warning("Not enough images for SfM. Need at least 3 images.")
#         return flight_group
    
#     # COLMAP database path
#     database_path = colmap_dir / 'database.db'
    
#     # Create optimized COLMAP config
#     config_path = colmap_dir / 'colmap_config.ini'
#     create_colmap_config(config_path, use_gpu)
    
#     # Clear any existing database
#     if database_path.exists():
#         os.remove(database_path)
    
#     # 1. Feature extraction
#     logger.info("Running COLMAP feature extraction...")
    
#     feature_extractor_cmd = [
#         'colmap', 'feature_extractor',
#         '--database_path', str(database_path),
#         '--image_path', str(images_dir),
#         '--SiftExtraction.use_gpu', '1' if use_gpu else '0',
#         '--SiftExtraction.max_image_size', '2000',
#         '--SiftExtraction.max_num_features', '8192'
#     ]
    
#     if not run_colmap_process(feature_extractor_cmd, desc="Feature extraction"):
#         logger.error("Feature extraction failed, aborting SfM")
#         return flight_group
    
#     # 2. Feature matching
#     logger.info("Running COLMAP sequential matching...")
    
#     matcher_cmd = [
#         'colmap', 'sequential_matcher',
#         '--database_path', str(database_path),
#         '--SiftMatching.use_gpu', '1' if use_gpu else '0',
#         '--SequentialMatching.overlap', '10',
#         '--SequentialMatching.quadratic_overlap', '0'
#     ]
    
#     if not run_colmap_process(matcher_cmd, desc="Feature matching"):
#         logger.error("Feature matching failed, aborting SfM")
#         return flight_group
    
#     # 3. Incremental Structure-from-Motion
#     logger.info("Running COLMAP mapper...")
    
#     mapper_cmd = [
#         'colmap', 'mapper',
#         '--database_path', str(database_path),
#         '--image_path', str(images_dir),
#         '--output_path', str(sparse_dir),
#         '--Mapper.ba_global_max_num_iterations', '50',
#         '--Mapper.ba_local_max_num_iterations', '25',
#         '--Mapper.min_num_matches', '10'
#     ]
    
#     if not run_colmap_process(mapper_cmd, timeout=COLMAP_TIMEOUT * 2, desc="Mapping"):  # Double timeout for mapper
#         logger.error("Mapper failed, trying with more restrictive parameters")
        
#         # Try again with more restrictive parameters
#         mapper_cmd.extend([
#             '--Mapper.ba_global_max_num_iterations', '25',
#             '--Mapper.ba_local_max_num_iterations', '10',
#             '--Mapper.max_num_models', '25',
#             '--Mapper.min_model_size', '5'
#         ])
        
#         if not run_colmap_process(mapper_cmd, desc="Mapping (restrictive)"):
#             logger.error("Mapper failed again, aborting SfM")
#             return flight_group
    
#     # 4. Extract camera orientations from COLMAP output
#     logger.info("Extracting orientation data from COLMAP results...")
    
#     # Find the largest reconstruction
#     reconstruction_dirs = [d for d in os.listdir(sparse_dir) if d.isdigit()]
    
#     if not reconstruction_dirs:
#         logger.error("No COLMAP reconstruction found")
#         return flight_group
    
#     # Use the largest reconstruction
#     largest_recon = max(reconstruction_dirs, key=lambda d: os.path.getsize(sparse_dir / d))
#     logger.info(f"Using reconstruction {largest_recon}")
    
#     # Parse COLMAP output files
#     images_file = sparse_dir / largest_recon / 'images.txt'
    
#     # Images.txt contains camera poses
#     orientations = {}
    
#     if not images_file.exists():
#         logger.error(f"COLMAP images file not found: {images_file}")
#         return flight_group
    
#     # Parse images.txt to get orientations
#     with open(images_file, 'r') as f:
#         lines = f.readlines()
    
#     for i in range(0, len(lines), 2):
#         if i+1 >= len(lines) or lines[i].startswith('#'):
#             continue
            
#         # Parse image line
#         parts = lines[i].strip().split()
        
#         # Format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#         if len(parts) < 10:
#             continue
            
#         image_id = int(parts[0])
#         qw, qx, qy, qz = map(float, parts[1:5])
#         tx, ty, tz = map(float, parts[5:8])
#         image_name = parts[9]
        
#         # Convert quaternion to Euler angles (pitch, yaw, roll)
#         pitch, yaw, roll = quaternion_to_euler(qw, qx, qy, qz)
        
#         # Store orientations
#         orientations[image_name] = {
#             'Phi (Pitch)': pitch,
#             'Kappa (Yaw)': yaw,
#             'Omega (Roll)': roll,
#             'TX': tx,
#             'TY': ty,
#             'TZ': tz
#         }
    
#     # Update flight group with COLMAP orientations
#     logger.info(f"Found orientation data for {len(orientations)} images")
    
#     orientations_set = 0
#     updated_group = flight_group.copy()
    
#     with tqdm(total=len(copied_images), desc="Updating orientation data", unit="img") as pbar:
#         for idx, filename in copied_images:
#             if filename in orientations:
#                 # Update orientation data
#                 updated_group.at[idx, 'Phi (Pitch)'] = orientations[filename]['Phi (Pitch)']
#                 updated_group.at[idx, 'Kappa (Yaw)'] = orientations[filename]['Kappa (Yaw)']
#                 updated_group.at[idx, 'Omega (Roll)'] = orientations[filename]['Omega (Roll)']
#                 updated_group.at[idx, 'TX'] = orientations[filename]['TX']
#                 updated_group.at[idx, 'TY'] = orientations[filename]['TY']
#                 updated_group.at[idx, 'TZ'] = orientations[filename]['TZ']
#                 updated_group.at[idx, 'Orientation_Source'] = 'COLMAP_SfM'
#                 orientations_set += 1
#             pbar.update(1)
    
#     logger.info(f"Updated orientation data for {orientations_set} images")
    
#     return updated_group

def split_large_flight_group(group_df, max_images=MAX_IMAGES_PER_BATCH):
    """
    Split a large flight group into more manageable sub-groups
    
    Args:
        group_df: DataFrame with flight group images
        max_images: Maximum images per sub-group
        
    Returns:
        list: List of sub-groups
    """
    if len(group_df) <= max_images:
        return [group_df]
    
    # If we have timestamp data, split by time
    if 'Timestamp' in group_df.columns and group_df['Timestamp'].notna().any():
        sorted_df = group_df.sort_values('Timestamp')
        
        # Calculate number of sub-groups needed
        num_groups = math.ceil(len(sorted_df) / max_images)
        
        # Split into roughly equal sized chunks
        return np.array_split(sorted_df, num_groups)
    else:
        # If no timestamp, just split randomly
        return np.array_split(group_df, math.ceil(len(group_df) / max_images))