"""
PyCOLMAP Wrapper for SfM-based orientation estimation

This module provides pycolmap binding alternatives to the subprocess-based COLMAP 
functions in colmap_helpers.py. It maintains the same interface for easy swapping.
"""

import os
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

def has_pycolmap():
    """Check if pycolmap is available"""
    try:
        import pycolmap
        return True
    except ImportError:
        return False

def run_pycolmap_sfm(flight_group, output_dir, use_gpu=True):
    """
    Run COLMAP SfM using native pycolmap bindings
    
    This function provides the same interface as run_colmap_sfm_optimized
    from colmap_helpers.py, but uses pycolmap bindings instead of subprocess calls.
    
    Args:
        flight_group: DataFrame with flight group images
        output_dir: Directory for COLMAP output
        use_gpu: Whether to use GPU for COLMAP processing
        
    Returns:
        DataFrame: Updated with orientation data from COLMAP
    """
    if not has_pycolmap():
        logger.error("pycolmap not installed. Please install with: pip install pycolmap")
        # Import the subprocess version as fallback
        from .colmap_helpers import run_colmap_sfm_optimized
        return run_colmap_sfm_optimized(flight_group, output_dir, use_gpu)
    
    logger.info(f"Running pycolmap SfM, outputting to {output_dir}")
    
    try:
        import pycolmap
    except ImportError:
        logger.error("Failed to import pycolmap")
        from .colmap_helpers import run_colmap_sfm_optimized
        return run_colmap_sfm_optimized(flight_group, output_dir, use_gpu)
    
    # Import necessary functions from colmap_helpers, but avoid circular imports
    # by importing only what we need
    try:
        from .colmap_helpers import downsample_images_for_sfm, quaternion_to_euler
        from .colmap_helpers import MAX_IMAGES_PER_BATCH  # Import constant
    except ImportError:
        # Define defaults if import fails
        logger.warning("Could not import from colmap_helpers, using default values")
        MAX_IMAGES_PER_BATCH = 150
        
        def downsample_images_for_sfm(flight_group, max_images=150):
            """Simple version of downsampling if import fails"""
            if len(flight_group) <= max_images:
                return flight_group
            return flight_group.sample(max_images)
            
        def quaternion_to_euler(qw, qx, qy, qz):
            """Convert quaternion to Euler angles if import fails"""
            import math
            
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (qw * qy - qz * qx)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)
            else:
                pitch = math.asin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # Convert to degrees
            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)
            roll_deg = math.degrees(roll)
            
            return pitch_deg, yaw_deg, roll_deg
    
    # Create directories
    colmap_dir = Path(output_dir)
    os.makedirs(colmap_dir, exist_ok=True)
    images_dir = colmap_dir / 'images'
    sparse_dir = colmap_dir / 'sparse'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Downsample images for SfM if needed
    sfm_images = downsample_images_for_sfm(flight_group)
    logger.info(f"Using {len(sfm_images)} images for SfM processing")
    
    # Copy/link images to COLMAP directory
    copied_images = []
    
    with tqdm(total=len(sfm_images), desc="Preparing images", unit="img") as pbar:
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
            
            try:
                # Remove existing file/link if it exists
                if os.path.exists(img_dest):
                    os.remove(img_dest)
                    
                # Create symbolic link
                os.symlink(os.path.abspath(img_source), img_dest)
                copied_images.append((idx, filename))
            except Exception as e:
                logger.error(f"Error creating symlink for {img_source}: {str(e)}")
            
            pbar.update(1)
    
    logger.info(f"Prepared {len(copied_images)} images for SfM processing")
    
    if len(copied_images) < 3:
        logger.warning("Not enough images for SfM. Need at least 3 images.")
        return flight_group
    
    # Database path
    database_path = colmap_dir / 'database.db'
    
    try:
        # Initialize database
        logger.info("Initializing COLMAP database...")
        database = pycolmap.Database()
        database.create_new(str(database_path))
        
        # Extract features
        logger.info("Extracting image features...")
        
        # Configure feature extraction options
        extraction_options = pycolmap.SiftExtractionOptions()
        extraction_options.use_gpu = use_gpu
        extraction_options.estimate_affine_shape = False
        extraction_options.domain_size_pooling = True
        extraction_options.max_image_size = 1600  # Limit size for memory
        extraction_options.max_num_features = 8192
        extraction_options.first_octave = -1  # Start at lower resolution
        extraction_options.peak_threshold = 0.0066667  # More conservative threshold
            
        # Extract features for all images
        pycolmap.extract_features(database, str(images_dir), extraction_options)
        
        # Match features
        logger.info("Matching image features...")
        
        # Configure matching options
        matching_options = pycolmap.SiftMatchingOptions()
        matching_options.use_gpu = use_gpu
        matching_options.max_ratio = 0.8
        matching_options.cross_check = True
        matching_options.guided_matching = True  # More precise matches
        
        # Try different matching strategies
        if len(copied_images) < 30:
            # For small datasets, use exhaustive matching
            logger.info("Using exhaustive matching for small dataset...")
            try:
                pycolmap.match_exhaustive(database, matching_options)
            except Exception as e:
                logger.warning(f"Exhaustive matching failed: {e}, falling back to sequential matching...")
                sequential_options = pycolmap.SequentialMatchingOptions()
                sequential_options.overlap = 15
                pycolmap.match_sequential(database, matching_options, sequential_options)
        else:
            # For larger datasets, use sequential matching
            logger.info("Using sequential matching for larger dataset...")
            try:
                sequential_options = pycolmap.SequentialMatchingOptions()
                sequential_options.overlap = 15
                sequential_options.loop_detection = True
                pycolmap.match_sequential(database, matching_options, sequential_options)
            except Exception as e:
                logger.warning(f"Sequential matching failed: {e}, falling back to spatial matching...")
                
                try:
                    spatial_options = pycolmap.SpatialMatchingOptions()
                    spatial_options.max_num_neighbors = 50
                    pycolmap.match_spatial(database, matching_options, spatial_options)
                except Exception as e:
                    logger.error(f"All matching strategies failed. Error: {e}")
                    # Fall back to subprocess version
                    from .colmap_helpers import run_colmap_sfm_optimized
                    return run_colmap_sfm_optimized(flight_group, output_dir, use_gpu)
        
        # Run reconstruction
        logger.info("Running SfM reconstruction...")
        
        # Configure mapper options
        mapper_options = pycolmap.IncrementalMapperOptions()
        mapper_options.ba_global_max_num_iterations = 75
        mapper_options.ba_local_max_num_iterations = 30
        mapper_options.ba_global_max_refinements = 5
        mapper_options.min_num_matches = 10
        mapper_options.init_min_num_inliers = 15
        mapper_options.abs_pose_min_num_inliers = 15
        mapper_options.abs_pose_min_inlier_ratio = 0.25
        mapper_options.filter_max_reproj_error = 8.0
        
        # Run incremental mapping
        reconstructions = pycolmap.incremental_mapping(database_path=str(database_path),
                                                   image_path=str(images_dir),
                                                   output_path=str(sparse_dir),
                                                   options=mapper_options)
        
        # If reconstructions failed or are empty, try with more permissive settings
        if not reconstructions or len(reconstructions) == 0:
            logger.warning("Reconstruction failed with initial parameters, trying with more permissive settings...")
            
            # More permissive mapper options
            mapper_options.init_min_num_inliers = 8
            mapper_options.abs_pose_min_num_inliers = 8
            mapper_options.abs_pose_min_inlier_ratio = 0.15
            mapper_options.min_num_matches = 5
            mapper_options.filter_max_reproj_error = 12.0
            
            # Try again
            reconstructions = pycolmap.incremental_mapping(database_path=str(database_path),
                                                       image_path=str(images_dir),
                                                       output_path=str(sparse_dir),
                                                       options=mapper_options)
        
        # If still failed, return original data
        if not reconstructions or len(reconstructions) == 0:
            logger.error("All reconstruction attempts failed")
            # Fall back to subprocess version
            from .colmap_helpers import run_colmap_sfm_optimized
            return run_colmap_sfm_optimized(flight_group, output_dir, use_gpu)
        
        # Find best reconstruction (with most images)
        best_reconstruction = max(reconstructions, key=lambda r: r.num_registered_images())
        logger.info(f"Best reconstruction has {best_reconstruction.num_registered_images()} images")
        
        # Extract orientations from reconstruction
        orientations = {}
        
        for image_id, image in best_reconstruction.images.items():
            image_name = image.name
            
            # Get camera pose (quaternion and translation)
            qvec = image.qvec
            tvec = image.tvec
            
            # Convert quaternion to Euler angles
            pitch, yaw, roll = quaternion_to_euler(qvec[0], qvec[1], qvec[2], qvec[3])
            
            # Store orientations
            orientations[image_name] = {
                'Phi (Pitch)': pitch,
                'Kappa (Yaw)': yaw,
                'Omega (Roll)': roll,
                'TX': tvec[0],
                'TY': tvec[1],
                'TZ': tvec[2]
            }
        
        # Update flight group with orientations
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
                    updated_group.at[idx, 'Orientation_Source'] = 'PyCOLMAP_SfM'
                    updated_group.at[idx, 'Orientation_Confidence'] = 'High'
                    orientations_set += 1
                pbar.update(1)
        
        logger.info(f"Updated orientation data for {orientations_set} images")
        
        return updated_group
        
    except Exception as e:
        logger.error(f"Error in pycolmap processing: {e}")
        # Fall back to subprocess version
        from .colmap_helpers import run_colmap_sfm_optimized
        return run_colmap_sfm_optimized(flight_group, output_dir, use_gpu)

def run_pycolmap_sfm_with_enhanced_stability(flight_group, output_dir, use_gpu=True):
    """
    Run COLMAP SfM with enhanced stability using pycolmap bindings
    
    This function provides the same interface as run_colmap_sfm_with_enhanced_stability
    from colmap_helpers.py, but uses pycolmap bindings instead of subprocess calls.
    
    Args:
        flight_group: DataFrame with flight group images
        output_dir: Directory for COLMAP output
        use_gpu: Whether to use GPU for COLMAP processing
        
    Returns:
        DataFrame: Updated with orientation data from COLMAP
    """
    # For enhanced stability, we use a modified version of run_pycolmap_sfm
    # with more conservative parameters and additional preprocessing
    if not has_pycolmap():
        logger.error("pycolmap not installed. Please install with: pip install pycolmap")
        # Import the subprocess version as fallback
        from .colmap_helpers import run_colmap_sfm_with_enhanced_stability
        return run_colmap_sfm_with_enhanced_stability(flight_group, output_dir, use_gpu)
    
    logger.info(f"Running pycolmap SfM with enhanced stability, outputting to {output_dir}")
    
    try:
        import pycolmap
        import cv2
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        from .colmap_helpers import run_colmap_sfm_with_enhanced_stability
        return run_colmap_sfm_with_enhanced_stability(flight_group, output_dir, use_gpu)
    
    # Import necessary functions from colmap_helpers
    try:
        from .colmap_helpers import downsample_images_for_sfm, quaternion_to_euler
    except ImportError:
        # Define simplified versions if import fails
        logger.warning("Could not import from colmap_helpers, using simplified functions")
        # (simplified functions same as above)
    
    # Create directories
    colmap_dir = Path(output_dir)
    os.makedirs(colmap_dir, exist_ok=True)
    images_dir = colmap_dir / 'images'
    preprocessed_dir = colmap_dir / 'preprocessed'
    sparse_dir = colmap_dir / 'sparse'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    
    # Downsample images for SfM if needed
    sfm_images = downsample_images_for_sfm(flight_group)
    logger.info(f"Using {len(sfm_images)} images for SfM processing")
    
    # Copy and preprocess images with enhanced stability
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
                img = cv2.imread(str(img_source))
                if img is None:
                    logger.warning(f"Could not read image: {img_source}")
                    pbar.update(1)
                    continue
                
                # Resize image to limit memory usage
                max_dimension = 1600
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
    
    # Database path
    database_path = colmap_dir / 'database.db'
    
    try:
        # Initialize database
        logger.info("Initializing COLMAP database...")
        database = pycolmap.Database()
        database.create_new(str(database_path))
        
        # Extract features with enhanced stability parameters
        logger.info("Extracting image features with enhanced stability...")
        
        # Configure feature extraction options for enhanced stability
        extraction_options = pycolmap.SiftExtractionOptions()
        extraction_options.use_gpu = use_gpu
        extraction_options.estimate_affine_shape = False
        extraction_options.domain_size_pooling = True
        extraction_options.max_image_size = 1600  # Limit size for memory
        extraction_options.max_num_features = 4096  # Reduced for memory
        extraction_options.first_octave = -1  # Start at lower resolution
        extraction_options.peak_threshold = 0.0066667  # More conservative threshold
        
        # Extract features for all images
        pycolmap.extract_features(database, str(images_dir), extraction_options)
        
        # Match features with enhanced stability parameters
        logger.info("Matching image features with enhanced stability...")
        
        # Configure matching options for enhanced stability
        matching_options = pycolmap.SiftMatchingOptions()
        matching_options.use_gpu = use_gpu
        matching_options.max_ratio = 0.8
        matching_options.cross_check = True
        matching_options.multiple_models = False  # More stable with single model
        matching_options.guided_matching = True  # More precise matches
        
        # Try different matching strategies with better error handling
        if len(copied_images) < 30:
            # For small datasets, use exhaustive matching
            logger.info("Using exhaustive matching for small dataset...")
            try:
                pycolmap.match_exhaustive(database, matching_options)
            except Exception as e:
                logger.warning(f"Exhaustive matching failed: {e}, falling back to sequential matching...")
                try:
                    sequential_options = pycolmap.SequentialMatchingOptions()
                    sequential_options.overlap = 15
                    pycolmap.match_sequential(database, matching_options, sequential_options)
                except Exception as e2:
                    logger.error(f"Sequential matching also failed: {e2}, falling back to subprocess method")
                    from .colmap_helpers import run_colmap_sfm_with_enhanced_stability
                    return run_colmap_sfm_with_enhanced_stability(flight_group, output_dir, use_gpu)
        else:
            # For larger datasets, use sequential matching
            logger.info("Using sequential matching for larger dataset...")
            try:
                sequential_options = pycolmap.SequentialMatchingOptions()
                sequential_options.overlap = 15
                sequential_options.loop_detection = True
                pycolmap.match_sequential(database, matching_options, sequential_options)
            except Exception as e:
                logger.warning(f"Sequential matching failed: {e}, falling back to spatial matching...")
                
                try:
                    spatial_options = pycolmap.SpatialMatchingOptions()
                    spatial_options.max_num_neighbors = 50
                    pycolmap.match_spatial(database, matching_options, spatial_options)
                except Exception as e2:
                    logger.error(f"All matching strategies failed: {e2}, falling back to subprocess method")
                    from .colmap_helpers import run_colmap_sfm_with_enhanced_stability
                    return run_colmap_sfm_with_enhanced_stability(flight_group, output_dir, use_gpu)
        
        # Run reconstruction with enhanced stability parameters
        logger.info("Running SfM reconstruction with enhanced stability...")
        
        # Configure mapper options for enhanced stability
        mapper_options = pycolmap.IncrementalMapperOptions()
        mapper_options.ba_global_max_num_iterations = 75  # More iterations
        mapper_options.ba_local_max_num_iterations = 30  # More iterations
        mapper_options.ba_global_max_refinements = 5  # More refinements
        mapper_options.min_num_matches = 10  # Fewer required matches
        mapper_options.ba_global_function_tolerance = 0.001  # Relaxed tolerance
        mapper_options.filter_max_reproj_error = 8.0  # More permissive threshold
        mapper_options.init_min_num_inliers = 15  # Fewer required inliers
        mapper_options.abs_pose_min_num_inliers = 15  # Fewer required inliers
        mapper_options.abs_pose_min_inlier_ratio = 0.25  # More permissive ratio
        
        # Run incremental mapping
        try:
            reconstructions = pycolmap.incremental_mapping(database_path=str(database_path),
                                                       image_path=str(images_dir),
                                                       output_path=str(sparse_dir),
                                                       options=mapper_options)
        except Exception as e:
            logger.warning(f"Incremental mapping failed: {e}, trying with more permissive settings...")
            
            # More permissive mapper options
            mapper_options.init_min_num_inliers = 8  # Bare minimum
            mapper_options.abs_pose_min_num_inliers = 8  # Bare minimum
            mapper_options.abs_pose_min_inlier_ratio = 0.15  # Very permissive
            mapper_options.min_num_matches = 5  # Minimum matches
            mapper_options.filter_max_reproj_error = 12.0  # Very permissive
            
            try:
                reconstructions = pycolmap.incremental_mapping(database_path=str(database_path),
                                                           image_path=str(images_dir),
                                                           output_path=str(sparse_dir),
                                                           options=mapper_options)
            except Exception as e2:
                logger.error(f"All reconstruction attempts failed: {e2}, falling back to subprocess method")
                from .colmap_helpers import run_colmap_sfm_with_enhanced_stability
                return run_colmap_sfm_with_enhanced_stability(flight_group, output_dir, use_gpu)
        
        # If reconstructions failed or are empty, fall back to subprocess method
        if not reconstructions or len(reconstructions) == 0:
            logger.error("Reconstruction returned no results, falling back to subprocess method")
            from .colmap_helpers import run_colmap_sfm_with_enhanced_stability
            return run_colmap_sfm_with_enhanced_stability(flight_group, output_dir, use_gpu)
        
        # Find best reconstruction (with most images)
        best_reconstruction = max(reconstructions, key=lambda r: r.num_registered_images())
        logger.info(f"Best reconstruction has {best_reconstruction.num_registered_images()} images")
        
        # Extract orientations from reconstruction
        orientations = {}
        
        for image_id, image in best_reconstruction.images.items():
            image_name = image.name
            
            # Get camera pose (quaternion and translation)
            qvec = image.qvec
            tvec = image.tvec
            
            # Convert quaternion to Euler angles
            pitch, yaw, roll = quaternion_to_euler(qvec[0], qvec[1], qvec[2], qvec[3])
            
            # Store orientations
            orientations[image_name] = {
                'Phi (Pitch)': pitch,
                'Kappa (Yaw)': yaw,
                'Omega (Roll)': roll,
                'TX': tvec[0],
                'TY': tvec[1],
                'TZ': tvec[2]
            }
        
        # Update flight group with orientations
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
                    updated_group.at[idx, 'Orientation_Source'] = 'Enhanced_PyCOLMAP_SfM'
                    updated_group.at[idx, 'Orientation_Confidence'] = 'High'
                    orientations_set += 1
                pbar.update(1)
        
        logger.info(f"Updated orientation data for {orientations_set} images")
        
        return updated_group
        
    except Exception as e:
        logger.error(f"Error in enhanced pycolmap processing: {e}")
        # Fall back to subprocess version
        from .colmap_helpers import run_colmap_sfm_with_enhanced_stability
        return run_colmap_sfm_with_enhanced_stability(flight_group, output_dir, use_gpu)


def run_hierarchical_sfm(flight_group, output_dir, use_gpu=True):
    """
    Run hierarchical COLMAP SfM for large datasets
    
    This function is specifically designed for very large datasets that would
    be inefficient to process with the standard incremental pipeline.
    
    Args:
        flight_group: DataFrame with flight group images
        output_dir: Directory for COLMAP output
        use_gpu: Whether to use GPU for COLMAP processing
        
    Returns:
        DataFrame: Updated with orientation data from COLMAP
    """
    if not has_pycolmap():
        logger.error("pycolmap not installed. Please install with: pip install pycolmap")
        # Import the subprocess version as fallback
        from .colmap_helpers import run_colmap_sfm_optimized
        return run_colmap_sfm_optimized(flight_group, output_dir, use_gpu)
    
    try:
        import pycolmap
        # Check if hierarchical_mapping is available in this pycolmap version
        if not hasattr(pycolmap, 'hierarchical_mapping'):
            logger.warning("hierarchical_mapping not available in this pycolmap version")
            # Fall back to standard pycolmap
            return run_pycolmap_sfm(flight_group, output_dir, use_gpu)
    except ImportError:
        logger.error("Failed to import pycolmap")
        from .colmap_helpers import run_colmap_sfm_optimized
        return run_colmap_sfm_optimized(flight_group, output_dir, use_gpu)
    
    # Import necessary functions from colmap_helpers
    try:
        from .colmap_helpers import quaternion_to_euler
    except ImportError:
        # Define simplified version if import fails
        logger.warning("Could not import from colmap_helpers, using simplified functions")
        
        def quaternion_to_euler(qw, qx, qy, qz):
            """Convert quaternion to Euler angles if import fails"""
            import math
            
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (qw * qx + qy * qz)
            cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (qw * qy - qz * qx)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)
            else:
                pitch = math.asin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            
            # Convert to degrees
            pitch_deg = math.degrees(pitch)
            yaw_deg = math.degrees(yaw)
            roll_deg = math.degrees(roll)
            
            return pitch_deg, yaw_deg, roll_deg
    
    logger.info(f"Running hierarchical COLMAP SfM for large dataset, outputting to {output_dir}")
    
    # Create directories
    colmap_dir = Path(output_dir)
    os.makedirs(colmap_dir, exist_ok=True)
    images_dir = colmap_dir / 'images'
    sparse_dir = colmap_dir / 'sparse'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)
    
    # For hierarchical processing, we don't need to downsample as much
    # since it's designed to handle large datasets more efficiently
    
    # Copy/link images to COLMAP directory
    copied_images = []
    
    with tqdm(total=len(flight_group), desc="Preparing images", unit="img") as pbar:
        for idx, row in flight_group.iterrows():
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
            
            try:
                # Remove existing file/link if it exists
                if os.path.exists(img_dest):
                    os.remove(img_dest)
                    
                # Create symbolic link
                os.symlink(os.path.abspath(img_source), img_dest)
                copied_images.append((idx, filename))
            except Exception as e:
                logger.error(f"Error creating symlink for {img_source}: {str(e)}")
            
            pbar.update(1)
    
    logger.info(f"Prepared {len(copied_images)} images for hierarchical SfM processing")
    
    if len(copied_images) < 10:
        logger.warning("Too few images for hierarchical SfM. Using standard SfM instead.")
        return run_pycolmap_sfm(flight_group, output_dir, use_gpu)
    
    # Database path
    database_path = colmap_dir / 'database.db'
    
    try:
        # Initialize database
        logger.info("Initializing COLMAP database...")
        database = pycolmap.Database()
        database.create_new(str(database_path))
        
        # Extract features
        logger.info("Extracting image features...")
        
        # Configure feature extraction options
        extraction_options = pycolmap.SiftExtractionOptions()
        extraction_options.use_gpu = use_gpu
        extraction_options.max_image_size = 1600  # Memory optimization for large datasets
        extraction_options.max_num_features = 4096  # Memory optimization for large datasets
        
        # Extract features for all images
        pycolmap.extract_features(database, str(images_dir), extraction_options)
        
        # Match features using vocabulary tree for efficiency with large datasets
        logger.info("Matching image features for hierarchical SfM...")
        
        # Configure matching options
        matching_options = pycolmap.SiftMatchingOptions()
        matching_options.use_gpu = use_gpu
        matching_options.max_ratio = 0.8
        matching_options.cross_check = True
        
        # For large datasets, sequential matching with higher overlap
        sequential_options = pycolmap.SequentialMatchingOptions()
        sequential_options.overlap = 20  # Higher overlap for hierarchical processing
        sequential_options.loop_detection = True
        pycolmap.match_sequential(database, matching_options, sequential_options)
        
        # Run hierarchical reconstruction
        logger.info("Running hierarchical SfM reconstruction...")
        
        # Configure hierarchical mapper options
        hierarchical_options = pycolmap.HierarchicalMapperOptions()
        
        # Run hierarchical mapping
        try:
            reconstructions = pycolmap.hierarchical_mapping(
                database_path=str(database_path),
                image_path=str(images_dir),
                output_path=str(sparse_dir),
                options=hierarchical_options
            )
        except Exception as e:
            logger.warning(f"Hierarchical mapping failed: {e}, falling back to incremental mapping")
            # Fall back to incremental mapping with memory-optimized settings
            mapper_options = pycolmap.IncrementalMapperOptions()
            mapper_options.ba_global_max_num_iterations = 50
            mapper_options.ba_local_max_num_iterations = 20
            mapper_options.min_num_matches = 10
            
            reconstructions = pycolmap.incremental_mapping(
                database_path=str(database_path),
                image_path=str(images_dir),
                output_path=str(sparse_dir),
                options=mapper_options
            )
        
        # If reconstructions failed or are empty, fall back to standard pycolmap
        if not reconstructions or len(reconstructions) == 0:
            logger.error("Hierarchical reconstruction failed, falling back to standard SfM")
            return run_pycolmap_sfm(flight_group, output_dir, use_gpu)
        
        # Find best reconstruction (with most images)
        best_reconstruction = max(reconstructions, key=lambda r: r.num_registered_images())
        logger.info(f"Best reconstruction has {best_reconstruction.num_registered_images()} images")
        
        # Extract orientations from reconstruction
        orientations = {}
        
        for image_id, image in best_reconstruction.images.items():
            image_name = image.name
            
            # Get camera pose (quaternion and translation)
            qvec = image.qvec
            tvec = image.tvec
            
            # Convert quaternion to Euler angles
            pitch, yaw, roll = quaternion_to_euler(qvec[0], qvec[1], qvec[2], qvec[3])
            
            # Store orientations
            orientations[image_name] = {
                'Phi (Pitch)': pitch,
                'Kappa (Yaw)': yaw,
                'Omega (Roll)': roll,
                'TX': tvec[0],
                'TY': tvec[1],
                'TZ': tvec[2]
            }
        
        # Update flight group with orientations
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
                    updated_group.at[idx, 'Orientation_Source'] = 'Hierarchical_PyCOLMAP'
                    updated_group.at[idx, 'Orientation_Confidence'] = 'High'
                    orientations_set += 1
                pbar.update(1)
        
        logger.info(f"Updated orientation data for {orientations_set} images")
        
        return updated_group
        
    except Exception as e:
        logger.error(f"Error in hierarchical pycolmap processing: {e}")
        # Fall back to standard pycolmap
        return run_pycolmap_sfm(flight_group, output_dir, use_gpu)