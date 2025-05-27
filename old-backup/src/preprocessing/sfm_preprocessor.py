import os
import sys
import shutil
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import cv2
import argparse
from tqdm import tqdm
from collections import defaultdict
import logging
import cv2

# Import the enhanced metadata extractor
from enhanced_metadata_extractor import EnhancedMetadataExtractor

class SfMPreprocessor:
    """
    Preprocessing workflow for Structure-from-Motion with incomplete metadata.
    This class prepares image datasets for 3D reconstruction software by:
    1. Organizing images into coherent datasets
    2. Enhancing metadata where possible
    3. Generating camera calibration files
    4. Creating input files for popular SfM software
    """
    
    def __init__(self, config):
        """
        Initialize the SfM preprocessor
        
        Args:
            config: Configuration dictionary with the following keys:
                - input_dir: Directory containing input images
                - output_dir: Directory for preprocessed data
                - work_dir: Temporary working directory
                - thermal_suffix: Suffix for thermal images
                - rgb_suffix: Suffix for RGB images
                - sfm_software: Target SfM software ('colmap', 'opendronemapopen_sfm', 'metashape')
                - group_by_timestamp: Whether to group images by timestamp
                - max_time_diff: Maximum time difference for grouping (seconds)
        """
        self.config = config
        self.input_dir = config['input_dir']
        self.output_dir = config['output_dir']
        self.work_dir = config['work_dir']
        self.thermal_suffix = config.get('thermal_suffix', ['IR', 'THERMAL'])
        self.rgb_suffix = config.get('rgb_suffix', ['RGB', 'MIX'])
        self.sfm_software = config.get('sfm_software', 'colmap')
        self.group_by_timestamp = config.get('group_by_timestamp', True)
        self.max_time_diff = config.get('max_time_diff', 2)  # seconds
        
        # Create necessary directories
        for directory in [self.output_dir, self.work_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'preprocessing.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SfMPreprocessor')
        
        # Will hold the dataset metadata
        self.metadata_df = None
        self.image_groups = None
        self.extractor = None
        
    def _is_thermal_image(self, filename):
        """Check if an image is a thermal image based on its filename"""
        return any(suffix in filename for suffix in self.thermal_suffix)
    
    def _is_rgb_image(self, filename):
        """Check if an image is an RGB image based on its filename"""
        return any(suffix in filename for suffix in self.rgb_suffix)
        
    def extract_metadata(self):
        """
        Extract and enhance metadata from all images
        
        Returns:
            DataFrame with enhanced metadata
        """
        self.logger.info("Extracting metadata from input images...")
        
        # Create the metadata extractor
        self.extractor = EnhancedMetadataExtractor(
            self.input_dir, 
            os.path.join(self.output_dir, 'metadata')
        )
        
        # Extract basic metadata
        self.metadata_df = self.extractor.extract_basic_metadata()
        
        # Process camera calibration
        self.extractor.extract_camera_calibration()
        
        # Infer missing orientation data
        self.extractor.infer_missing_orientation()
        
        # Prepare the final photogrammetry dataset
        self.metadata_df = self.extractor.prepare_photogrammetry_dataset()
        
        self.logger.info(f"Found {len(self.metadata_df)} usable images for SfM")
        
        return self.metadata_df
    
    def group_images(self):
        """
        Group images into datasets for reconstruction
        - Images can be grouped by timestamp (sequence) or by spatial proximity
        - Both RGB and corresponding thermal images are grouped together
        
        Returns:
            Dictionary of image groups
        """
        if self.metadata_df is None:
            self.extract_metadata()
            
        self.logger.info("Grouping images into reconstruction sets...")
        
        # Initialize groups
        self.image_groups = defaultdict(list)
        
        if self.group_by_timestamp:
            # Group images by timestamp (within max_time_diff seconds)
            # First, ensure timestamp is present
            if 'Timestamp' not in self.metadata_df.columns or self.metadata_df['Timestamp'].isna().all():
                self.logger.warning("No timestamp information available. Using filename patterns instead.")
                self._group_by_filename_pattern()
            else:
                # Sort by timestamp
                sorted_df = self.metadata_df.sort_values('Timestamp').copy()
                
                # Assign groups based on time gaps
                current_group = 0
                last_time = None
                
                for idx, row in sorted_df.iterrows():
                    current_time = row['Timestamp']
                    
                    if last_time is not None:
                        # Calculate time difference in seconds
                        time_diff = (current_time - last_time).total_seconds()
                        
                        # If time gap is too large, start a new group
                        if time_diff > self.max_time_diff:
                            current_group += 1
                    
                    # Add to current group
                    group_name = f"sequence_{current_group:03d}"
                    self.image_groups[group_name].append(row['SourceFile'])
                    
                    # Update last time
                    last_time = current_time
        else:
            # Group by spatial proximity (clustering GPS coordinates)
            # This is a more complex approach for when timestamps aren't reliable
            self._group_by_spatial_clustering()
        
        # Log grouping results
        self.logger.info(f"Created {len(self.image_groups)} image groups")
        for group_name, files in self.image_groups.items():
            self.logger.info(f"Group {group_name}: {len(files)} images")
        
        return self.image_groups
    
    def _group_by_filename_pattern(self):
        """Group images based on filename patterns"""
        # Try to extract sequence numbers from filenames
        pattern_groups = defaultdict(list)
        
        for idx, row in self.metadata_df.iterrows():
            filename = os.path.basename(row['SourceFile'])
            
            # Try to identify a sequence pattern
            # Example: if files are named like "IMG_0001.JPG", "IMG_0002.JPG"
            # Extract the prefix and sequence number
            parts = filename.split('_')
            if len(parts) >= 2:
                prefix = parts[0]
                try:
                    # Try to get a sequence number from the second part
                    seq_part = parts[1].split('.')[0]
                    # Find any consecutive digits
                    digits = ''.join(c for c in seq_part if c.isdigit())
                    if digits:
                        # Use first 2 digits for grouping (adjust as needed)
                        group_id = digits[:2]
                        group_name = f"{prefix}_{group_id}"
                        pattern_groups[group_name].append(row['SourceFile'])
                        continue
                except (IndexError, ValueError):
                    pass
            
            # Fallback: use directory as group
            directory = os.path.dirname(row['SourceFile'])
            dir_name = os.path.basename(directory)
            pattern_groups[dir_name].append(row['SourceFile'])
        
        self.image_groups = pattern_groups
    
    def _group_by_spatial_clustering(self):
        """Group images based on spatial clustering of GPS coordinates"""
        # Check if GPS coordinates are available
        if ('Latitude' not in self.metadata_df.columns or 
            'Longitude' not in self.metadata_df.columns or
            self.metadata_df['Latitude'].isna().all() or
            self.metadata_df['Longitude'].isna().all()):
            
            self.logger.warning("GPS coordinates not available for spatial clustering. Using filename patterns instead.")
            self._group_by_filename_pattern()
            return
        
        # Extract coordinates for clustering
        coords_df = self.metadata_df.dropna(subset=['Latitude', 'Longitude']).copy()
        
        # Convert string coordinates to float if needed
        for col in ['Latitude', 'Longitude']:
            if coords_df[col].dtype == 'object':
                # Parse DMS format like "40 deg 31' 27.83\" N"
                coords_df[col] = coords_df[col].apply(self._parse_dms_to_decimal)
        
        # Perform simple distance-based clustering
        from sklearn.cluster import DBSCAN
        
        # Prepare coordinates array for clustering
        coords = coords_df[['Latitude', 'Longitude']].values
        
        # Perform DBSCAN clustering
        # eps is in degrees (approximately 50-100 meters)
        clustering = DBSCAN(eps=0.001, min_samples=3).fit(coords)
        
        # Get cluster labels
        labels = clustering.labels_
        
        # Assign groups based on clusters
        coords_df['cluster'] = labels
        
        # Create groups
        for cluster_id in set(labels):
            if cluster_id == -1:
                # Noise points (assign to individual groups)
                noise_points = coords_df[coords_df['cluster'] == -1]
                for i, (idx, row) in enumerate(noise_points.iterrows()):
                    group_name = f"isolated_{i:03d}"
                    self.image_groups[group_name].append(row['SourceFile'])
            else:
                # Regular clusters
                cluster_points = coords_df[coords_df['cluster'] == cluster_id]
                group_name = f"location_{cluster_id:03d}"
                for _, row in cluster_points.iterrows():
                    self.image_groups[group_name].append(row['SourceFile'])
    
    def _parse_dms_to_decimal(self, dms_str):
        """Convert DMS (Degrees, Minutes, Seconds) string to decimal degrees"""
        if not isinstance(dms_str, str):
            return dms_str
            
        try:
            # Handle various formats
            # "40 deg 31' 27.83\" N"
            # "40°31'27.83\"N"
            # "40.523841 N"
            
            # First, determine if it's already decimal
            if ' deg ' not in dms_str and '°' not in dms_str and "'" not in dms_str:
                # Likely already decimal
                parts = dms_str.strip().split()
                decimal = float(parts[0])
                
                # Check for cardinal direction
                if len(parts) > 1 and parts[1] in ['S', 'W']:
                    decimal = -decimal
                
                return decimal
            
            # Handle standard DMS format
            direction = 1
            if any(cardinal in dms_str for cardinal in ['S', 'W']):
                direction = -1
                
            # Remove cardinal directions and other text
            for cardinal in ['N', 'S', 'E', 'W', 'deg', '°', '"', "'"]:
                dms_str = dms_str.replace(cardinal, ' ')
                
            # Split into components
            parts = [float(p) for p in dms_str.strip().split() if p]
            
            if len(parts) >= 3:
                # DMS format
                degrees, minutes, seconds = parts[:3]
                decimal = direction * (degrees + minutes/60 + seconds/3600)
            elif len(parts) >= 2:
                # DM format
                degrees, minutes = parts[:2]
                decimal = direction * (degrees + minutes/60)
            else:
                # D format
                decimal = direction * parts[0]
                
            return decimal
        except (ValueError, IndexError):
            self.logger.warning(f"Could not parse DMS string: {dms_str}")
            return None
    
    def prepare_colmap_data(self):
        """
        Prepare data for COLMAP reconstruction
        """
        if self.image_groups is None:
            self.group_images()
            
        self.logger.info("Preparing data for COLMAP reconstruction...")
        
        for group_name, image_files in self.image_groups.items():
            # Create group directory
            group_dir = os.path.join(self.output_dir, 'colmap', group_name)
            images_dir = os.path.join(group_dir, 'images')
            
            os.makedirs(images_dir, exist_ok=True)
            
            # Copy images to group directory
            for img_file in tqdm(image_files, desc=f"Copying images for {group_name}"):
                # Only copy if exists
                if os.path.exists(img_file):
                    shutil.copy2(img_file, images_dir)
            
            # Create camera calibration file
            self.extractor.generate_camera_file(output_format="colmap")
            
            # Copy calibration file to group directory
            calib_src = os.path.join(self.output_dir, 'metadata', 'cameras.txt')
            calib_dst = os.path.join(group_dir, 'cameras.txt')
            
            if os.path.exists(calib_src):
                shutil.copy2(calib_src, calib_dst)
            
            # Create images.txt (with camera poses) if orientation data is available
            self._create_colmap_images_file(group_dir, image_files)
            
            # Create COLMAP database
            self._create_colmap_database(group_dir)
            
            # Generate project file
            self._create_colmap_project_file(group_dir)
    
    def _create_colmap_images_file(self, group_dir, image_files):
        """Create COLMAP images.txt file with camera poses if available"""
        # Filter metadata for the group images
        filenames = [os.path.basename(f) for f in image_files]
        group_meta = self.metadata_df[self.metadata_df['Filename'].isin(filenames)]
        
        # Check if we have orientation data
        has_orientation = (
            group_meta['Phi (Pitch)'].notna().any() and
            group_meta['Kappa (Yaw)'].notna().any() and
            group_meta['Omega (Roll)'].notna().any()
        )
        
        if not has_orientation:
            self.logger.warning("No orientation data available for COLMAP images.txt")
            return
        
        # Create images.txt file
        images_file = os.path.join(group_dir, 'images.txt')
        
        with open(images_file, 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            
            for idx, row in group_meta.iterrows():
                if pd.isna(row['Phi (Pitch)']) or pd.isna(row['Kappa (Yaw)']) or pd.isna(row['Omega (Roll)']):
                    # Skip images without orientation
                    continue
                
                # Extract Euler angles (in degrees)
                phi = float(row['Phi (Pitch)'])
                kappa = float(row['Kappa (Yaw)'])
                omega = float(row['Omega (Roll)'])
                
                # Convert Euler angles to quaternion
                qw, qx, qy, qz = self._euler_to_quaternion(omega, phi, kappa)
                
                # Extract translation (GPS coordinates)
                tx, ty, tz = 0, 0, 0  # Default if no GPS
                
                if not pd.isna(row['Latitude']) and not pd.isna(row['Longitude']) and not pd.isna(row['Altitude']):
                    # Convert GPS to local coordinate system
                    # This is a simplified approach - in practice, you'd use a proper coordinate system conversion
                    lat = self._parse_dms_to_decimal(row['Latitude'])
                    lon = self._parse_dms_to_decimal(row['Longitude'])
                    alt = float(str(row['Altitude']).replace(' m', ''))
                    
                    # Simple conversion to local coordinates (centered at first image)
                    if idx == group_meta.index[0]:
                        self.ref_lat = lat
                        self.ref_lon = lon
                        self.ref_alt = alt
                        tx, ty, tz = 0, 0, 0
                    else:
                        # Convert to approx. meters
                        # 1 degree lat ≈ 111km, 1 degree lon ≈ 111km * cos(lat)
                        earth_radius = 6371000  # meters
                        tx = (lon - self.ref_lon) * np.pi/180 * earth_radius * np.cos(self.ref_lat * np.pi/180)
                        ty = (lat - self.ref_lat) * np.pi/180 * earth_radius
                        tz = alt - self.ref_alt
                
                # Write image entry
                image_id = idx + 1
                camera_id = 1  # Assuming single camera model
                image_name = row['Filename']
                
                f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {image_name}\n")
                f.write("\n")  # Empty line for points2D (we don't have these yet)
    
    def _euler_to_quaternion(self, roll, pitch, yaw):
        """
        Convert Euler angles to quaternion
        
        Args:
            roll: Roll angle in degrees
            pitch: Pitch angle in degrees
            yaw: Yaw angle in degrees
            
        Returns:
            Quaternion as (w, x, y, z)
        """
        # Convert to radians
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
        
        # Calculate quaternion components
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        
        return w, x, y, z
    
    def _create_colmap_database(self, group_dir):
        """Create empty COLMAP database"""
        import sqlite3
        
        db_path = os.path.join(group_dir, 'database.db')
        
        # Check if COLMAP schema SQL is available
        schema_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'colmap_schema.sql')
        
        if not os.path.exists(schema_path):
            # Create a simplified schema
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create minimal tables
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                model INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                params BLOB,
                prior_focal_length INTEGER NOT NULL
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                name TEXT NOT NULL UNIQUE,
                camera_id INTEGER NOT NULL,
                prior_qw REAL,
                prior_qx REAL,
                prior_qy REAL,
                prior_qz REAL,
                prior_tx REAL,
                prior_ty REAL,
                prior_tz REAL,
                CONSTRAINT image_camera_id_fkey FOREIGN KEY(camera_id) REFERENCES cameras(camera_id)
            )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Created simplified COLMAP database at {db_path}")
        else:
            # Use COLMAP schema
            with open(schema_path, 'r') as f:
                schema = f.read()
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.executescript(schema)
            conn.commit()
            conn.close()
            
            self.logger.info(f"Created COLMAP database with official schema at {db_path}")
    
    def _create_colmap_project_file(self, group_dir):
        """Create COLMAP project file"""
        project_path = os.path.join(group_dir, 'project.ini')
        
        with open(project_path, 'w') as f:
            f.write("[General]\n")
            f.write(f"database_path={os.path.join(group_dir, 'database.db')}\n")
            f.write(f"image_path={os.path.join(group_dir, 'images')}\n")
            f.write("skip_distortion=false\n")
            
            # Add camera calibration section if available
            calib_path = os.path.join(group_dir, 'cameras.txt')
            if os.path.exists(calib_path):
                f.write("\n[ImageReader]\n")
                f.write("single_camera=true\n")
                f.write("camera_model=SIMPLE_PINHOLE\n")
                f.write(f"camera_params_path={calib_path}\n")
        
        self.logger.info(f"Created COLMAP project file at {project_path}")
    
    def prepare_opendronemap_data(self):
        """
        Prepare data for OpenDroneMap reconstruction
        """
        if self.image_groups is None:
            self.group_images()
            
        self.logger.info("Preparing data for OpenDroneMap reconstruction...")
        
        for group_name, image_files in self.image_groups.items():
            # Create group directory
            group_dir = os.path.join(self.output_dir, 'odm', group_name)
            images_dir = os.path.join(group_dir, 'images')
            
            os.makedirs(images_dir, exist_ok=True)
            
            # Copy images to group directory
            for img_file in tqdm(image_files, desc=f"Copying images for {group_name}"):
                if os.path.exists(img_file):
                    shutil.copy2(img_file, images_dir)
            
            # Create camera parameters file (OpenDroneMap format)
            self._create_odm_camera_params(group_dir)
            
            # Create GCP file if GPS data is available
            self._create_odm_gcp_file(group_dir, image_files)
    
    def _create_odm_camera_params(self, group_dir):
        """Create camera parameters file for OpenDroneMap"""
        camera_file = os.path.join(group_dir, 'camera_calibration.json')
        
        # Get camera calibration data
        calib_file = os.path.join(self.output_dir, 'metadata', 'camera_calibration.csv')
        
        if not os.path.exists(calib_file):
            self.logger.warning("No camera calibration data available for OpenDroneMap")
            return
        
        # Load calibration data
        calib_df = pd.read_csv(calib_file)
        
        # Create ODM camera calibration JSON
        camera_params = []
        
        for _, row in calib_df.iterrows():
            # Only include cameras with sufficient data
            if pd.isna(row['FocalLength']) or pd.isna(row['ImageWidth']) or pd.isna(row['ImageHeight']):
                continue
                
            # Convert focal length to pixels if needed
            focal_length = float(row['FocalLength'])
            width = int(row['ImageWidth'])
            height = int(row['ImageHeight'])
            
            # Convert principal point
            cx = row['PrincipalPointX'] if not pd.isna(row['PrincipalPointX']) else width / 2
            cy = row['PrincipalPointY'] if not pd.isna(row['PrincipalPointY']) else height / 2
            
            # Create camera entry
            camera = {
                "camera_model": "brown",
                "width": width,
                "height": height,
                "focal_x": focal_length,
                "focal_y": focal_length,
                "c_x": cx,
                "c_y": cy,
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "k3": 0.0
            }
            
            camera_params.append(camera)
        
        # Write to file
        with open(camera_file, 'w') as f:
            json.dump(camera_params, f, indent=2)
            
        self.logger.info(f"Created OpenDroneMap camera parameters at {camera_file}")
    
    def _create_odm_gcp_file(self, group_dir, image_files):
        """Create GCP (Ground Control Points) file for OpenDroneMap if GPS data is available"""
        # Filter metadata for the group images
        filenames = [os.path.basename(f) for f in image_files]
        group_meta = self.metadata_df[self.metadata_df['Filename'].isin(filenames)]
        
        # Check if we have GPS data
        has_gps = (
            group_meta['Latitude'].notna().any() and
            group_meta['Longitude'].notna().any() and
            group_meta['Altitude'].notna().any()
        )
        
        if not has_gps:
            self.logger.warning("No GPS data available for OpenDroneMap GCP file")
            return
        
        # Create GCP file
        gcp_file = os.path.join(group_dir, 'gcp_list.txt')
        
        with open(gcp_file, 'w') as f:
            f.write("# EPSG:4326\n")  # WGS84
            f.write("# X/Longitude Y/Latitude Z/Altitude pixelRow pixelCol imageName\n")
            
            for idx, row in group_meta.iterrows():
                if pd.isna(row['Latitude']) or pd.isna(row['Longitude']) or pd.isna(row['Altitude']):
                    continue
                
                # Parse coordinates
                lon = self._parse_dms_to_decimal(row['Longitude'])
                lat = self._parse_dms_to_decimal(row['Latitude'])
                
                # Parse altitude
                alt_str = str(row['Altitude'])
                alt = float(alt_str.replace(' m', ''))
                
                # We don't have pixel coordinates for GPS center, so use image center
                if not pd.isna(row['ImageWidth']) and not pd.isna(row['ImageHeight']):
                    width = int(row['ImageWidth'])
                    height = int(row['ImageHeight'])
                    pixel_row = height / 2
                    pixel_col = width / 2
                else:
                    # Default values if image dimensions are unknown
                    pixel_row = 1000
                    pixel_col = 1000
                
                # Write GCP entry
                f.write(f"{lon} {lat} {alt} {pixel_row} {pixel_col} {row['Filename']}\n")
        
        self.logger.info(f"Created OpenDroneMap GCP file at {gcp_file}")
    
    def prepare_metashape_data(self):
        """
        Prepare data for Agisoft Metashape reconstruction
        """
        if self.image_groups is None:
            self.group_images()
            
        self.logger.info("Preparing data for Metashape reconstruction...")
        
        for group_name, image_files in self.image_groups.items():
            # Create group directory
            group_dir = os.path.join(self.output_dir, 'metashape', group_name)
            images_dir = os.path.join(group_dir, 'images')
            
            os.makedirs(images_dir, exist_ok=True)
            
            # Copy images to group directory
            for img_file in tqdm(image_files, desc=f"Copying images for {group_name}"):
                if os.path.exists(img_file):
                    shutil.copy2(img_file, images_dir)
            
            # Generate camera calibration file
            self.extractor.generate_camera_file(output_format="metashape")
            
            # Copy calibration file to group directory
            calib_src = os.path.join(self.output_dir, 'metadata', 'cameras.xml')
            calib_dst = os.path.join(group_dir, 'cameras.xml')
            
            if os.path.exists(calib_src):
                shutil.copy2(calib_src, calib_dst)
            
            # Create reference file for camera positions
            self._create_metashape_camera_reference(group_dir, image_files)
            
            # Create Python script for Metashape automation
            self._create_metashape_script(group_dir)
    
    def _create_metashape_camera_reference(self, group_dir, image_files):
        """Create camera reference file for Metashape"""
        # Filter metadata for the group images
        filenames = [os.path.basename(f) for f in image_files]
        group_meta = self.metadata_df[self.metadata_df['Filename'].isin(filenames)]
        
        # Check if we have position/orientation data
        has_position = (
            group_meta['Latitude'].notna().any() and
            group_meta['Longitude'].notna().any() and
            group_meta['Altitude'].notna().any()
        )
        
        has_orientation = (
            group_meta['Phi (Pitch)'].notna().any() and
            group_meta['Kappa (Yaw)'].notna().any() and
            group_meta['Omega (Roll)'].notna().any()
        )
        
        if not has_position and not has_orientation:
            self.logger.warning("No position or orientation data available for
