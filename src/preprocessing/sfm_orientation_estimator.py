"""
SfM-based orientation estimator for drone imagery

This module provides functions to estimate camera orientations using Structure-from-Motion
techniques. It organizes images into flight groups, uses COLMAP for SfM processing, and
propagates orientation data to associated images.

It integrates with the existing metadata processing pipeline to complete missing
orientation data (Phi/Pitch, Kappa/Yaw, Omega/Roll) required for photogrammetry.
"""

import os
import subprocess
import pandas as pd
import numpy as np
import math
import logging
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import json
from .colmap_helpers import (
    run_colmap_sfm_optimized,
    run_colmap_sfm_with_enhanced_stability,  
    split_large_flight_group,
    MAX_IMAGES_PER_BATCH,
    COLMAP_TIMEOUT
)
from collections import defaultdict
import re 
import cv2

# Set up logging
logger = logging.getLogger(__name__)

def group_images_by_flight(metadata_df, time_gap_threshold=120):  # 120 seconds = 2 minutes
    """
    Group images into separate flights based on timestamp gaps
    
    Args:
        metadata_df (pandas.DataFrame): DataFrame with image metadata
        time_gap_threshold (int): Threshold in seconds to consider a new flight
        
    Returns:
        pandas.DataFrame: DataFrame with FlightGroup column added
    """
    logger.info("Grouping images by flight using time gaps...")
    
    # Convert timestamp to datetime if needed
    if 'Timestamp' in metadata_df.columns and not pd.api.types.is_datetime64_any_dtype(metadata_df['Timestamp']):
        metadata_df['Timestamp'] = pd.to_datetime(metadata_df['Timestamp'], errors='coerce')
    
    # Handle missing timestamps
    if metadata_df['Timestamp'].isna().all():
        logger.warning("No timestamp data available for flight grouping")
        metadata_df['FlightGroup'] = 0
        return metadata_df
    
    # Sort by timestamp
    sorted_df = metadata_df.dropna(subset=['Timestamp']).sort_values('Timestamp').reset_index(drop=True)
    
    # Initialize flight groups
    sorted_df['FlightGroup'] = 0
    current_group = 0
    
    # Group by time gaps
    for i in range(1, len(sorted_df)):
        time_diff = (sorted_df.iloc[i]['Timestamp'] - sorted_df.iloc[i-1]['Timestamp']).total_seconds()
        if time_diff > time_gap_threshold:
            # Found a gap, start new group
            current_group += 1
            logger.debug(f"Found time gap of {time_diff:.1f}s, starting flight group {current_group}")
        
        sorted_df.iloc[i, sorted_df.columns.get_loc('FlightGroup')] = current_group
    
    # Add flight group information to original dataframe
    flight_groups = sorted_df[['SourceFile', 'FlightGroup']]
    
    # For images with missing timestamps, assign to group -1
    metadata_df = metadata_df.merge(flight_groups, on='SourceFile', how='left')
    metadata_df['FlightGroup'] = metadata_df['FlightGroup'].fillna(-1).astype(int)
    
    # Log flight groups
    group_counts = metadata_df['FlightGroup'].value_counts().sort_index()
    logger.info(f"Found {len(group_counts)} flight groups")
    for group_id, count in group_counts.items():
        logger.info(f"  - Flight {group_id}: {count} images")
    
    return metadata_df

# def propagate_within_triplets(metadata_df):
#     """
#     Propagate orientation data between images in the same triplet
    
#     Args:
#         metadata_df (pandas.DataFrame): DataFrame with image metadata
        
#     Returns:
#         pandas.DataFrame: Updated DataFrame with propagated orientation
#     """
#     logger.info("Propagating orientation data within image triplets...")
    
#     # Extract sequence numbers from filenames
#     metadata_df['SequenceNum'] = metadata_df['Filename'].apply(
#         lambda x: x.split('_')[1].split('.')[0] if isinstance(x, str) and '_' in x and '.' in x else None
#     )
    
#     # Process each sequence group
#     propagated_count = 0
    
#     for seq_num, group in metadata_df.groupby('SequenceNum'):
#         if seq_num is None or pd.isna(seq_num):
#             continue
            
#         # Check if any image in the group has complete orientation data
#         has_orientation = (
#             group['Phi (Pitch)'].notna() & 
#             group['Kappa (Yaw)'].notna() & 
#             group['Omega (Roll)'].notna()
#         )
        
#         if not has_orientation.any():
#             continue
        
#         # Find images with complete orientation data
#         source_images = group[has_orientation]
        
#         if len(source_images) == 0:
#             continue
            
#         # Use first image with orientation data as source
#         source_img = source_images.iloc[0]
        
#         # Propagate to others in same triplet
#         for idx, row in group.iterrows():
#             if not has_orientation.loc[idx]:
#                 # Copy orientation data
#                 if pd.isna(metadata_df.at[idx, 'Phi (Pitch)']):
#                     metadata_df.at[idx, 'Phi (Pitch)'] = source_img['Phi (Pitch)']
#                 if pd.isna(metadata_df.at[idx, 'Kappa (Yaw)']):
#                     metadata_df.at[idx, 'Kappa (Yaw)'] = source_img['Kappa (Yaw)']
#                 if pd.isna(metadata_df.at[idx, 'Omega (Roll)']):
#                     metadata_df.at[idx, 'Omega (Roll)'] = source_img['Omega (Roll)']
                
#                 # Mark as propagated
#                 metadata_df.at[idx, 'Orientation_Source'] = 'Triplet_Propagation'
#                 propagated_count += 1
    
#     logger.info(f"Propagated orientation data to {propagated_count} images")
    
#     return metadata_df
def propagate_within_triplets(metadata_df):
    """
    Significantly enhanced triplet propagation to identify related images 
    through multiple detection methods
    """
    logger.info("Running enhanced triplet propagation...")
    
    # Track propagated count
    propagated_count = 0
    
    # APPROACH 1: FILENAME PATTERN MATCHING
    # More sophisticated pattern matching for various naming conventions
    pattern_groups = defaultdict(list)
    
    for idx, row in metadata_df.iterrows():
        filename = row['Filename'] if not pd.isna(row['Filename']) else ''
        
        # Try different pattern extractions
        patterns = []
        
        # Pattern type 1: DJI_0001_XXX.JPG -> "DJI_0001"
        match = re.search(r'(.+_\d+)_[^_]+\.[^.]+$', filename)
        if match:
            patterns.append(f"pattern1_{match.group(1)}")
            
        # Pattern type 2: IMG_20230101_120000_XXX.JPG -> "IMG_20230101_120000"
        match = re.search(r'(.+_\d+_\d+)_[^_]+\.[^.]+$', filename)
        if match:
            patterns.append(f"pattern2_{match.group(1)}")
            
        # Pattern type 3: Filename without extension and suffix
        base_name = os.path.splitext(filename)[0]
        if '_' in base_name:
            base_pattern = '_'.join(base_name.split('_')[:-1])
            patterns.append(f"pattern3_{base_pattern}")
        
        # Add all potential patterns
        for pattern in patterns:
            pattern_groups[pattern].append(idx)
    
    # APPROACH 2: TIMESTAMP-BASED GROUPING
    # Group images captured within a very short time window (e.g., 1 second)
    if 'Timestamp' in metadata_df.columns:
        # Convert timestamp to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(metadata_df['Timestamp']):
            metadata_df['Timestamp'] = pd.to_datetime(metadata_df['Timestamp'], errors='coerce')
        
        # Sort by timestamp
        sorted_df = metadata_df.dropna(subset=['Timestamp']).sort_values('Timestamp')
        
        # Group by time proximity (within 1 second)
        current_group = None
        current_time = None
        
        for idx, row in sorted_df.iterrows():
            if current_time is None:
                current_group = f"time_{len(pattern_groups)}"
                pattern_groups[current_group] = [idx]
                current_time = row['Timestamp']
            else:
                time_diff = (row['Timestamp'] - current_time).total_seconds()
                
                if time_diff <= 1.0:  # 1 second threshold
                    # Same capture moment, add to current group
                    pattern_groups[current_group].append(idx)
                else:
                    # New capture moment
                    current_group = f"time_{len(pattern_groups)}"
                    pattern_groups[current_group] = [idx]
                    current_time = row['Timestamp']
    
    # APPROACH 3: SPATIAL PROXIMITY + TIME
    # For images with GPS data, group by both spatial and temporal proximity
    if all(col in metadata_df.columns for col in ['Latitude', 'Longitude', 'Timestamp']):
        # Function to parse coordinates
        def parse_coords(lat, lon):
            try:
                if isinstance(lat, str) and isinstance(lon, str):
                    # Parse DMS strings if needed
                    return parse_dms_to_decimal(lat), parse_dms_to_decimal(lon)
                else:
                    return float(lat), float(lon)
            except (ValueError, TypeError):
                return None, None
        
        # Group images by spatial+temporal proximity
        spatial_groups = []
        
        # Get valid coordinates and timestamps
        coords_df = metadata_df.dropna(subset=['Latitude', 'Longitude', 'Timestamp']).copy()
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(coords_df['Timestamp']):
            coords_df['Timestamp'] = pd.to_datetime(coords_df['Timestamp'], errors='coerce')
        
        # Sort by timestamp
        coords_df = coords_df.sort_values('Timestamp')
        
        # Group by time windows (e.g., 5 seconds)
        time_windows = []
        current_window = []
        last_time = None
        
        for idx, row in coords_df.iterrows():
            if last_time is None:
                current_window = [idx]
                last_time = row['Timestamp']
            else:
                time_diff = (row['Timestamp'] - last_time).total_seconds()
                
                if time_diff <= 5.0:  # 5 second window
                    current_window.append(idx)
                else:
                    # Save current window and start new one
                    if len(current_window) > 1:
                        time_windows.append(current_window)
                    current_window = [idx]
                    last_time = row['Timestamp']
        
        # Add final window
        if len(current_window) > 1:
            time_windows.append(current_window)
        
        # Within each time window, look for spatial clusters
        for window_idx, window in enumerate(time_windows):
            if len(window) <= 1:
                continue
                
            # Process each time window for spatial proximity
            window_positions = []
            for idx in window:
                lat, lon = parse_coords(metadata_df.loc[idx, 'Latitude'], 
                                       metadata_df.loc[idx, 'Longitude'])
                if lat is not None and lon is not None:
                    window_positions.append((idx, lat, lon))
            
            # If we have at least 2 valid positions, check proximity
            if len(window_positions) >= 2:
                # Calculate distance matrix
                distances = {}
                for i, (idx1, lat1, lon1) in enumerate(window_positions):
                    for j, (idx2, lat2, lon2) in enumerate(window_positions[i+1:], i+1):
                        distance = haversine_distance(lat1, lon1, lat2, lon2)
                        distances[(idx1, idx2)] = distance
                
                # Group by very close proximity (e.g., within 2 meters)
                # Simplistic approach: if any pair is close, consider them part of same triplet
                proximity_threshold = 2.0  # meters
                close_pairs = [(idx1, idx2) for (idx1, idx2), dist in distances.items() 
                              if dist <= proximity_threshold]
                
                # Create a graph of close pairs
                graph = defaultdict(set)
                for idx1, idx2 in close_pairs:
                    graph[idx1].add(idx2)
                    graph[idx2].add(idx1)
                
                # Find connected components (potential triplets)
                visited = set()
                for idx in [idx for idx, _, _ in window_positions]:
                    if idx not in visited:
                        component = []
                        queue = [idx]
                        while queue:
                            current = queue.pop(0)
                            if current not in visited:
                                visited.add(current)
                                component.append(current)
                                queue.extend(graph[current] - visited)
                        
                        if len(component) >= 2:
                            group_name = f"spatial_{window_idx}_{len(pattern_groups)}"
                            pattern_groups[group_name] = component
    
    # Process all potential groups to propagate orientation
    logger.info(f"Found {len(pattern_groups)} potential triplet groups")
    
    for group_name, indices in pattern_groups.items():
        # Only consider groups with multiple images
        if len(indices) <= 1:
            continue
            
        # Find images with orientation data
        oriented_indices = []
        for idx in indices:
            try:
                if (not pd.isna(metadata_df.loc[idx, 'Phi (Pitch)']) and
                    not pd.isna(metadata_df.loc[idx, 'Kappa (Yaw)']) and
                    not pd.isna(metadata_df.loc[idx, 'Omega (Roll)'])):
                    oriented_indices.append(idx)
            except KeyError:
                # Handle any index errors
                continue
        
        # Skip if no oriented images
        if not oriented_indices:
            continue
            
        # Determine best source for orientation
        source_idx = oriented_indices[0]
        source_confidence = 'Low'
        
        # If we have confidence information, use the highest confidence source
        if 'Orientation_Confidence' in metadata_df.columns:
            for idx in oriented_indices:
                confidence = metadata_df.loc[idx, 'Orientation_Confidence']
                if confidence == 'High':
                    source_idx = idx
                    source_confidence = 'High'
                    break
                elif confidence == 'Medium' and source_confidence != 'High':
                    source_idx = idx
                    source_confidence = 'Medium'
        
        # Propagate orientation data
        source_pitch = metadata_df.loc[source_idx, 'Phi (Pitch)']
        source_yaw = metadata_df.loc[source_idx, 'Kappa (Yaw)']
        source_roll = metadata_df.loc[source_idx, 'Omega (Roll)']
        source_file = metadata_df.loc[source_idx, 'Filename']
        
        logger.debug(f"Using {source_file} as orientation source for group {group_name}")
        
        # Apply to all non-oriented images in this group
        for idx in indices:
            if idx not in oriented_indices:
                try:
                    metadata_df.loc[idx, 'Phi (Pitch)'] = source_pitch
                    metadata_df.loc[idx, 'Kappa (Yaw)'] = source_yaw
                    metadata_df.loc[idx, 'Omega (Roll)'] = source_roll
                    metadata_df.loc[idx, 'Orientation_Source'] = 'Triplet_Propagation'
                    
                    # Set confidence based on source and image similarity
                    if source_confidence == 'High':
                        metadata_df.loc[idx, 'Orientation_Confidence'] = 'Medium'
                    else:
                        metadata_df.loc[idx, 'Orientation_Confidence'] = 'Low'
                    
                    propagated_count += 1
                except KeyError:
                    # Handle any index errors
                    continue
    
    logger.info(f"Enhanced triplet propagation completed: propagated orientation to {propagated_count} images")
    return metadata_df

def detect_flight_pattern(flight_group):
    """
    Analyze GPS coordinates to detect flight pattern (orbit, grid, linear)
    
    Args:
        flight_group (pandas.DataFrame): DataFrame with flight images
        
    Returns:
        str: Detected pattern ('orbit', 'grid', 'linear', or 'unknown')
    """
    # Skip if no GPS data
    if flight_group['Latitude'].isna().all() or flight_group['Longitude'].isna().all():
        return 'unknown'
    
    # Extract lat/lon coordinates
    coords = flight_group[['Latitude', 'Longitude', 'Timestamp']].dropna().copy()
    
    # Convert string coordinates to decimal if needed
    for col in ['Latitude', 'Longitude']:
        if coords[col].dtype == 'object':
            coords[col] = coords[col].apply(parse_dms_to_decimal)
    
    # Not enough points for pattern detection
    if len(coords) < 5:
        return 'unknown'
    
    # Calculate centroid
    centroid = (coords['Latitude'].mean(), coords['Longitude'].mean())
    
    # Calculate distance to centroid for each point
    coords['distance_to_center'] = coords.apply(
        lambda row: haversine_distance(row['Latitude'], row['Longitude'], centroid[0], centroid[1]),
        axis=1
    )
    
    # Calculate mean distance and standard deviation
    mean_distance = coords['distance_to_center'].mean()
    std_distance = coords['distance_to_center'].std()
    
    # Sort by timestamp and calculate bearing changes
    coords = coords.sort_values('Timestamp')
    bearings = calculate_bearings(coords)
    bearing_changes = np.abs(np.diff(bearings))
    
    # Wrap around 360 degrees for bearing changes
    bearing_changes = np.minimum(bearing_changes, 360 - bearing_changes)
    
    # Count significant turns
    turns_90deg = np.sum((bearing_changes > 80) & (bearing_changes < 100))
    turns_180deg = np.sum((bearing_changes > 170) & (bearing_changes < 190))
    
    # Features for pattern detection
    cv_distance = std_distance / mean_distance if mean_distance > 0 else float('inf')  # Coefficient of variation
    bearing_change_mean = np.mean(bearing_changes)
    
    # Detect patterns
    # Orbit: Consistent distance to center, gradual bearing changes
    if cv_distance < 0.15 and bearing_change_mean < 25:
        return 'orbit'
    # Grid: Many 90-degree turns, variable distance from center
    elif turns_90deg > 1 and turns_180deg > 0:
        return 'grid'
    # Linear: Few sharp turns, variable distance
    else:
        return 'linear'

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points in meters
    
    Args:
        lat1, lon1: Coordinates of first point (decimal degrees)
        lat2, lon2: Coordinates of second point (decimal degrees)
        
    Returns:
        float: Distance in meters
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Earth radius in meters
    
    return c * r

def parse_dms_to_decimal(dms_str):
    """
    Convert DMS (Degrees, Minutes, Seconds) string to decimal degrees
    
    Args:
        dms_str: DMS coordinate string like "40 deg 31' 27.88\" N"
        
    Returns:
        float: Decimal degrees
    """
    if not isinstance(dms_str, str):
        return dms_str
        
    try:
        # Handle various formats
        
        # Check if it's already decimal
        if ' deg ' not in dms_str and '°' not in dms_str and "'" not in dms_str:
            parts = dms_str.strip().split()
            decimal = float(parts[0])
            
            # Apply sign based on direction
            if len(parts) > 1 and parts[1] in ['S', 'W']:
                decimal = -decimal
            
            return decimal
        
        # Handle DMS format
        direction = 1
        if any(cardinal in dms_str for cardinal in ['S', 'W']):
            direction = -1
            
        # Remove cardinal directions and other text
        for cardinal in ['N', 'S', 'E', 'W', 'deg', '°', '"', "'"]:
            dms_str = dms_str.replace(cardinal, ' ')
            
        # Split into components
        parts = [float(p) for p in dms_str.strip().split() if p and p.replace('.', '', 1).isdigit()]
        
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
    except (ValueError, IndexError) as e:
        logger.warning(f"Could not parse DMS string: {dms_str}, error: {e}")
        return None

def calculate_bearings(coords_df):
    """
    Calculate bearings between consecutive points
    
    Args:
        coords_df: DataFrame with Latitude and Longitude columns
        
    Returns:
        numpy.ndarray: Array of bearings in degrees
    """
    bearings = []
    lats = coords_df['Latitude'].values
    lons = coords_df['Longitude'].values
    
    for i in range(len(lats) - 1):
        bearing = calculate_bearing(lats[i], lons[i], lats[i+1], lons[i+1])
        bearings.append(bearing)
    
    # Add last bearing (repeat the last one)
    if bearings:
        bearings.append(bearings[-1])
    else:
        bearings.append(0)
    
    return np.array(bearings)

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate bearing between two points in degrees
    
    Args:
        lat1, lon1: Coordinates of first point (decimal degrees)
        lat2, lon2: Coordinates of second point (decimal degrees)
        
    Returns:
        float: Bearing in degrees (0-360)
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Calculate bearing
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    bearing = math.atan2(y, x)
    
    # Convert to degrees
    bearing = math.degrees(bearing)
    
    # Normalize to 0-360
    bearing = (bearing + 360) % 360
    
    return bearing

def analyze_flight_pattern_for_initial_estimates(flight_group):
    """
    Analyze flight pattern to provide initial orientation estimates
    
    Args:
        flight_group: DataFrame with flight group images
        
    Returns:
        tuple: (DataFrame with initial estimates, pattern string)
    """
    # Detect flight pattern
    pattern = detect_flight_pattern(flight_group)
    logger.info(f"Detected flight pattern: {pattern}")
    
    # Use pattern for initial orientation estimates
    initial_estimates = flight_group.copy()
    
    if pattern == 'orbit':
        # For orbit, cameras point roughly toward center
        # Find centroid
        coords = flight_group[['Latitude', 'Longitude']].dropna().copy()
        
        # Convert string coordinates to decimal if needed
        for col in ['Latitude', 'Longitude']:
            if coords[col].dtype == 'object':
                coords[col] = coords[col].apply(parse_dms_to_decimal)
        
        centroid = (coords['Latitude'].mean(), coords['Longitude'].mean())
        
        # Estimate camera orientation for each image
        for idx, row in initial_estimates.iterrows():
            if pd.isna(row['Latitude']) or pd.isna(row['Longitude']):
                continue
                
            lat = parse_dms_to_decimal(row['Latitude'])
            lon = parse_dms_to_decimal(row['Longitude'])
            
            # Calculate bearing to center (this will be camera yaw)
            bearing_to_center = calculate_bearing(lat, lon, centroid[0], centroid[1])
            
            # For orbit pattern, point camera toward center
            if pd.isna(row['Kappa (Yaw)']):
                initial_estimates.at[idx, 'Kappa (Yaw)'] = bearing_to_center
                initial_estimates.at[idx, 'Initial_Estimate'] = True
                
            if pd.isna(row['Phi (Pitch)']):
                # For orbit, assume horizontal view (0 degrees pitch)
                initial_estimates.at[idx, 'Phi (Pitch)'] = 0.0
                initial_estimates.at[idx, 'Initial_Estimate'] = True
                
            if pd.isna(row['Omega (Roll)']):
                initial_estimates.at[idx, 'Omega (Roll)'] = 0.0  # Level
                initial_estimates.at[idx, 'Initial_Estimate'] = True
    
    elif pattern == 'grid':
        # Find image sequence
        if 'Timestamp' in flight_group.columns and flight_group['Timestamp'].notna().any():
            sorted_group = flight_group.sort_values('Timestamp')
        else:
            sorted_group = flight_group
        
        # Calculate flight directions
        for idx, row in initial_estimates.iterrows():
            # For grid pattern, cameras typically point straight down
            if pd.isna(row['Phi (Pitch)']):
                initial_estimates.at[idx, 'Phi (Pitch)'] = 90.0  # Straight down
                initial_estimates.at[idx, 'Initial_Estimate'] = True
                
            if pd.isna(row['Omega (Roll)']):
                initial_estimates.at[idx, 'Omega (Roll)'] = 0.0  # Level
                initial_estimates.at[idx, 'Initial_Estimate'] = True
    
    elif pattern == 'linear':
        # Find image sequence
        if 'Timestamp' in flight_group.columns and flight_group['Timestamp'].notna().any():
            sorted_group = flight_group.sort_values('Timestamp')
        else:
            sorted_group = flight_group
        
        # Calculate bearings between consecutive points
        coords = sorted_group[['Latitude', 'Longitude']].dropna().copy()
        
        # Convert string coordinates to decimal if needed
        for col in ['Latitude', 'Longitude']:
            if coords[col].dtype == 'object':
                coords[col] = coords[col].apply(parse_dms_to_decimal)
        
        bearings = calculate_bearings(coords)
        
        # Apply bearings to sorted group
        sorted_group['Bearing'] = np.pad(bearings, (0, len(sorted_group) - len(bearings)), 'edge')
        
        # Merge bearings back to initial estimates
        bearing_map = dict(zip(sorted_group.index, sorted_group['Bearing']))
        
        for idx, row in initial_estimates.iterrows():
            # For linear flight, camera typically follows flight direction
            if pd.isna(row['Phi (Pitch)']):
                initial_estimates.at[idx, 'Phi (Pitch)'] = 75.0  # Slightly angled down
                initial_estimates.at[idx, 'Initial_Estimate'] = True
                
            if pd.isna(row['Kappa (Yaw)']) and idx in bearing_map:
                initial_estimates.at[idx, 'Kappa (Yaw)'] = bearing_map[idx]
                initial_estimates.at[idx, 'Initial_Estimate'] = True
                
            if pd.isna(row['Omega (Roll)']):
                initial_estimates.at[idx, 'Omega (Roll)'] = 0.0  # Level
                initial_estimates.at[idx, 'Initial_Estimate'] = True
    
    return initial_estimates, pattern

def has_colmap():
    """Check if COLMAP is available on the system"""
    try:
        subprocess.run(['colmap', '--help'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False

def run_colmap_sfm(flight_group, output_dir, use_gpu=True):
    """
    Run COLMAP SfM on a flight group to estimate camera orientations
    
    Args:
        flight_group: DataFrame with flight group images
        output_dir: Directory for COLMAP output
        use_gpu: Whether to use GPU for COLMAP processing
        
    Returns:
        DataFrame: Updated with orientation data from COLMAP
    """
    # Call the optimized version
    return run_colmap_sfm_optimized(flight_group, output_dir, use_gpu)

def process_metadata_with_sfm(metadata_df, output_dir, use_gpu=True, method='auto'):
    """
    Process all metadata with SfM-based orientation estimation
    
    Args:
        metadata_df: DataFrame with image metadata
        output_dir: Directory for output
        use_gpu: Whether to use GPU for COLMAP processing
        method: COLMAP implementation to use: 'auto', 'subprocess', 'pycolmap', 'hierarchical'
        
    Returns:
        DataFrame: With completed orientation data
    """
    import os
    import importlib.util
    from pathlib import Path
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Add IsRGB flag if not present
    if 'IsRGB' not in metadata_df.columns:
        # Assume thermal images have IRX in the filename
        metadata_df['IsRGB'] = ~metadata_df['Filename'].str.contains('IRX', case=False, na=False)
    
    # 1. Group images by flight
    logger.info("Grouping images by flight...")
    metadata_df = group_images_by_flight(metadata_df)
    
    # Check if pycolmap is available if requested
    pycolmap_available = False
    if method in ['auto', 'pycolmap', 'hierarchical']:
        pycolmap_spec = importlib.util.find_spec("pycolmap")
        if pycolmap_spec is not None:
            try:
                from .pycolmap_wrapper import has_pycolmap
                pycolmap_available = has_pycolmap()
            except ImportError:
                logger.warning("Failed to import pycolmap_wrapper")
    
    # Determine method to use based on availability and request
    if method == 'auto':
        if pycolmap_available:
            selected_method = 'pycolmap'
            logger.info("Using pycolmap for SfM processing (auto-selected)")
        else:
            selected_method = 'subprocess'
            logger.info("Using subprocess-based COLMAP for SfM processing (auto-selected)")
    else:
        selected_method = method
        if selected_method in ['pycolmap', 'hierarchical'] and not pycolmap_available:
            logger.warning(f"{selected_method} method requested but pycolmap not available, falling back to subprocess")
            selected_method = 'subprocess'
        logger.info(f"Using {selected_method} method for SfM processing (user-selected)")
    
    # Import appropriate functions based on selected method
    if selected_method == 'pycolmap':
        try:
            from .pycolmap_wrapper import run_pycolmap_sfm, run_pycolmap_sfm_with_enhanced_stability
        except ImportError:
            logger.error("Failed to import pycolmap wrapper, falling back to subprocess method")
            selected_method = 'subprocess'
    elif selected_method == 'hierarchical':
        try:
            from .pycolmap_wrapper import run_hierarchical_sfm
        except ImportError:
            logger.error("Failed to import hierarchical method, falling back to subprocess method")
            selected_method = 'subprocess'
    
    # Continue with your existing code for processing flight groups, but use the selected method
    all_results = []
    
    for group_id, group_df in metadata_df.groupby('FlightGroup'):
        logger.info(f"\nProcessing flight group {group_id} with {len(group_df)} images")
        
        # Skip very small groups
        if len(group_df) < 3:
            logger.warning(f"Skipping group {group_id}: too few images")
            all_results.append(group_df)
            continue
        
        # Create output directory for this flight
        flight_dir = Path(output_dir) / f"flight_{group_id}"
        os.makedirs(flight_dir, exist_ok=True)
        
        # Check if this is a large flight group that needs batch processing
        if len(group_df) > MAX_IMAGES_PER_BATCH:
            # Special handling for very large groups and hierarchical method
            if selected_method == 'hierarchical' and len(group_df) > MAX_IMAGES_PER_BATCH * 2:
                logger.info(f"Using hierarchical SfM for large flight group {group_id}")
                
                # Process group
                # 1. First, propagate within triplets
                processed_group = propagate_within_triplets(group_df)
                
                # 2. Analyze flight pattern for initial estimates
                initial_estimates, pattern = analyze_flight_pattern_for_initial_estimates(processed_group)
                
                # 3. Run hierarchical SfM
                sfm_results = run_hierarchical_sfm(initial_estimates, flight_dir, use_gpu)
                
                # 4. Validate and refine
                flight_results = validate_and_refine_orientations(sfm_results, pattern)
            else:
                # Standard batch processing for large groups
                # Split into batches
                batches = split_large_flight_group(group_df, MAX_IMAGES_PER_BATCH)
                logger.info(f"Split into {len(batches)} batches")
                
                # Process each batch
                batch_results = []
                
                for i, batch_df in enumerate(batches):
                    batch_dir = flight_dir / f"batch_{i}"
                    logger.info(f"Processing batch {i} with {len(batch_df)} images")
                    
                    # Check if this batch has enough RGB images
                    rgb_count = batch_df['IsRGB'].sum()
                    if rgb_count < 3:
                        logger.warning(f"Skipping batch {i}: not enough RGB images ({rgb_count})")
                        batch_results.append(batch_df)
                        continue
                    
                    # Process batch
                    # 1. First, propagate within triplets
                    processed_batch = propagate_within_triplets(batch_df)
                    
                    # 2. Analyze flight pattern for initial estimates
                    initial_estimates, pattern = analyze_flight_pattern_for_initial_estimates(processed_batch)
                    
                    # 3. Run SfM with selected method
                    if selected_method == 'pycolmap':
                        # Use pycolmap with enhanced stability for batch processing
                        sfm_results = run_pycolmap_sfm_with_enhanced_stability(initial_estimates, batch_dir, use_gpu)
                    else:
                        # Use subprocess method
                        sfm_results = run_colmap_sfm_with_enhanced_stability(initial_estimates, batch_dir, use_gpu)
                    
                    # 4. Validate and refine
                    final_batch = validate_and_refine_orientations(sfm_results, pattern)
                    batch_results.append(final_batch)
                
                # Combine batch results
                combined_results = pd.concat(batch_results)
                
                # Propagate within triplets for the combined results
                flight_results = propagate_within_triplets(combined_results)
        else:
            # Regular processing for normal-sized flight groups
            # 1. First, propagate within triplets
            processed_group = propagate_within_triplets(group_df)
            
            # 2. Analyze flight pattern for initial estimates
            initial_estimates, pattern = analyze_flight_pattern_for_initial_estimates(processed_group)
            
            # 3. Run SfM with selected method
            if selected_method == 'pycolmap':
                logger.info(f"Running pycolmap SfM for flight {group_id}...")
                sfm_results = run_pycolmap_sfm(initial_estimates, flight_dir, use_gpu)
            elif selected_method == 'hierarchical':
                logger.info(f"Running hierarchical SfM for flight {group_id}...")
                sfm_results = run_hierarchical_sfm(initial_estimates, flight_dir, use_gpu)
            else:
                # Use subprocess method
                if has_colmap():
                    logger.info(f"Running subprocess COLMAP SfM for flight {group_id}...")
                    sfm_results = run_colmap_sfm_optimized(initial_estimates, flight_dir, use_gpu)
                else:
                    logger.warning("COLMAP not available, using pattern-based estimates only")
                    sfm_results = initial_estimates
            
            # 4. Validate and refine
            flight_results = validate_and_refine_orientations(sfm_results, pattern)
        
        # Save flight results
        flight_output = flight_dir / f"flight_{group_id}_metadata.csv"
        flight_results.to_csv(flight_output, index=False)
        logger.info(f"Saved flight {group_id} results to {flight_output}")
        
        # Add to all results
        all_results.append(flight_results)
    
    # Combine all processed flight groups
    final_df = pd.concat(all_results)
    
    # Save the final metadata
    final_output = Path(output_dir) / "completed_metadata.csv"
    final_df.to_csv(final_output, index=False)
    logger.info(f"Saved combined results to {final_output}")
    
    # Report completion statistics
    has_orientation = (
        final_df['Phi (Pitch)'].notna() & 
        final_df['Kappa (Yaw)'].notna() & 
        final_df['Omega (Roll)'].notna()
    )
    
    orientation_count = has_orientation.sum()
    logger.info(f"Completed orientation for {orientation_count} of {len(final_df)} images ({orientation_count/len(final_df)*100:.1f}%)")
    
    # Report by source
    source_counts = final_df['Orientation_Source'].value_counts()
    logger.info("Orientation sources:")
    for source, count in source_counts.items():
        logger.info(f"  - {source}: {count} images")
        
    return final_df


def quaternion_to_euler(qw, qx, qy, qz):
    """
    Convert quaternion to Euler angles (degrees)
    
    Args:
        qw, qx, qy, qz: Quaternion components
        
    Returns:
        tuple: (pitch, yaw, roll) in degrees
    """
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

def validate_and_refine_orientations(orientation_df, pattern):
    """
    Validate and refine orientation estimates
    
    Args:
        orientation_df: DataFrame with orientation data
        pattern: Flight pattern ('orbit', 'grid', 'linear', 'unknown')
        
    Returns:
        DataFrame: Validated and refined orientations
    """
    logger.info("Validating and refining orientation estimates...")
    
    # Check how many images have orientation data
    has_orientation = (
        orientation_df['Phi (Pitch)'].notna() & 
        orientation_df['Kappa (Yaw)'].notna() & 
        orientation_df['Omega (Roll)'].notna()
    )
    
    orientation_count = has_orientation.sum()
    logger.info(f"{orientation_count} of {len(orientation_df)} images have orientation data ({orientation_count/len(orientation_df)*100:.1f}%)")
    
    # Add confidence scores
    orientation_df['Orientation_Confidence'] = 'Unknown'
    
    for idx, row in orientation_df.iterrows():
        if has_orientation.loc[idx]:
            if row.get('Orientation_Source') == 'COLMAP_SfM':
                orientation_df.at[idx, 'Orientation_Confidence'] = 'High'
            elif row.get('Orientation_Source') == 'Triplet_Propagation':
                # If source was from high confidence, propagate confidence
                source_confidence = 'Medium'
                orientation_df.at[idx, 'Orientation_Confidence'] = source_confidence
            elif row.get('Orientation_Source') == 'Original':
                orientation_df.at[idx, 'Orientation_Confidence'] = 'High'
            else:
                orientation_df.at[idx, 'Orientation_Confidence'] = 'Medium'
    
    # Validate orientation values based on flight pattern
    if pattern == 'grid':
        # For grid pattern, pitch should be close to 90 degrees (down)
        for idx, row in orientation_df.iterrows():
            if has_orientation.loc[idx]:
                pitch = row['Phi (Pitch)']
                # If pitch is wildly off from expected value, flag it
                if abs(pitch - 90) > 30:
                    logger.warning(f"Unexpected pitch value {pitch}° for grid pattern (expected ~90°)")
                    if row['Orientation_Confidence'] != 'High':
                        # Adjust the value for medium/low confidence estimates
                        orientation_df.at[idx, 'Phi (Pitch)'] = 90.0
                        orientation_df.at[idx, 'Orientation_Confidence'] = 'Adjusted'
    
    elif pattern == 'orbit':
        # For orbit pattern, check that cameras point toward center
        # This is more complex and would require geometric validation
        pass
    
    # Return refined orientations
    return orientation_df
