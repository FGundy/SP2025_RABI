#!/usr/bin/env python3
"""
Test SfM Orientation Estimator

This script tests the core functions of the SfM orientation estimator module.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.append(src_dir)

# Import our modules
from preprocessing.sfm_orientation_estimator import (
    group_images_by_flight,
    propagate_within_triplets,
    detect_flight_pattern,
    parse_dms_to_decimal,
    calculate_bearing,
    analyze_flight_pattern_for_initial_estimates,
    has_colmap
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_sfm_orientation")

def create_test_data():
    """Create a sample dataset for testing"""
    # Create timestamps with a gap to test flight grouping
    timestamps = [
        "2022-09-29 16:27:28",
        "2022-09-29 16:27:30",
        "2022-09-29 16:27:32",
        # 3-minute gap
        "2022-09-29 16:30:32",
        "2022-09-29 16:30:34",
        "2022-09-29 16:30:36"
    ]
    
    # Create test data
    data = []
    
    # First flight - orbit pattern
    data.append({
        "Filename": "MAX_0001.JPG",
        "SourceFile": "/mnt/d/DCIM/100MEDIA/MAX_0001.JPG",
        "Make": "Autel Robotics",
        "Model": "XT706",
        "Timestamp": timestamps[0],
        "Phi (Pitch)": 0.0,
        "Kappa (Yaw)": 90.0,
        "Omega (Roll)": 0.0,
        "Latitude": "40 deg 31' 27.88\" N",
        "Longitude": "74 deg 27' 36.16\" W",
        "Altitude": "14 m Above Sea Level",
        "IsRGB": True,
        "IsThermal": False
    })
    
    data.append({
        "Filename": "MIX_0001.JPG",
        "SourceFile": "/mnt/d/DCIM/100MEDIA/MIX_0001.JPG",
        "Make": "Autel Robotics",
        "Model": "XT706",
        "Timestamp": timestamps[0],
        "Phi (Pitch)": None,  # Missing orientation
        "Kappa (Yaw)": None,
        "Omega (Roll)": None,
        "Latitude": "40 deg 31' 27.88\" N",
        "Longitude": "74 deg 27' 36.16\" W",
        "Altitude": "14 m Above Sea Level",
        "IsRGB": True,
        "IsThermal": False
    })
    
    data.append({
        "Filename": "IRX_0001.JPG",
        "SourceFile": "/mnt/d/DCIM/100MEDIA/IRX_0001.JPG",
        "Make": "Autel Robotics",
        "Model": "XT706",
        "Timestamp": timestamps[0],
        "Phi (Pitch)": None,  # Missing orientation
        "Kappa (Yaw)": None,
        "Omega (Roll)": None,
        "Latitude": "40 deg 31' 27.88\" N",
        "Longitude": "74 deg 27' 36.16\" W",
        "Altitude": "14 m Above Sea Level",
        "IsRGB": False,
        "IsThermal": True
    })
    
    data.append({
        "Filename": "MAX_0002.JPG",
        "SourceFile": "/mnt/d/DCIM/100MEDIA/MAX_0002.JPG",
        "Make": "Autel Robotics",
        "Model": "XT706",
        "Timestamp": timestamps[1],
        "Phi (Pitch)": 0.0,
        "Kappa (Yaw)": 180.0,
        "Omega (Roll)": 0.0,
        "Latitude": "40 deg 31' 27.90\" N",
        "Longitude": "74 deg 27' 36.20\" W",
        "Altitude": "14 m Above Sea Level",
        "IsRGB": True,
        "IsThermal": False
    })
    
    data.append({
        "Filename": "MAX_0003.JPG",
        "SourceFile": "/mnt/d/DCIM/100MEDIA/MAX_0003.JPG",
        "Make": "Autel Robotics",
        "Model": "XT706",
        "Timestamp": timestamps[2],
        "Phi (Pitch)": 0.0,
        "Kappa (Yaw)": 270.0,
        "Omega (Roll)": 0.0,
        "Latitude": "40 deg 31' 27.92\" N",
        "Longitude": "74 deg 27' 36.24\" W",
        "Altitude": "14 m Above Sea Level",
        "IsRGB": True,
        "IsThermal": False
    })
    
    # Second flight - grid pattern
    data.append({
        "Filename": "MAX_0004.JPG",
        "SourceFile": "/mnt/d/DCIM/100MEDIA/MAX_0004.JPG",
        "Make": "Autel Robotics",
        "Model": "XT706",
        "Timestamp": timestamps[3],
        "Phi (Pitch)": 90.0,  # Pointing down
        "Kappa (Yaw)": 0.0,
        "Omega (Roll)": 0.0,
        "Latitude": "40 deg 30' 53.87\" N",
        "Longitude": "74 deg 26' 0.14\" W",
        "Altitude": "20 m Above Sea Level",
        "IsRGB": True,
        "IsThermal": False
    })
    
    data.append({
        "Filename": "MAX_0005.JPG",
        "SourceFile": "/mnt/d/DCIM/100MEDIA/MAX_0005.JPG",
        "Make": "Autel Robotics",
        "Model": "XT706",
        "Timestamp": timestamps[4],
        "Phi (Pitch)": None,  # Missing orientation
        "Kappa (Yaw)": None,
        "Omega (Roll)": None,
        "Latitude": "40 deg 30' 53.97\" N",
        "Longitude": "74 deg 26' 0.24\" W",
        "Altitude": "20 m Above Sea Level",
        "IsRGB": True,
        "IsThermal": False
    })
    
    data.append({
        "Filename": "MAX_0006.JPG",
        "SourceFile": "/mnt/d/DCIM/100MEDIA/MAX_0006.JPG",
        "Make": "Autel Robotics",
        "Model": "XT706",
        "Timestamp": timestamps[5],
        "Phi (Pitch)": None,  # Missing orientation
        "Kappa (Yaw)": None,
        "Omega (Roll)": None,
        "Latitude": "40 deg 30' 54.07\" N",
        "Longitude": "74 deg 26' 0.34\" W",
        "Altitude": "20 m Above Sea Level",
        "IsRGB": True,
        "IsThermal": False
    })
    
    return pd.DataFrame(data)

def test_flight_grouping():
    """Test grouping images by flight"""
    logger.info("Testing flight grouping...")
    
    # Create test data
    test_df = create_test_data()
    
    # Group by flight
    grouped_df = group_images_by_flight(test_df)
    
    # Check results
    flight_counts = grouped_df['FlightGroup'].value_counts()
    logger.info(f"Flight groups: {dict(flight_counts)}")
    
    # We should have 2 flight groups
    assert len(flight_counts) == 2, f"Expected 2 flight groups, got {len(flight_counts)}"
    
    # First group should have 5 images
    assert flight_counts[0] == 5, f"Expected 5 images in first flight, got {flight_counts[0]}"
    
    # Second group should have 3 images
    assert flight_counts[1] == 3, f"Expected 3 images in second flight, got {flight_counts[1]}"
    
    logger.info("Flight grouping test passed ✓")
    return grouped_df

def test_triplet_propagation():
    """Test propagating orientation within triplets"""
    logger.info("Testing triplet propagation...")
    
    # Create test data
    test_df = create_test_data()
    
    # Before propagation
    before_count = (
        test_df['Phi (Pitch)'].notna() & 
        test_df['Kappa (Yaw)'].notna() & 
        test_df['Omega (Roll)'].notna()
    ).sum()
    
    logger.info(f"Before propagation: {before_count} images have orientation data")
    
    # Propagate within triplets
    propagated_df = propagate_within_triplets(test_df)
    
    # After propagation
    after_count = (
        propagated_df['Phi (Pitch)'].notna() & 
        propagated_df['Kappa (Yaw)'].notna() & 
        propagated_df['Omega (Roll)'].notna()
    ).sum()
    
    logger.info(f"After propagation: {after_count} images have orientation data")
    logger.info(f"Propagated to {after_count - before_count} images")
    
    # Check that MIX_0001.JPG and IRX_0001.JPG now have orientation data from MAX_0001.JPG
    mix_row = propagated_df[propagated_df['Filename'] == 'MIX_0001.JPG']
    irx_row = propagated_df[propagated_df['Filename'] == 'IRX_0001.JPG']
    
    assert not mix_row['Phi (Pitch)'].isna().all(), "MIX_0001.JPG should have pitch data after propagation"
    assert not irx_row['Phi (Pitch)'].isna().all(), "IRX_0001.JPG should have pitch data after propagation"
    
    logger.info("Triplet propagation test passed ✓")
    return propagated_df

def test_flight_pattern_detection():
    """Test flight pattern detection"""
    logger.info("Testing flight pattern detection...")
    
    # Create test data
    test_df = create_test_data()
    
    # Group by flight
    grouped_df = group_images_by_flight(test_df)
    
    # Test on first flight group (orbit pattern)
    flight1 = grouped_df[grouped_df['FlightGroup'] == 0]
    pattern1 = detect_flight_pattern(flight1)
    logger.info(f"Detected pattern for flight 1: {pattern1}")
    
    # Test on second flight group (grid pattern)
    flight2 = grouped_df[grouped_df['FlightGroup'] == 1]
    pattern2 = detect_flight_pattern(flight2)
    logger.info(f"Detected pattern for flight 2: {pattern2}")
    
    logger.info("Flight pattern detection test completed")
    return pattern1, pattern2

def test_coordinate_parsing():
    """Test parsing coordinates"""
    logger.info("Testing coordinate parsing...")
    
    # Test cases
    test_cases = [
        ("40 deg 31' 27.88\" N", 40.52441111111111),
        ("74 deg 27' 36.16\" W", -74.46004444444445),
        ("40.5244 N", 40.5244),
        ("74.4600 W", -74.46),
        ("40°31'27.88\"N", 40.52441111111111),
        ("-40.5244", -40.5244)
    ]
    
    for dms, expected in test_cases:
        result = parse_dms_to_decimal(dms)
        logger.info(f"Parsed '{dms}' to {result} (expected {expected})")
        assert abs(result - expected) < 0.0001, f"Parsing error: {dms} -> {result} != {expected}"
    
    logger.info("Coordinate parsing test passed ✓")

def test_bearing_calculation():
    """Test bearing calculation"""
    logger.info("Testing bearing calculation...")
    
    # Test cases (lat1, lon1, lat2, lon2, expected_bearing)
    test_cases = [
        (40.0, -74.0, 41.0, -74.0, 0.0),    # North
        (40.0, -74.0, 40.0, -73.0, 90.0),   # East
        (40.0, -74.0, 39.0, -74.0, 180.0),  # South
        (40.0, -74.0, 40.0, -75.0, 270.0)   # West
    ]
    
    for lat1, lon1, lat2, lon2, expected in test_cases:
        result = calculate_bearing(lat1, lon1, lat2, lon2)
        logger.info(f"Bearing from ({lat1}, {lon1}) to ({lat2}, {lon2}): {result}° (expected {expected}°)")
        assert abs(result - expected) < 0.1, f"Bearing calculation error: {result} != {expected}"
    
    logger.info("Bearing calculation test passed ✓")

def test_colmap_availability():
    """Test if COLMAP is available"""
    logger.info("Testing COLMAP availability...")
    
    is_available = has_colmap()
    if is_available:
        logger.info("COLMAP is available on this system ✓")
    else:
        logger.warning("COLMAP is NOT available on this system - SfM will use pattern-based estimates only")

def main():
    """Run all tests"""
    logger.info("Starting SfM orientation estimator tests")
    
    try:
        # Run tests
        grouped_df = test_flight_grouping()
        propagated_df = test_triplet_propagation()
        pattern1, pattern2 = test_flight_pattern_detection()
        test_coordinate_parsing()
        test_bearing_calculation()
        test_colmap_availability()
        
        # Overall summary
        logger.info("\nOverall test summary:")
        logger.info("- Flight grouping: PASSED")
        logger.info("- Triplet propagation: PASSED")
        logger.info("- Flight pattern detection: COMPLETED")
        logger.info("  - Flight 1 pattern: " + pattern1)
        logger.info("  - Flight 2 pattern: " + pattern2)
        logger.info("- Coordinate parsing: PASSED")
        logger.info("- Bearing calculation: PASSED")
        
        logger.info("\nAll tests completed successfully!")
        
    except AssertionError as e:
        logger.error(f"Test failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
