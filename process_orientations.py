#!/usr/bin/env python3
"""
Process Orientations - SfM-based orientation estimation for drone imagery

This script processes the metadata extracted by process_metadata.py to estimate
camera orientations (Phi/Pitch, Kappa/Yaw, Omega/Roll) using Structure-from-Motion
techniques with COLMAP.

Usage:
    python process_orientations.py [--input INPUT_CSV] [--output OUTPUT_DIR] [--cpu]

Arguments:
    --input: Path to input metadata CSV file (default: ./data/processed/metadata/basic_metadata.csv)
    --output: Path to output directory (default: ./data/processed/orientations)
    --cpu: Use CPU only (no GPU acceleration)
"""

import os
import argparse
import logging
import sys
import pandas as pd
from pathlib import Path
import cv2

# Add src directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
sys.path.append(src_dir)

# Import our modules
from preprocessing.sfm_orientation_estimator import process_metadata_with_sfm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("orientation_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("process_orientations")

def main():
    """Main function to process orientations"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process orientations using SfM")
    parser.add_argument('--input', type=str, 
                       default='./data/processed/metadata/basic_metadata.csv',
                       help='Path to input metadata CSV file')
    parser.add_argument('--output', type=str, 
                       default='./data/processed/orientations',
                       help='Path to output directory')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU only (no GPU acceleration)')
    parser.add_argument('--method', type=str, 
                       choices=['auto', 'subprocess', 'pycolmap', 'hierarchical'],
                       default='auto',
                       help='COLMAP implementation to use: auto (default), subprocess, pycolmap, or hierarchical')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load input metadata
    logger.info(f"Loading metadata from {args.input}")
    try:
        metadata_df = pd.read_csv(args.input)
        logger.info(f"Loaded metadata for {len(metadata_df)} images")
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return 1
    
    # Display metadata stats
    has_orientation = (
        metadata_df['Phi (Pitch)'].notna() & 
        metadata_df['Kappa (Yaw)'].notna() & 
        metadata_df['Omega (Roll)'].notna()
    )
    
    initial_orientation_count = has_orientation.sum()
    logger.info(f"Initial metadata: {initial_orientation_count} of {len(metadata_df)} images have orientation data ({initial_orientation_count/len(metadata_df)*100:.1f}%)")
    
    # Process metadata with SfM
    logger.info(f"Processing metadata with SfM using {args.method} method...")
    try:
        result_df = process_metadata_with_sfm(
            metadata_df, 
            args.output, 
            use_gpu=not args.cpu,
            method=args.method
        )
    except Exception as e:
        logger.error(f"Error processing orientations: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    # Save result
    output_file = os.path.join(args.output, "completed_metadata.csv")
    result_df.to_csv(output_file, index=False)
    logger.info(f"Saved completed metadata to {output_file}")
    
    # Final stats
    has_orientation = (
        result_df['Phi (Pitch)'].notna() & 
        result_df['Kappa (Yaw)'].notna() & 
        result_df['Omega (Roll)'].notna()
    )
    
    final_orientation_count = has_orientation.sum()
    logger.info(f"Final result: {final_orientation_count} of {len(result_df)} images have orientation data ({final_orientation_count/len(result_df)*100:.1f}%)")
    logger.info(f"Added orientation data for {final_orientation_count - initial_orientation_count} images")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
