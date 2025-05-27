import os
import sys
import pandas as pd
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("path_check.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("path_check")

def check_paths(metadata_file):
    """
    Check if all source files in metadata exist
    
    Args:
        metadata_file: Path to metadata CSV file
    """
    # Load metadata
    logger.info(f"Loading metadata from {metadata_file}")
    metadata_df = pd.read_csv(metadata_file)
    logger.info(f"Loaded metadata for {len(metadata_df)} images")
    
    # Check if SourceFile column exists
    if 'SourceFile' not in metadata_df.columns:
        logger.error("Metadata does not have a 'SourceFile' column")
        return
    
    # Check if files exist
    missing_files = []
    unreadable_files = []
    valid_files = []
    
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Checking files"):
        source_file = row['SourceFile']
        
        if not os.path.exists(source_file):
            missing_files.append((idx, source_file))
            continue
            
        # Check if file is readable
        try:
            with open(source_file, 'rb') as f:
                # Just read a small header
                f.read(100)
            valid_files.append((idx, source_file))
        except Exception as e:
            unreadable_files.append((idx, source_file, str(e)))
    
    # Report results
    logger.info(f"Results: {len(valid_files)} valid, {len(missing_files)} missing, {len(unreadable_files)} unreadable")
    
    if missing_files:
        logger.info("Sample of missing files:")
        for idx, path in missing_files[:10]:
            logger.info(f"  - {path}")
    
    if unreadable_files:
        logger.info("Sample of unreadable files:")
        for idx, path, error in unreadable_files[:10]:
            logger.info(f"  - {path}: {error}")
    
    # Check if there's an IsRGB column
    if 'IsRGB' in metadata_df.columns:
        rgb_count = metadata_df['IsRGB'].sum()
        logger.info(f"Found {rgb_count} RGB images in metadata")
        
        # Check RGB files specifically
        rgb_missing = sum(1 for idx, path in missing_files if metadata_df.loc[idx, 'IsRGB'])
        rgb_unreadable = sum(1 for idx, path, _ in unreadable_files if metadata_df.loc[idx, 'IsRGB'])
        logger.info(f"Of RGB images: {rgb_missing} missing, {rgb_unreadable} unreadable")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python check_file_paths.py <metadata_csv>")
        return 1
    
    metadata_file = sys.argv[1]
    if not os.path.exists(metadata_file):
        print(f"Metadata file not found: {metadata_file}")
        return 1
    
    check_paths(metadata_file)
    return 0

if __name__ == "__main__":
    sys.exit(main())
