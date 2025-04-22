import os
import shutil
import pandas as pd
from tqdm import tqdm

# Configuration
metadata_file = '/home/fabio/University/research/SP2025_RABI/data/processed/orientations/attempt1/completed_metadata.csv'  # Adjust this path if needed
source_root = '/mnt/d/'  # Mounted microSD card path
dest_root = os.path.expanduser('~/University/research_datasets')

# Load metadata
metadata_df = pd.read_csv(metadata_file)

# Ensure FlightGroup column exists
if 'FlightGroup' not in metadata_df.columns:
    raise ValueError("Metadata file must contain 'FlightGroup' column.")

# Exclude MP4 files
metadata_df = metadata_df[~metadata_df['SourceFile'].str.lower().str.endswith('.mp4')]

# Copy files into flight group directories
for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Copying files"):
    source_file = row['SourceFile']
    flight_group = row['FlightGroup']

    if flight_group == -1:
        group_dir = os.path.join(dest_root, 'unassigned')
    else:
        group_dir = os.path.join(dest_root, f'flightgroup_{flight_group:02d}')

    os.makedirs(group_dir, exist_ok=True)

    # Construct destination path
    filename = os.path.basename(source_file)
    dest_path = os.path.join(group_dir, filename)

    # Ensure source file path is correctly prefixed
    if not source_file.startswith(source_root):
        source_file = os.path.join(source_root, source_file.lstrip('/'))

    # Copy file if it exists
    if os.path.exists(source_file):
        shutil.copy2(source_file, dest_path)
    else:
        print(f"Warning: Source file not found {source_file}")
