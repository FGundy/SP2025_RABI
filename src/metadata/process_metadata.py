# src/metadata/process_metadata.py
import os
import sys
import subprocess
import json
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import time

def run_exiftool(directory, timeout=60):
    """Run ExifTool with timeout on a specific directory"""
    print(f"Processing directory: {directory}")
    
    cmd = [
        "exiftool",
        "-json",
        "-Make", "-Model", "-Software",
        "-GPSLatitude", "-GPSLongitude", "-GPSAltitude", 
        "-GPSAltitudeRef", "-GPSPosition",
        "-XMP-Camera:Pitch", "-XMP-Camera:Yaw", "-XMP-Camera:Roll",
        "-FocalLength", "-FocalLengthIn35mmFormat",
        "-XMP-Camera:PrincipalPoint", "-XMP-Camera:PerspectiveFocalLength",
        "-XMP-Camera:PerspectiveDistortion",
        "-ImageWidth", "-ImageHeight",
        "-CreateDate", "-DateTimeOriginal",
        "-FileName", "-Directory",
        directory
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        
        # Parse JSON output
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON output for {directory}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"Timeout expired processing {directory}")
        return None

def process_dcim_directories(base_path, output_dir):
    """Process each directory in DCIM separately"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all directories
    directories = [d for d in os.listdir(base_path) 
                  if os.path.isdir(os.path.join(base_path, d))]
    
    print(f"Found {len(directories)} directories to process")
    
    # Prepare to store all results
    all_metadata = []
    
    # Process each directory
    for directory in tqdm(directories, desc="Processing directories"):
        dir_path = os.path.join(base_path, directory)
        metadata = run_exiftool(dir_path)
        
        if metadata:
            print(f"Found metadata for {len(metadata)} files in {directory}")
            all_metadata.extend(metadata)
        
        # Save progress after each directory
        if all_metadata:
            temp_file = os.path.join(output_dir, f"metadata_partial_{len(all_metadata)}.json")
            with open(temp_file, "w") as f:
                json.dump(all_metadata, f)
    
    if not all_metadata:
        print("No metadata was extracted!")
        return None
    
    # Process into DataFrame
    data_rows = []
    
    for img_data in tqdm(all_metadata, desc="Processing metadata entries"):
        filename = os.path.basename(img_data.get("SourceFile", "Unknown"))
        source_file = img_data.get("SourceFile", "Unknown")
        
        # Extract timestamps
        timestamp = None
        if "DateTimeOriginal" in img_data:
            try:
                timestamp = datetime.strptime(img_data["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
            except (ValueError, TypeError):
                pass
        elif "CreateDate" in img_data:
            try:
                timestamp = datetime.strptime(img_data["CreateDate"], "%Y:%m:%d %H:%M:%S")
            except (ValueError, TypeError):
                pass
        
        # Check if this is an RGB or thermal image
        is_thermal = filename.startswith("IR") or "thermal" in filename.lower()
        
        # Extract data into structured format
        row = {
            "Filename": filename,
            "SourceFile": source_file,
            "Make": img_data.get("Make"),
            "Model": img_data.get("Model"),
            "Software": img_data.get("Software"),
            "Timestamp": timestamp,
            "Phi (Pitch)": img_data.get("Pitch"),
            "Kappa (Yaw)": img_data.get("Yaw"),
            "Omega (Roll)": img_data.get("Roll"),
            "Latitude": img_data.get("GPSLatitude"),
            "Longitude": img_data.get("GPSLongitude"),
            "Altitude": img_data.get("GPSAltitude"),
            "AltitudeRef": img_data.get("GPSAltitudeRef"),
            "FocalLength": img_data.get("FocalLength"),
            "FocalLength35mm": img_data.get("FocalLengthIn35mmFormat"),
            "PrincipalPoint": img_data.get("PrincipalPoint"),
            "PerspectiveFocalLength": img_data.get("PerspectiveFocalLength"),
            "PerspectiveDistortion": img_data.get("PerspectiveDistortion"),
            "ImageWidth": img_data.get("ImageWidth"),
            "ImageHeight": img_data.get("ImageHeight"),
            "IsRGB": not is_thermal,
            "IsThermal": is_thermal
        }
        
        data_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows)
    
    # Save metadata to CSV
    output_file = os.path.join(output_dir, "basic_metadata.csv")
    df.to_csv(output_file, index=False)
    print(f"Metadata saved to {output_file}")
    
    return df

if __name__ == "__main__":
    dcim_path = "/mnt/d/DCIM"
    output_dir = "./data/processed/metadata"
    
    # Process all directories
    df = process_dcim_directories(dcim_path, output_dir)
    
    if df is not None:
        print(f"Processed metadata for {len(df)} images")
