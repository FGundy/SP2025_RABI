import subprocess
import json
import os
import pandas as pd

def extract_gimbal_metadata(image_folder):
    """
    Extract gimbal orientation metadata from Autel drone images
    using ExifTool
    
    Args:
        image_folder: Path to folder containing drone images
    
    Returns:
        DataFrame with image names and gimbal orientation data
    """
    print(f"Searching for images in: {image_folder}")
    
    # Build command to extract specific tags using ExifTool
    cmd = [
        "exiftool",
        "-json",
        "-r",  # Recursive
        "-XMP-Camera:Pitch",
        "-XMP-Camera:Yaw", 
        "-XMP-Camera:Roll",
        "-GPS*",  # All GPS tags
        "-GPSLatitude",
        "-GPSLongitude", 
        "-GPSAltitude",
        image_folder
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run ExifTool command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error executing ExifTool: {result.stderr}")
        return None
    
    print(f"ExifTool executed successfully. Processing results...")
    
    # Parse JSON output
    try:
        metadata = json.loads(result.stdout)
        print(f"Found metadata for {len(metadata)} files")
    except json.JSONDecodeError:
        print("Failed to parse ExifTool output")
        return None
    
    # Create empty lists to store data
    data = []
    
    # Process each image's metadata
    for img_data in metadata:
        filename = os.path.basename(img_data.get("SourceFile", "Unknown"))
        source_file = img_data.get("SourceFile", "Unknown")
        
        # Extract gimbal data from XMP-Camera namespace
        # Map to Omega (Roll), Phi (Pitch), Kappa (Yaw)
        phi = img_data.get("Pitch")   # Phi = Pitch
        kappa = img_data.get("Yaw")   # Kappa = Yaw
        omega = img_data.get("Roll")  # Omega = Roll
        
        # Extract GPS data
        lat = img_data.get("GPSLatitude")
        lon = img_data.get("GPSLongitude")
        alt = img_data.get("GPSAltitude")
        
        # Add to dataset
        data.append({
            "Filename": filename,
            "SourceFile": source_file,
            "Phi (Pitch)": phi,
            "Kappa (Yaw)": kappa,
            "Omega (Roll)": omega,
            "Latitude": lat,
            "Longitude": lon,
            "Altitude": alt
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

# Example usage
if __name__ == "__main__":
    # Path to the mounted SD card
    sd_card_path = "/mnt/d/DCIM"
    
    # Extract gimbal metadata from all files
    results = extract_gimbal_metadata(sd_card_path)
    
    if results is not None:
        # Count files with gimbal data
        has_gimbal = results.dropna(subset=['Phi (Pitch)', 'Kappa (Yaw)', 'Omega (Roll)'], how='all').shape[0]
        total_files = len(results)
        
        print(f"\nSummary:")
        print(f"Total files processed: {total_files}")
        print(f"Files with gimbal orientation data: {has_gimbal}")
        
        # Save to CSV
        results.to_csv("autel_gimbal_metadata.csv", index=False)
        print("Results saved to autel_gimbal_metadata.csv")
        
        # Display first few rows
        print("\nSample of extracted data:")
        print(results.head())
