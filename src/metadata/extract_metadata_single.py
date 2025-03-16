import subprocess
import json
import os

def extract_metadata_single(image_path):
    """
    Extract all metadata from a single image and save to JSON
    
    Args:
        image_path: Path to the image file
    """
    print(f"Extracting metadata from: {image_path}")
    
    # Build command to extract all metadata using ExifTool
    cmd = [
        "exiftool",
        "-json",
        "-a",  # All tags
        "-u",  # Include unknown tags
        "-g",  # Group tags by category
        image_path
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run ExifTool command
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error executing ExifTool: {result.stderr}")
        return None
    
    # Parse JSON output
    try:
        metadata = json.loads(result.stdout)
        print(f"Successfully extracted metadata")
        
        # Save to JSON file
        output_file = "image_metadata.json"
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {output_file}")
        return metadata
    except json.JSONDecodeError:
        print("Failed to parse ExifTool output")
        return None

# Main execution
if __name__ == "__main__":
    # Path to the specific image
    image_path = "/mnt/d/DCIM/107MEDIA/MIX_2002.JPG"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist")
    else:
        # Extract metadata
        metadata = extract_metadata_single(image_path)
