import subprocess
import json
import os
import pandas as pd
from datetime import datetime
import numpy as np
from tqdm import tqdm

class EnhancedMetadataExtractor:
    """
    Enhanced metadata extractor class that provides robust extraction of 
    photogrammetry-critical metadata from drone imagery
    """
    
    # Camera models and their default parameters
    CAMERA_DEFAULTS = {
        "XT706": {
            "make": "Autel Robotics",
            "model": "XT706",
            "sensor_width": 7.4,  # mm
            "sensor_height": 5.55,  # mm
            "principal_point_x": 3.84,  # mm
            "principal_point_y": 3.072,  # mm
            "focal_length": 14.0,  # mm
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],  # k1, k2, p1, p2, k3
            "pixel_size": 0.012,  # mm
        }
    }

    def __init__(self, base_path, output_dir="./processed"):
        """
        Initialize the metadata extractor
        
        Args:
            base_path: Base path to search for images
            output_dir: Directory to save processed metadata
        """
        self.base_path = base_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metadata storage
        self.metadata_df = None
        self.full_metadata = {}
        
    def _run_exiftool(self, cmd_args):
        """
        Run ExifTool with provided arguments and return result
        
        Args:
            cmd_args: List of command line arguments for ExifTool
            
        Returns:
            Parsed JSON result from ExifTool
        """
        cmd = ["exiftool", "-json"] + cmd_args
        
        print(f"Running ExifTool command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error executing ExifTool: {result.stderr}")
            return None
        
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            print("Failed to parse ExifTool output")
            return None
    
    def extract_basic_metadata(self):
        """
        Extract basic metadata from all images in the base path
        
        Returns:
            DataFrame with basic metadata
        """
        print(f"Extracting basic metadata from {self.base_path}")
        
        # Extract all key photogrammetry tags
        cmd_args = [
            "-r",  # Recursive
            # Camera identification
            "-Make", "-Model", "-Software",
            # Location
            "-GPSLatitude", "-GPSLongitude", "-GPSAltitude", 
            "-GPSAltitudeRef", "-GPSPosition",
            # Camera orientation
            "-XMP-Camera:Pitch", "-XMP-Camera:Yaw", "-XMP-Camera:Roll",
            # Internal camera parameters
            "-FocalLength", "-FocalLengthIn35mmFormat",
            "-XMP-Camera:PrincipalPoint", "-XMP-Camera:PerspectiveFocalLength",
            "-XMP-Camera:PerspectiveDistortion",
            # Image dimensions
            "-ImageWidth", "-ImageHeight",
            # Time information
            "-CreateDate", "-DateTimeOriginal",
            # File information
            "-FileName", "-Directory",
            self.base_path
        ]
        
        metadata = self._run_exiftool(cmd_args)
        
        if not metadata:
            return None
            
        print(f"Found metadata for {len(metadata)} files")
        
        # Process into DataFrame
        data_rows = []
        
        for img_data in tqdm(metadata, desc="Processing metadata entries"):
            filename = os.path.basename(img_data.get("SourceFile", "Unknown"))
            source_file = img_data.get("SourceFile", "Unknown")
            
            # Extract timestamps
            timestamp = None
            if "DateTimeOriginal" in img_data:
                try:
                    timestamp = datetime.strptime(img_data["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass
            elif "CreateDate" in img_data:
                try:
                    timestamp = datetime.strptime(img_data["CreateDate"], "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass
            
            # Check if this is an RGB or thermal image
            # This is a simplified detection - might need adjustment for your specific case
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
                "IsThermal": is_thermal,
                "HasFullMetadata": self._has_full_metadata(img_data)
            }
            
            data_rows.append(row)
        
        # Create DataFrame
        self.metadata_df = pd.DataFrame(data_rows)
        
        # Save basic metadata to CSV
        output_file = os.path.join(self.output_dir, "basic_metadata.csv")
        self.metadata_df.to_csv(output_file, index=False)
        print(f"Basic metadata saved to {output_file}")
        
        return self.metadata_df
    
    def _has_full_metadata(self, img_data):
        """
        Check if an image has all the essential metadata for photogrammetry
        
        Args:
            img_data: Image metadata dictionary
            
        Returns:
            Boolean indicating whether image has full metadata
        """
        essential_fields = [
            "Pitch", "Yaw", "Roll",  # Camera orientation
            "GPSLatitude", "GPSLongitude", "GPSAltitude",  # Position
            "FocalLength"  # Internal parameter
        ]
        
        return all(field in img_data for field in essential_fields)
    
    def extract_full_metadata(self, max_images=None):
        """
        Extract full metadata from all images and store for reference
        
        Args:
            max_images: Maximum number of images to process (for testing)
            
        Returns:
            Dictionary mapping filenames to full metadata
        """
        if self.metadata_df is None:
            self.extract_basic_metadata()
        
        print("Extracting full metadata for all images...")
        
        # Get list of all image files
        image_files = self.metadata_df["SourceFile"].tolist()
        
        if max_images:
            image_files = image_files[:max_images]
        
        # Process each image
        for img_file in tqdm(image_files, desc="Extracting full metadata"):
            cmd_args = [
                "-a",  # All tags
                "-u",  # Include unknown tags
                "-g",  # Group tags by category
                img_file
            ]
            
            metadata = self._run_exiftool(cmd_args)
            
            if metadata and len(metadata) > 0:
                filename = os.path.basename(img_file)
                self.full_metadata[filename] = metadata[0]
        
        # Save full metadata to JSON
        output_file = os.path.join(self.output_dir, "full_metadata.json")
        with open(output_file, "w") as f:
            json.dump(self.full_metadata, f, indent=2)
        
        print(f"Full metadata saved to {output_file}")
        
        return self.full_metadata
    
    def extract_camera_calibration(self):
        """
        Extract or estimate camera calibration parameters
        
        Returns:
            DataFrame with camera calibration parameters
        """
        if self.metadata_df is None:
            self.extract_basic_metadata()
        
        print("Extracting camera calibration parameters...")
        
        # Group by camera model
        model_groups = self.metadata_df.groupby("Model")
        
        calibration_data = []
        
        for model, group in model_groups:
            # Check if we have default parameters for this model
            default_params = self.CAMERA_DEFAULTS.get(model)
            
            # Find images with full calibration data
            has_principal_point = group["PrincipalPoint"].notna()
            has_focal_length = group["FocalLength"].notna()
            
            # Sample image with good data
            good_samples = group[has_principal_point & has_focal_length]
            
            if not good_samples.empty:
                sample = good_samples.iloc[0]
                
                # Parse principal point
                principal_point = sample["PrincipalPoint"]
                if isinstance(principal_point, str) and "," in principal_point:
                    pp_x, pp_y = map(float, principal_point.split(","))
                else:
                    pp_x, pp_y = None, None
                
                # Extract focal length
                focal_length = sample["FocalLength"]
                if isinstance(focal_length, str) and "mm" in focal_length:
                    focal_length = float(focal_length.replace(" mm", ""))
                
                calibration_data.append({
                    "Model": model,
                    "Source": "Extracted from metadata",
                    "FocalLength": focal_length,
                    "PrincipalPointX": pp_x,
                    "PrincipalPointY": pp_y,
                    "ImageWidth": sample["ImageWidth"],
                    "ImageHeight": sample["ImageHeight"],
                    "Confidence": "High"
                })
            elif default_params:
                # Use default parameters
                calibration_data.append({
                    "Model": model,
                    "Source": "Default parameters",
                    "FocalLength": default_params["focal_length"],
                    "PrincipalPointX": default_params["principal_point_x"],
                    "PrincipalPointY": default_params["principal_point_y"],
                    "SensorWidth": default_params["sensor_width"],
                    "SensorHeight": default_params["sensor_height"],
                    "PixelSize": default_params["pixel_size"],
                    "Confidence": "Medium"
                })
            else:
                # No calibration data available
                print(f"Warning: No calibration data available for model {model}")
                
                # Take best guess from available data
                sample = group.iloc[0]
                width = sample["ImageWidth"]
                height = sample["ImageHeight"]
                
                # Estimate principal point (image center)
                pp_x = width / 2 if width else None
                pp_y = height / 2 if height else None
                
                # Estimate focal length if available
                focal_length = None
                if sample["FocalLength"] and isinstance(sample["FocalLength"], str) and "mm" in sample["FocalLength"]:
                    focal_length = float(sample["FocalLength"].replace(" mm", ""))
                elif sample["FocalLength35mm"] and isinstance(sample["FocalLength35mm"], str) and "mm" in sample["FocalLength35mm"]:
                    # Rough estimate based on 35mm equivalent
                    focal_length = float(sample["FocalLength35mm"].replace(" mm", "")) / 5.5  # Approximation
                
                calibration_data.append({
                    "Model": model,
                    "Source": "Estimated",
                    "FocalLength": focal_length,
                    "PrincipalPointX": pp_x,
                    "PrincipalPointY": pp_y,
                    "ImageWidth": width,
                    "ImageHeight": height,
                    "Confidence": "Low"
                })
        
        # Create DataFrame
        calibration_df = pd.DataFrame(calibration_data)
        
        # Save calibration data to CSV
        output_file = os.path.join(self.output_dir, "camera_calibration.csv")
        calibration_df.to_csv(output_file, index=False)
        print(f"Camera calibration parameters saved to {output_file}")
        
        return calibration_df
    
    def infer_missing_orientation(self):
        """
        Infer missing orientation data based on temporal and spatial patterns
        
        Returns:
            Updated metadata DataFrame with inferred orientation
        """
        if self.metadata_df is None:
            self.extract_basic_metadata()
        
        print("Inferring missing orientation data...")
        
        # Sort by timestamp for sequential analysis
        if "Timestamp" in self.metadata_df.columns:
            df = self.metadata_df.sort_values("Timestamp").copy()
        else:
            df = self.metadata_df.copy()
        
        # Identify images with missing orientation
        missing_pitch = df["Phi (Pitch)"].isna()
        missing_yaw = df["Kappa (Yaw)"].isna()
        missing_roll = df["Omega (Roll)"].isna()
        
        missing_any = missing_pitch | missing_yaw | missing_roll
        
        print(f"Found {missing_any.sum()} images with missing orientation data")
        
        if missing_any.sum() == 0:
            print("No missing orientation data to infer")
            return self.metadata_df
        
        # Function to infer orientation based on nearby images
        def infer_from_neighbors(idx, column, window=5):
            """Infer value based on temporal neighbors"""
            lower_idx = max(0, idx - window)
            upper_idx = min(len(df), idx + window + 1)
            
            neighbors = df.iloc[lower_idx:upper_idx]
            valid_neighbors = neighbors[neighbors[column].notna()]
            
            if len(valid_neighbors) > 0:
                # Weight by proximity (closer = higher weight)
                distances = np.abs(valid_neighbors.index - idx)
                weights = 1 / (distances + 1)
                
                # Weighted average
                weighted_values = valid_neighbors[column] * weights
                return weighted_values.sum() / weights.sum()
            
            return None
        
        # Apply inference
        for idx, row in tqdm(df[missing_any].iterrows(), desc="Inferring orientation"):
            if missing_pitch[idx]:
                df.at[idx, "Phi (Pitch)"] = infer_from_neighbors(idx, "Phi (Pitch)")
                df.at[idx, "Phi_Confidence"] = "Inferred"
                
            if missing_yaw[idx]:
                df.at[idx, "Kappa (Yaw)"] = infer_from_neighbors(idx, "Kappa (Yaw)")
                df.at[idx, "Kappa_Confidence"] = "Inferred"
                
            if missing_roll[idx]:
                df.at[idx, "Omega (Roll)"] = infer_from_neighbors(idx, "Omega (Roll)")
                df.at[idx, "Omega_Confidence"] = "Inferred"
        
        # Update metadata DataFrame
        self.metadata_df = df
        
        # Save inferred metadata to CSV
        output_file = os.path.join(self.output_dir, "inferred_metadata.csv")
        self.metadata_df.to_csv(output_file, index=False)
        print(f"Metadata with inferred orientation saved to {output_file}")
        
        return self.metadata_df
    
    def prepare_photogrammetry_dataset(self):
        """
        Prepare a complete dataset ready for photogrammetry software
        
        Returns:
            DataFrame with complete photogrammetry parameters
        """
        # Ensure all processing steps have been completed
        if self.metadata_df is None:
            self.extract_basic_metadata()
        
        self.extract_camera_calibration()
        self.infer_missing_orientation()
        
        print("Preparing photogrammetry dataset...")
        
        # Create a copy of the metadata DataFrame
        photo_df = self.metadata_df.copy()
        
        # Add confidence columns if not already present
        if "Phi_Confidence" not in photo_df.columns:
            photo_df["Phi_Confidence"] = photo_df["Phi (Pitch)"].apply(
                lambda x: "Original" if pd.notna(x) else "Missing"
            )
        
        if "Kappa_Confidence" not in photo_df.columns:
            photo_df["Kappa_Confidence"] = photo_df["Kappa (Yaw)"].apply(
                lambda x: "Original" if pd.notna(x) else "Missing"
            )
            
        if "Omega_Confidence" not in photo_df.columns:
            photo_df["Omega_Confidence"] = photo_df["Omega (Roll)"].apply(
                lambda x: "Original" if pd.notna(x) else "Missing"
            )
        
        # Calculate overall quality score
        def calculate_quality(row):
            score = 0
            
            # Position data
            if pd.notna(row["Latitude"]) and pd.notna(row["Longitude"]):
                score += 30
            
            # Orientation data
            orientation_confidence = {
                "Original": 20,
                "Inferred": 10,
                "Missing": 0
            }
            
            score += orientation_confidence.get(row["Phi_Confidence"], 0)
            score += orientation_confidence.get(row["Kappa_Confidence"], 0)
            score += orientation_confidence.get(row["Omega_Confidence"], 0)
            
            # Camera parameters
            if pd.notna(row["FocalLength"]):
                score += 10
                
            if pd.notna(row["PrincipalPoint"]):
                score += 10
                
            # Categorize
            if score >= 80:
                return "Excellent"
            elif score >= 60:
                return "Good"
            elif score >= 40:
                return "Fair"
            elif score >= 20:
                return "Poor"
            else:
                return "Unusable"
        
        photo_df["QualityScore"] = photo_df.apply(calculate_quality, axis=1)
        
        # Filter out unusable images
        usable_df = photo_df[photo_df["QualityScore"] != "Unusable"]
        
        print(f"Final dataset contains {len(usable_df)} usable images:")
        print(usable_df["QualityScore"].value_counts())
        
        # Save photogrammetry dataset to CSV
        output_file = os.path.join(self.output_dir, "photogrammetry_dataset.csv")
        usable_df.to_csv(output_file, index=False)
        print(f"Photogrammetry dataset saved to {output_file}")
        
        return usable_df
    
    def generate_camera_file(self, output_format="colmap"):
        """
        Generate camera calibration file for photogrammetry software
        
        Args:
            output_format: Format of output file (colmap, metashape, etc.)
            
        Returns:
            Path to generated camera file
        """
        # Extract camera calibration if not already done
        calibration = self.extract_camera_calibration()
        
        print(f"Generating camera file in {output_format} format...")
        
        if output_format.lower() == "colmap":
            # COLMAP camera file format
            output_file = os.path.join(self.output_dir, "cameras.txt")
            
            with open(output_file, "w") as f:
                f.write("# Camera list with one line of data per camera:\n")
                f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
                
                for idx, row in calibration.iterrows():
                    camera_id = idx + 1
                    model = "SIMPLE_PINHOLE"  # Simplified model
                    width = int(row["ImageWidth"]) if pd.notna(row["ImageWidth"]) else 0
                    height = int(row["ImageHeight"]) if pd.notna(row["ImageHeight"]) else 0
                    
                    # Camera parameters: focal_length, cx, cy
                    focal = row["FocalLength"] if pd.notna(row["FocalLength"]) else 0
                    cx = row["PrincipalPointX"] if pd.notna(row["PrincipalPointX"]) else (width / 2)
                    cy = row["PrincipalPointY"] if pd.notna(row["PrincipalPointY"]) else (height / 2)
                    
                    f.write(f"{camera_id} {model} {width} {height} {focal} {cx} {cy}\n")
            
            print(f"COLMAP camera file saved to {output_file}")
            return output_file
            
        elif output_format.lower() == "metashape":
            # Agisoft Metashape XML format
            output_file = os.path.join(self.output_dir, "cameras.xml")
            
            with open(output_file, "w") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write('<calibration>\n')
                
                for idx, row in calibration.iterrows():
                    camera_id = idx + 1
                    width = int(row["ImageWidth"]) if pd.notna(row["ImageWidth"]) else 0
                    height = int(row["ImageHeight"]) if pd.notna(row["ImageHeight"]) else 0
                    
                    focal = row["FocalLength"] if pd.notna(row["FocalLength"]) else 0
                    cx = row["PrincipalPointX"] if pd.notna(row["PrincipalPointX"]) else (width / 2)
                    cy = row["PrincipalPointY"] if pd.notna(row["PrincipalPointY"]) else (height / 2)
                    
                    f.write(f'  <sensor id="{camera_id}" model="{row["Model"]}">\n')
                    f.write(f'    <resolution width="{width}" height="{height}"/>\n')
                    f.write(f'    <calibration>\n')
                    f.write(f'      <f>{focal}</f>\n')
                    f.write(f'      <cx>{cx}</cx>\n')
                    f.write(f'      <cy>{cy}</cy>\n')
                    f.write(f'      <k1>0.0</k1>\n')
                    f.write(f'      <k2>0.0</k2>\n')
                    f.write(f'      <p1>0.0</p1>\n')
                    f.write(f'      <p2>0.0</p2>\n')
                    f.write(f'    </calibration>\n')
                    f.write(f'  </sensor>\n')
                
                f.write('</calibration>\n')
            
            print(f"Metashape camera file saved to {output_file}")
            return output_file
            
        else:
            print(f"Unsupported output format: {output_format}")
            return None


# Example usage
if __name__ == "__main__":
    # Path to the mounted SD card
    sd_card_path = "/mnt/d/DCIM"
    
    # Create enhanced metadata extractor
    extractor = EnhancedMetadataExtractor(sd_card_path, "./data/processed")
    
    # Extract and process metadata
    extractor.extract_basic_metadata()
    extractor.extract_full_metadata(max_images=10)  # Limit for testing
    extractor.extract_camera_calibration()
    extractor.infer_missing_orientation()
    
    # Prepare final dataset for photogrammetry
    photo_dataset = extractor.prepare_photogrammetry_dataset()
    
    # Generate camera calibration file for COLMAP
    extractor.generate_camera_file(output_format="colmap")
