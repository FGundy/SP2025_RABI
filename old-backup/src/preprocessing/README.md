# Preprocessing
This directory contains scripts for image preprocessing and alignment.




Project Context for Camera Orientation Estimation
Project Overview: SP2025_RABI (Research on Autonomous Building Inspection)
This project focuses on building 3D models from drone imagery (both RGB and thermal) using photogrammetry. A key challenge is that many of the 4,465 drone images lack complete metadata, particularly camera orientation parameters (Phi/Pitch, Kappa/Yaw, Omega/Roll) that are essential for accurate 3D reconstruction.
Current Repository Structure
CopySP2025_RABI/
├── data/
│   ├── processed/
│   │   └── metadata/
│   │       ├── basic_metadata.csv
│   │       └── camera_calibration.csv
│   └── raw/
├── src/
│   ├── metadata/
│   │   └── enhanced_metadata_extractor.py
│   ├── preprocessing/
│   │   ├── sfm_preprocessor.py
│   │   └── sfm_orientation_estimator.py
│   ├── reconstruction/
│   └── thermal/
├── tests/
├── process_metadata.py
└── process_orientations.py
Current Implementation Status
We've implemented a pipeline that:

Groups images by flight based on timestamp gaps
Associates related image types (MAX/RGB, MIX/RGB, IRX/thermal) captured at the same moment
Detects flight patterns (orbit, grid, linear)
Uses COLMAP for Structure-from-Motion to estimate missing orientations
Propagates orientation data within image triplets

The initial results from our test run show:

4,465 total images across the dataset
Initially only 14% (625 images) have orientation data
22 flight groups identified based on timestamp gaps
COLMAP process appears to be very slow or potentially hanging

Technical Details

Drone Model: Autel Robotics with XT706 camera
Image Types: MAX (high-res RGB), MIX (low-res RGB), IRX (thermal)
Flight Patterns: Orbit mode (circular around buildings) and Grid mode (for roof inspection)
Hardware: System has NVIDIA RTX GPU available

Current Challenge
The COLMAP-based SfM process appears to be very slow or potentially hanging. We need to:

Diagnose why COLMAP is taking so long
Potentially optimize the pipeline for better performance
Consider alternative approaches if COLMAP isn't feasible
Complete the orientation estimation for the entire dataset

Any insights on optimizing our approach, debugging the COLMAP process, or alternative methods to estimate camera orientations would be extremely valuable.



It looks like your pipeline is working well! The output shows that the script is correctly:

Loading the metadata from 4,465 images
Identifying that only 14% (625 images) currently have orientation data
Grouping the images into 22 flight groups based on timestamp gaps
Starting the process of estimating missing orientations for each flight group
Beginning COLMAP SfM processing on the first flight group

I notice the flight groups look reasonable, with several large groups containing hundreds of images each. These likely correspond to your different drone flights for building inspection.
The fact that flight group -1 has 82 images suggests there are 82 images without timestamp information. This is expected and your code is handling these appropriately.
The pipeline is now running COLMAP on each flight group. This process will take some time to complete, as COLMAP needs to:

Extract features from all images
Match features between images
Build a 3D reconstruction
Extract camera orientations



