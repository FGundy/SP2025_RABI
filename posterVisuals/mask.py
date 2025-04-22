Gmail	Fabio Gunderson <fgunderson.jobs@gmail.com>
OpenSource Git Notebook Runs Part 2
Gunderson, Fabio [JRDUS] <FGunders@its.jnj.com>	Mon, Mar 31, 2025 at 11:31 AM
To: "fgunderson.jobs@gmail.com" <fgunderson.jobs@gmail.com>
# CELL 1: Imports, Configuration, and Data Loading Functions

 

### This cell contains the config parameters, and these methods:

### ### check_memory_useage, load_segmentation_json, find_annotated_frames

### ### extract_annotation_data, resolve_frame_directory, load_frame_directory

 

import os

import json

import numpy as np

import matplotlib.pyplot as plt

import cv2

import torch

from PIL import Image

import tempfile

import psutil

from typing import Dict, List, Tuple, Optional, Any, Union

from datetime import datetime

import pandas as pd

from tqdm.notebook import tqdm

import logging

import matplotlib

from sklearn.metrics import precision_score, recall_score, f1_score

import random

 

# Configure logging

logging.basicConfig(

    level=logging.INFO,

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'

)

logger = logging.getLogger("SAM2-Evaluation")

 

# Configuration

CONFIG = {

    # Paths

    "segmentation_json_path": "/domino/datasets/local/cv_endoscopy/fgunderson/sam2-results/analysis/sam2/notebooks/superset_batch5_segmentation_per_video.json",

    "sam2_checkpoint": "../checkpoints/sam2.1_hiera_base_plus.pt",  # Update with your path

    "model_cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",  # Update with your path

 

    # Video and frame range configuration - to be updated for each run

    "folder_id": None,  # Will be set in execution cell

    "frame_range": None,  # Will be set in execution cell as [start, end]

    "start_frame_index": None,  # Which annotated frame to use as starting point

   

    # Processing options

    "use_mask_input": False,  # If True, use contour-based mask initialization instead of bbox+points

    "separate_objects": True,  # If True, process each object with separate inference state

    "save_intermediates": True,  # Save intermediate masks for debugging

    "save_to_disk": False,  # If True, save masks to disk to reduce memory usage for long sequences

    "output_dir": "./sam2_evaluation_results",

    "device": "cuda" if torch.cuda.is_available() else "cpu",

   

    # Memory monitoring

    "memory_check_interval": 20,  # Check memory usage every X frames

    "max_memory_usage_percent": 90  # Warning threshold for memory usage

}

 

def check_memory_usage():

    """Monitor memory usage and log warnings if it gets too high."""

    memory_usage = psutil.virtual_memory().percent

    if memory_usage > CONFIG["max_memory_usage_percent"]:

        logger.warning(f"HIGH MEMORY USAGE: {memory_usage}% of RAM is being used!")

    return memory_usage

 

def load_segmentation_json(json_path: str) -> Dict:

    """Load the segmentation JSON file."""

    logger.info(f"Loading segmentation data from {json_path}")

    with open(json_path, 'r') as f:

        data = json.load(f)

    logger.info(f"Loaded segmentation data with {len(data)} videos")

    return data

 

def get_video_data(segmentation_data: Dict, folder_id: str) -> Tuple[Dict, Dict]:

    """Extract video data and frames data for a specific video."""

    if folder_id not in segmentation_data:

        raise ValueError(f"Folder ID {folder_id} not found in segmentation data")

       

    video_data = segmentation_data[folder_id]

    frames_data = video_data['frames']

   

    logger.info(f"Loaded data for video {folder_id} with {len(frames_data)} frames")

    return video_data, frames_data

 

def find_annotated_frames(frames_data: Dict, frame_range: List[int]) -> List[int]:

    """Find all annotated frames in the given range across all reviewers."""

    start_frame, end_frame = frame_range

    annotated_frames = []

   

    for frame_name, frame_data in frames_data.items():

        frame_number = frame_data['metadata']['frame_number']

       

        # Check if frame is in range

        if frame_number < start_frame or frame_number > end_frame:

            continue

           

        # Check if frame has any reviewer data with merged contours

        found_contours = False

        if 'reviewer_data' in frame_data:

            for reviewer_id, reviewer_data in frame_data['reviewer_data'].items():

                if ('experiment' in reviewer_data and

                    'segments' in reviewer_data['experiment'] and

                    len(reviewer_data['experiment']['segments']) > 0 and

                    'groundtruth' in reviewer_data and

                    ('merged_contours' in reviewer_data['groundtruth'] or

                     'original_contours' in reviewer_data['groundtruth'])):

                    found_contours = True

                    break

       

        if found_contours:

            annotated_frames.append(frame_number)

   

    annotated_frames.sort()

    logger.info(f"Found {len(annotated_frames)} annotated frames in range {frame_range}")

    return annotated_frames

 

def extract_annotation_data(frames_data: Dict, frame_number: int) -> List[Dict]:

    """Extract annotations (bboxes, points, contours) from any available reviewer for a specific frame."""

    annotations = []

   

    # Find the frame data

    frame_data = None

    for frame_name, data in frames_data.items():

        if data['metadata']['frame_number'] == frame_number:

            frame_data = data

            break

   

    if frame_data is None:

        logger.warning(f"Frame {frame_number} not found in frames data")

        return annotations

       

    # Check if reviewer data exists

    if 'reviewer_data' not in frame_data:

        logger.warning(f"No reviewer data for frame {frame_number}")

        return annotations

   

    # Try to find annotations from any reviewer (prioritize the one with most segments)

    best_reviewer_id = None

    most_segments = 0

   

    for reviewer_id, reviewer_data in frame_data['reviewer_data'].items():

        if ('experiment' in reviewer_data and

            'segments' in reviewer_data['experiment'] and

            len(reviewer_data['experiment']['segments']) > 0):

           

            # Check if this reviewer has more segments than previously found

            num_segments = len(reviewer_data['experiment']['segments'])

            if num_segments > most_segments:

                most_segments = num_segments

                best_reviewer_id = reviewer_id

   

    if best_reviewer_id is None:

        logger.warning(f"No segments found for frame {frame_number}")

        return annotations

   

    # Extract data from the best reviewer

    reviewer_data = frame_data['reviewer_data'][best_reviewer_id]

    segments = reviewer_data['experiment']['segments']

   

    # Extract contours

    contours = None

    if 'groundtruth' in reviewer_data and 'merged_contours' in reviewer_data['groundtruth']:

        contours = reviewer_data['groundtruth']['merged_contours']

    elif 'groundtruth' in reviewer_data and 'original_contours' in reviewer_data['groundtruth']:

        contours = reviewer_data['groundtruth']['original_contours']

   

    # Process each segment

    for i, segment in enumerate(segments):

        annotation = {

            'obj_id': segment['id'],

            'bbox': segment['bounding_box']['bbox'],

            'positive_points': segment['positive_points'],

            'negative_points': segment['negative_points'],

            'contours': [contours[i]] if contours and i < len(contours) else [],

            'reviewer_id': best_reviewer_id  # Keep track of which reviewer was used

        }

        annotations.append(annotation)

   

    logger.info(f"Extracted {len(annotations)} annotations from frame {frame_number} using reviewer {best_reviewer_id}")

    return annotations

 

 

def resolve_frame_directory(raw_path: str) -> str:

    """Resolve the preprocessing directory path."""

    if os.path.exists(raw_path) and os.path.isdir(raw_path):

        return raw_path

    if ".png" in raw_path:

        last_slash_index = raw_path.rfind('/')

        if last_slash_index != -1:

            return raw_path[:last_slash_index]

    # Extract the base pattern (everything before the date)

    base_pattern = raw_path.split("__2024")[0] if "__2024" in raw_path else raw_path

    prod_dir = "/domino/datasets/local/arges_runs/prod"

   

    # Find matching directory

    matching_dirs = [d for d in os.listdir(prod_dir)

                    if d.startswith(os.path.basename(base_pattern))]

   

    if not matching_dirs:

        raise ValueError(f"No matching directory found for pattern: {base_pattern}")

       

    # Get the most recent matching directory

    latest_dir = sorted(matching_dirs)[-1]

   

    # Replace the old path prefix with the new one

    old_prefix = raw_path.split("/preprocessing")[0]

    new_prefix = os.path.join(prod_dir, latest_dir)

    preprocessed_path = raw_path.replace(old_prefix, new_prefix)

   

    if not os.path.exists(preprocessed_path):

        # Try removing the date suffix if it exists

        preprocessed_path = preprocessed_path.replace('_2019-10-07 10_05_15', '')

       

    if not os.path.exists(preprocessed_path):

        raise ValueError(f"Could not resolve frame directory: {preprocessed_path}")

       

    logger.info(f"Resolved preprocessing path: {preprocessed_path}")

    return preprocessed_path

 

def load_frame_image(frame_dir: str, frame_number: int) -> np.ndarray:

    """Load a frame image as numpy array."""

    frame_path = os.path.join(frame_dir, f"{frame_number}.png")

    if not os.path.exists(frame_path):

        raise ValueError(f"Frame image not found: {frame_path}")

       

    img = Image.open(frame_path)

    return np.array(img)

 

# CELL 2: SAM2 Setup and Propagation Functions

import sys

import os

def initialize_sam2():

    """Initialize the SAM2 model."""

    logger.info(f"Initializing SAM2 on {CONFIG['device']}...")

   

    try:

        # Import SAM2 modules

        sys.path.insert(0,str("/domino/datasets/local/cv_endoscopy/fgunderson/sam2-results/analysis/sam2"))

        if 'sam2' in sys.modules:

            del sys.modules['sam2']

           

        import sam2

        from sam2.build_sam import build_sam2_video_predictor

        print(f"Imported SAM2 from: {sam2.__file__}")

        # Build the predictor

        predictor = build_sam2_video_predictor(

            CONFIG['model_cfg'],

            CONFIG['sam2_checkpoint'],

            device=CONFIG['device']

        )

       

        logger.info("SAM2 initialized successfully")

        return predictor

    except Exception as e:

        logger.error(f"Failed to initialize SAM2: {str(e)}")

        raise

 

def contours_to_mask(contours: List[List[List[float]]], shape: Tuple[int, int]) -> np.ndarray:

    """Convert contour points to binary mask."""

    height, width = shape

    mask = np.zeros((height, width), dtype=np.int8)

 

    # Convert contours to the format expected by cv2.fillPoly

    formatted_contours = []

    for contour in contours:

        # Convert to integer points and proper shape for cv2

        pts = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))

        formatted_contours.append(pts)

 

    # Fill the contours with 1

    cv2.fillPoly(mask, formatted_contours, 1)

 

    # Convert to boolean mask

    return mask.astype(bool)

 

def prepare_temp_directory(frame_dir: str, frames: List[int]) -> str:

    """

    Create a temporary directory with symbolic links to frame images.

    Uses consecutive numbering for SAM2's internal use.

    Returns the path to the temporary directory.

    """

    temp_dir = tempfile.mkdtemp()

    logger.info(f"Created temporary directory: {temp_dir}")

   

    # Create symbolic links with consecutive numbering

    for i, frame_number in enumerate(frames):

        src_path = os.path.join(frame_dir, f"{frame_number}.png")

        if not os.path.exists(src_path):

            logger.warning(f"Frame file not found: {src_path}")

            continue

           

        # Create symbolic link with consecutive numbering

        dst_path = os.path.join(temp_dir, f"{i}.jpg")

        os.symlink(src_path, dst_path)

   

    logger.info(f"Prepared temporary directory with {len(frames)} frames")

    return temp_dir

 

def calculate_dice_coefficient(prediction: np.ndarray, ground_truth: np.ndarray) -> float:

    """Calculate the Dice coefficient between two binary masks."""

    intersection = np.logical_and(prediction, ground_truth).sum()

    union = prediction.sum() + ground_truth.sum()

   

    if union == 0:

        return 1.0  # Both masks empty - perfect agreement

       

    return 2 * intersection / union

 

def extract_metrics_from_output(output):

    """Extract IoU predictions and object scores from SAM2 output."""

    if isinstance(output, tuple) and len(output) > 2:

        # Handle tuple case

        prop_output = output[0]

    else:

        # Direct PropagationOutput

        prop_output = output

       

    iou_pred = None

    obj_score = None

   

    # Safely handle IoU predictions

    if hasattr(prop_output, 'iou_predictions') and prop_output.iou_predictions is not None:

        try:

            # Check if tensor exists and is not empty

            if isinstance(prop_output.iou_predictions, torch.Tensor) and prop_output.iou_predictions.numel() > 0:

                iou_pred = prop_output.iou_predictions[0].max().item()

        except (AttributeError, IndexError) as e:

            # Log error but continue

            print(f"Error extracting IoU prediction: {e}")

   

    # Safely handle object scores

    if hasattr(prop_output, 'object_score_logits') and prop_output.object_score_logits is not None:

        try:

            # Check if tensor exists and is not empty

            if isinstance(prop_output.object_score_logits, torch.Tensor) and prop_output.object_score_logits.numel() > 0:

                obj_score = prop_output.object_score_logits[0].item()

        except (AttributeError, IndexError) as e:

            # Log error but continue

            print(f"Error extracting object score: {e}")

   

    return iou_pred, obj_score

 

def propagate_masks_from_annotation(

    predictor,

    frame_dir: str,

    frames_sequence: List[int],

    annotations: List[Dict],

    use_mask_input: bool = False,

    separate_objects: bool = True

) -> Dict[int, Dict[str, Any]]:

    """

    Propagate masks from initial annotations through a sequence of frames.

   

    Args:

        predictor: SAM2 predictor instance

        frame_dir: Directory containing frame images

        frames_sequence: List of frame numbers in sequence

        annotations: List of annotation dictionaries for the initial frame

        use_mask_input: If True, use mask initialization instead of bbox+points

        separate_objects: If True, process each object with a separate inference state

                         If False, process all objects in a single inference state

       

    Returns:

        Dictionary mapping {frame_number: {

            obj_id: mask,

            "iou_predictions": {obj_id: iou_score},

            "object_score_logits": {obj_id: obj_score}

        }}

    """

    if not frames_sequence:

        logger.error("Empty frames sequence")

        return {}

       

    if not annotations:

        logger.error("No annotations provided")

        return {}

   

    # Prepare temporary directory with renumbered frames for SAM2

    temp_dir = prepare_temp_directory(frame_dir, frames_sequence)

   

    try:

        # Results dictionary

        results = {frame_num: {} for frame_num in frames_sequence}

       

        # Map from original frame numbers to temp directory indices

        frame_to_idx = {frame_num: i for i, frame_num in enumerate(frames_sequence)}

       

        # Get the starting frame and its index in the sequence

        start_frame = frames_sequence[0]

        start_idx = 0

       

        # Calculate frame dimensions from the first frame

        first_img = load_frame_image(frame_dir, start_frame)

        frame_height, frame_width = first_img.shape[:2]

       

        if separate_objects:

            # MODE 1: Process each annotation (object) separately with its own inference state

            logger.info("Using separate inference state for each object")

           

            for ann_idx, annotation in enumerate(annotations):

                obj_id = annotation['obj_id']

                logger.info(f"Processing object {obj_id}, annotation {ann_idx+1}/{len(annotations)}")

               

                # Initialize inference state for this object

                inference_state = predictor.init_state(

                    video_path=temp_dir,

                    start_frame=0,

                    end_frame=len(frames_sequence)-1,

                    async_loading_frames=False,

                    offload_video_to_cpu=True,

                    offload_state_to_cpu=True

                )

               

                # Get annotation details

                bbox = np.array(annotation['bbox'], dtype=np.float32)

                positive_points = np.array(annotation['positive_points'], dtype=np.float32) if annotation['positive_points'] else None

                negative_points = np.array(annotation['negative_points'], dtype=np.float32) if annotation['negative_points'] else None

                contours = annotation['contours']

 

                output = None

                # Initialize the mask on the first frame

                if use_mask_input and contours:

                    # Create mask from contours

                    mask = contours_to_mask(contours, (frame_height, frame_width))

                    mask_tensor = torch.tensor(mask, dtype=torch.bool)

                   

                    # Initialize with mask

                    output = predictor.add_new_mask(

                        inference_state=inference_state,

                        frame_idx=start_idx,

                        obj_id=int(obj_id),

                        mask=mask_tensor,

                        full_outputs=True

                    )

                else:

                    # Initialize with bounding box first

                    output = predictor.add_new_points_or_box(

                        inference_state=inference_state,

                        frame_idx=start_idx,

                        obj_id=int(obj_id),

                        box=bbox,

                        full_outputs=True

                    )

                   

                    # Then refine with points if available

                    if positive_points is not None or negative_points is not None:

                        points = []

                        labels = []

                       

                        if positive_points is not None:

                            points.extend(positive_points)

                            labels.extend([1] * len(positive_points))

                           

                        if negative_points is not None:

                            points.extend(negative_points)

                            labels.extend([0] * len(negative_points))

                       

                        if points:

                            points_arr = np.array(points, dtype=np.float32)

                            labels_arr = np.array(labels, dtype=np.int32)

                           

                            output = predictor.add_new_points_or_box(

                                inference_state=inference_state,

                                frame_idx=start_idx,

                                obj_id=int(obj_id),

                                points=points_arr,

                                labels=labels_arr,

                                full_outputs=True

                            )

 

                print(output)

                print(type(output.iou_predictions))

                print(output.iou_predictions)

                print(type(output.object_score_logits))

                print(output.object_score_logits)

                print(type(output.mask_logits))

                print(output.mask_logits)

 

                # Process the first frame output and save the mask

                if isinstance(output, tuple):

                    first_mask = (output[2][0] > 0.0).cpu().numpy()

                else:

                    first_mask = (output.mask_logits[0] > 0.0).cpu().numpy()

               

                # If save_to_disk is enabled, save masks to disk instead of keeping in memory

                if CONFIG.get("save_to_disk", False):

                    # Create directory for this run if it doesn't exist

                    temp_mask_dir = os.path.join(CONFIG["output_dir"], "temp_masks")

                    os.makedirs(temp_mask_dir, exist_ok=True)

                   

                    # Save mask to disk

                    mask_path = os.path.join(temp_mask_dir, f"frame_{start_frame}_obj_{obj_id}.npy")

                    np.save(mask_path, first_mask)

                   

                    # Store path instead of mask

                    results[start_frame][obj_id] = mask_path

                else:

                    # Store mask in memory

                    results[start_frame][obj_id] = first_mask

               

                # Store additional metrics from the output

                iou_pred, obj_score = extract_metrics_from_output(output)

               

                if iou_pred is not None:

                    if "iou_predictions" not in results[start_frame]:

                        results[start_frame]["iou_predictions"] = {}

                    results[start_frame]["iou_predictions"][obj_id] = iou_pred

               

                if obj_score is not None:

                    if "object_score_logits" not in results[start_frame]:

                        results[start_frame]["object_score_logits"] = {}

                    results[start_frame]["object_score_logits"][obj_id] = obj_score

               

                # Propagate to the rest of the frames

                max_frames_to_track = len(frames_sequence) - start_idx - 1

               

                if max_frames_to_track > 0:

                    try:

                        for out_idx, output in enumerate(predictor.propagate_in_video(

                            inference_state,

                            start_frame_idx=start_idx,

                            max_frame_num_to_track=max_frames_to_track,

                            reverse=False,

                            full_outputs=True

                        )):

                            # Calculate which frame this is

                            curr_idx = start_idx + out_idx + 1

                            if curr_idx >= len(frames_sequence):

                                logger.warning(f"Output index {curr_idx} out of range")

                                continue

                               

                            curr_frame = frames_sequence[curr_idx]

                           

                            # Extract mask

                            curr_mask = (output.mask_logits[0] > 0.0).cpu().numpy()

                           

                            # If save_to_disk is enabled, save masks to disk instead of keeping in memory

                            if CONFIG.get("save_to_disk", False):

                                # Create directory for this run if it doesn't exist

                                temp_mask_dir = os.path.join(CONFIG["output_dir"], "temp_masks")

                                os.makedirs(temp_mask_dir, exist_ok=True)

 

                               

                                # Save mask to disk

                                mask_path = os.path.join(temp_mask_dir, f"frame_{curr_frame}_obj_{obj_id}.npy")

                                np.save(mask_path, curr_mask)

                                print(mask_path)

                                # Store path instead of mask

                                results[curr_frame][obj_id] = mask_path

                                print(results[curr_frame])

                            else:

                                # Store mask in memory

                                results[curr_frame][obj_id] = curr_mask

                                print(results[curr_frame])

                           

                            # Store additional metrics from propagation output

                            iou_pred, obj_score = extract_metrics_from_output(output)

                           

                            if iou_pred is not None:

                                if "iou_predictions" not in results[curr_frame]:

                                    results[curr_frame]["iou_predictions"] = {}

                                results[curr_frame]["iou_predictions"][obj_id] = iou_pred

                           

                            if obj_score is not None:

                                if "object_score_logits" not in results[curr_frame]:

                                    results[curr_frame]["object_score_logits"] = {}

                                results[curr_frame]["object_score_logits"][obj_id] = obj_score

                           

                    except Exception as e:

                        logger.error(f"Error during propagation: {e}")

                       

                # Clean up resources for this object

                del inference_state

                if 'output' in locals():

                    del output

                   

                torch.cuda.empty_cache()

               

        else:

            # MODE 2: Process all objects in a single inference state

            logger.info("Using single inference state for all objects")

           

            # Initialize one inference state for all objects

            inference_state = predictor.init_state(

                video_path=temp_dir,

                start_frame=0,

                end_frame=len(frames_sequence)-1,

                async_loading_frames=False,

                offload_video_to_cpu=True,

                offload_state_to_cpu=True

            )

           

            try:

                # Process each annotation within the same inference state

                for ann_idx, annotation in enumerate(annotations):

                    obj_id = annotation['obj_id']

                    logger.info(f"Adding object {obj_id}, annotation {ann_idx+1}/{len(annotations)}")

                   

                    # Get annotation details

                    bbox = np.array(annotation['bbox'], dtype=np.float32)

                    positive_points = np.array(annotation['positive_points'], dtype=np.float32) if annotation['positive_points'] else None

                    negative_points = np.array(annotation['negative_points'], dtype=np.float32) if annotation['negative_points'] else None

                    contours = annotation['contours']

                   

                    # Initialize the mask on the first frame

                    if use_mask_input and contours:

                        # Create mask from contours

                        mask = contours_to_mask(contours, (frame_height, frame_width))

                        mask_tensor = torch.tensor(mask, dtype=torch.bool)

                       

                        # Initialize with mask

                        output = predictor.add_new_mask(

                            inference_state=inference_state,

                            frame_idx=start_idx,

                            obj_id=int(obj_id),

                            mask=mask_tensor,

                            full_outputs=True

                        )

                    else:

                        # Initialize with bounding box first

                        output = predictor.add_new_points_or_box(

                            inference_state=inference_state,

                            frame_idx=start_idx,

                            obj_id=int(obj_id),

                            box=bbox,

                            full_outputs=True

                        )

                       

                        # Then refine with points if available

                        if positive_points is not None or negative_points is not None:

                            points = []

                            labels = []

                           

                            if positive_points is not None:

                                points.extend(positive_points)

                                labels.extend([1] * len(positive_points))

                               

                            if negative_points is not None:

                                points.extend(negative_points)

                                labels.extend([0] * len(negative_points))

                           

                            if points:

                                points_arr = np.array(points, dtype=np.float32)

                                labels_arr = np.array(labels, dtype=np.int32)

                               

                                output = predictor.add_new_points_or_box(

                                    inference_state=inference_state,

                                    frame_idx=start_idx,

                                    obj_id=int(obj_id),

                                    points=points_arr,

                                    labels=labels_arr,

                                    full_outputs=True

                                )

                   

                    # Process the first frame output and save the mask

                    if isinstance(output, tuple):

                        first_mask = (output[2][0] > 0.0).cpu().numpy()

                    else:

                        first_mask = (output.mask_logits[0] > 0.0).cpu().numpy()

                       

                    results[start_frame][obj_id] = first_mask

                   

                    # Store additional metrics from the output

                    iou_pred, obj_score = extract_metrics_from_output(output)

                   

                    if iou_pred is not None:

                        if "iou_predictions" not in results[start_frame]:

                            results[start_frame]["iou_predictions"] = {}

                        results[start_frame]["iou_predictions"][obj_id] = iou_pred

                   

                    if obj_score is not None:

                        if "object_score_logits" not in results[start_frame]:

                            results[start_frame]["object_score_logits"] = {}

                        results[start_frame]["object_score_logits"][obj_id] = obj_score

               

                # Now propagate all objects together through all frames

                max_frames_to_track = len(frames_sequence) - start_idx - 1

               

                if max_frames_to_track > 0:

                    logger.info(f"Propagating all objects through {max_frames_to_track} frames")

                   

                    # Add progress bar for long propagations

                    progress_bar = tqdm(total=max_frames_to_track, desc=f"Propagating all objects")

                   

                    for out_idx, output in enumerate(predictor.propagate_in_video(

                        inference_state,

                        start_frame_idx=start_idx,

                        max_frame_num_to_track=max_frames_to_track,

                        reverse=False,

                        full_outputs=True

                    )):

                        # Calculate which frame this is

                        curr_idx = start_idx + out_idx + 1

                        if curr_idx >= len(frames_sequence):

                            logger.warning(f"Output index {curr_idx} out of range")

                            continue

                           

                        curr_frame = frames_sequence[curr_idx]

                       

                        # Process masks for each object

                        for obj_idx, annotation in enumerate(annotations):

                            obj_id = annotation['obj_id']

                           

                            # Extract mask for this object

                            # Note: In multi-object mode, mask_logits has shape [num_objects, height, width]

                            if hasattr(output, 'mask_logits') and len(output.mask_logits) > obj_idx:

                                curr_mask = (output.mask_logits[obj_idx] > 0.0).cpu().numpy()

                                results[curr_frame][obj_id] = curr_mask

                            else:

                                logger.warning(f"Missing mask for object {obj_id} at frame {curr_frame}")

                           

                            # Store additional metrics for this object from unified propagation

                            if hasattr(output, 'iou_predictions') and output.iou_predictions is not None:

                                if output.iou_predictions.shape[0] > obj_idx:

                                    # Get highest IoU prediction for this object

                                    iou_pred = output.iou_predictions[obj_idx].max().item()

                                    if "iou_predictions" not in results[curr_frame]:

                                        results[curr_frame]["iou_predictions"] = {}

                                    results[curr_frame]["iou_predictions"][obj_id] = iou_pred

                           

                            if hasattr(output, 'object_score_logits') and output.object_score_logits is not None:

                                if output.object_score_logits.shape[0] > obj_idx:

                                    obj_score = output.object_score_logits[obj_idx].item()

                                    if "object_score_logits" not in results[curr_frame]:

                                        results[curr_frame]["object_score_logits"] = {}

                                    results[curr_frame]["object_score_logits"][obj_id] = obj_score

                       

                        # Update progress bar

                        progress_bar.update(1)

                       

                        # Check memory periodically

                        if out_idx % CONFIG["memory_check_interval"] == 0:

                            check_memory_usage()

               

            except Exception as e:

                logger.error(f"Error during multi-object propagation: {e}")

               

            finally:

                if 'progress_bar' in locals():

                    progress_bar.close()

           

                # Clean up resources

                del inference_state

                if 'output' in locals():

                    del output

                torch.cuda.empty_cache()

       

        return results

       

    except Exception as e:

        logger.error(f"Error in propagation: {str(e)}")

        raise

       

    finally:

        # Remove temporary directory

        import shutil

        shutil.rmtree(temp_dir)

        logger.info(f"Removed temporary directory: {temp_dir}")

 

 

# CELL 3: Evaluation Functions

def evaluate_propagation(

    propagated_masks: Dict[int, Dict[int, Union[np.ndarray, str]]],

    frames_data: Dict,

    frame_range: List[int],

    frame_dir: str

) -> pd.DataFrame:

    """

    Evaluate propagated masks against ground truth for annotated frames.

   

    Args:

        propagated_masks: Dictionary mapping {frame_number: {obj_id: mask}}

        frames_data: Dictionary of frame data from segmentation JSON

        frame_range: Range of frames to evaluate [start, end]

        frame_dir: Directory containing frame images

       

    Returns:

        DataFrame with evaluation metrics

    """

    start_frame, end_frame = frame_range

    metrics = []

   

    # Find all annotated frames in range (from any reviewer)

    annotated_frames = find_annotated_frames(frames_data, frame_range)

   

    # Skip the first frame (the one used for initialization)

    eval_frames = annotated_frames[1:] if len(annotated_frames) > 1 else []

   

    if not eval_frames:

        logger.warning("No frames to evaluate - only found the initialization frame")

        return pd.DataFrame()

   

    # For each annotated frame (except the first)

    for frame_number in eval_frames:

        logger.info(f"Evaluating frame {frame_number}")

       

        # Skip if frame not in propagated results

        if frame_number not in propagated_masks:

            logger.warning(f"Frame {frame_number} not in propagation results, skipping")

            continue

           

        # Get ground truth annotations for this frame (from any reviewer)

        annotations = extract_annotation_data(frames_data, frame_number)

       

        if not annotations:

            logger.warning(f"No ground truth annotations for frame {frame_number}, skipping")

            continue

           

        # Load the frame image for dimensions

        frame_img = load_frame_image(frame_dir, frame_number)

        height, width = frame_img.shape[:2]

       

        # Create combined propagated mask (union of all object masks)

        combined_prop_mask = np.zeros((height, width), dtype=bool)

        for obj_id, mask_or_path in propagated_masks[frame_number].items():
            if not isinstance(obj_id, int) or isinstance(mask_or_path, dict):
                continue
            print(propagated_masks[frame_number].items())

            # Handle mask stored as file path

            if isinstance(mask_or_path, str) and mask_or_path.endswith('.npy'):

                mask = np.load(mask_or_path)

                logger.info(f"Loaded mask from disk with shape {mask.shape}")

            else:

                mask = mask_or_path

 

            if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
                logger.error(f"Invalid mask for object: {obj_id}")
                continue


                # Resize mask if needed

            logger.info(f"Mask: {mask}, {type(mask)}")

            if mask is None or mask.size==0:

                logger.error(f"Invalid frame dimension for object: {obj_id}")



            if len(mask.shape) == 3 and mask.shape[0] == 1:

                mask = mask[0]

                

            if mask.shape[:2] != (height, width):

                try:

                    mask = cv2.resize(

                        mask.astype(np.uint8),

                        (width, height),

                        interpolation=cv2.INTER_NEAREST

                    ).astype(bool)

                except cv2.error as e:

                    logger.error(f"Mask shape: {mask.shape}, Target: {height}x{width}")

            combined_prop_mask = np.logical_or(combined_prop_mask, mask)

       

        # Create combined ground truth mask from annotations

        combined_gt_mask = np.zeros((height, width), dtype=bool)

        for annotation in annotations:

            if annotation['contours']:

                obj_mask = contours_to_mask(annotation['contours'], (height, width))

                combined_gt_mask = np.logical_or(combined_gt_mask, obj_mask)

       

        # Calculate metrics

        dice = calculate_dice_coefficient(combined_prop_mask, combined_gt_mask)

        # precision = precision_score(combined_gt_mask.flatten(), combined_prop_mask.flatten(), zero_division=1)

        # recall = recall_score(combined_gt_mask.flatten(), combined_prop_mask.flatten(), zero_division=1)

        # f1 = f1_score(combined_gt_mask.flatten(), combined_prop_mask.flatten(), zero_division=1)

       

        # For object-level metrics, match each propagated object to its ground truth

        object_metrics = []

        for prop_obj_id, prop_mask_or_path in propagated_masks[frame_number].items():

            # Find best matching ground truth object (if available)

            best_dice = 0

            best_gt_id = None

           

            # Handle mask stored as file path

            if isinstance(prop_mask_or_path, str) and prop_mask_or_path.endswith('.npy'):

                prop_mask = np.load(prop_mask_or_path)

            else:

                prop_mask = prop_mask_or_path

           

            for annotation in annotations:

                gt_obj_id = annotation['obj_id']

               

                # Skip if no contours

                if not annotation['contours']:

                    continue

                   

                # Create GT mask

                gt_mask = contours_to_mask(annotation['contours'], (height, width))

 

                if len(prop_mask.shape) == 3 and prop_mask.shape[0] == 1:

                    prop_mask = prop_mask[0]

                # else:

                #     logger.info(f"GT Mask shape: {prop_mask.shape}, target: {height}x{width}")

               

                # Resize propagated mask if needed

                if prop_mask.shape[:2] != (height, width):

                    prop_mask = cv2.resize(

                        prop_mask.astype(np.uint8),

                        (width, height),

                        interpolation=cv2.INTER_NEAREST

                    ).astype(bool)

               

                # Calculate dice

                obj_dice = calculate_dice_coefficient(prop_mask, gt_mask)

               

                if obj_dice > best_dice:

                    best_dice = obj_dice

                    best_gt_id = gt_obj_id

           

            object_metrics.append({

                'prop_obj_id': prop_obj_id,

                'gt_obj_id': best_gt_id,

                'obj_dice': best_dice

            })

       

        # Calculate average object dice

        avg_obj_dice = np.mean([m['obj_dice'] for m in object_metrics]) if object_metrics else 0

       

        # Record metrics

        # Calculate average predicted IoU and object scores if available

        pred_iou = None

        obj_score = None

       

        if frame_number in propagated_masks:

            # Get predicted IoU if available

            if "iou_predictions" in propagated_masks[frame_number]:

                iou_values = list(propagated_masks[frame_number]["iou_predictions"].values())

                if iou_values:

                    pred_iou = np.mean(iou_values)

           

            # Get object scores if available

            if "object_score_logits" in propagated_masks[frame_number]:

                score_values = list(propagated_masks[frame_number]["object_score_logits"].values())

                if score_values:

                    obj_score = np.mean(score_values)

                   

        # Adjust num_objects to handle the new structure

        num_objects = len(propagated_masks[frame_number])

        if isinstance(next(iter(propagated_masks[frame_number].values())), dict):

            # If we're using the new structure with obj_masks

            num_objects = len([k for k in propagated_masks[frame_number].keys() if not k.startswith("iou") and not k.startswith("object")])

       

        # Record metrics

        metrics.append({

            'frame': frame_number,

            'frame_idx': annotated_frames.index(frame_number),

            'frames_from_start': frame_number - annotated_frames[0],

            'dice': dice,

            # 'precision': precision,

            # 'recall': recall,

            # 'f1': f1,

            'avg_obj_dice': avg_obj_dice,

            'num_objects': num_objects,

            'num_gt_objects': len(annotations),

            'object_metrics': object_metrics,

            'pred_iou': pred_iou,  # New field for predicted IoU

            'obj_score': obj_score  # New field for object score

        })

 

   

    # Convert to DataFrame

    metrics_df = pd.DataFrame(metrics)

   

    if not metrics_df.empty:

        logger.info(f"Evaluation results: Mean DICE = {metrics_df['dice'].mean():.4f}")

   

    return metrics_df

 

 

def load_combined_mask(results_dir: str, frame_number: int) -> Optional[np.ndarray]:

    """

    Load a combined mask (all objects) for a specific frame from saved results.

   

    Args:

        results_dir: Path to the results directory

        frame_number: Frame number to load

       

    Returns:

        Combined mask as numpy array, or None if not found

    """

    # Try loading from combined directory first

    combined_path = os.path.join(results_dir, "masks", "combined", f"frame_{frame_number}.npy")

   

    if os.path.exists(combined_path):

        return np.load(combined_path)

   

    # If no combined mask, try loading individual masks and combining them

    index_path = os.path.join(results_dir, "masks_index.json")

    if not os.path.exists(index_path):

        print(f"Masks index not found: {index_path}")

        return None

       

    with open(index_path, 'r') as f:

        masks_index = json.load(f)

   

    # Find the entry for this frame

    frame_entries = [entry for entry in masks_index if entry["frame"] == frame_number]

   

    if not frame_entries:

        print(f"No masks found for frame {frame_number}")

        return None

   

    frame_entry = frame_entries[0]

   

    # If only one object, just load that mask

    if len(frame_entry["objects"]) == 1:

        obj_id = frame_entry["objects"][0]

        return load_mask(results_dir, frame_number, obj_id)

   

    # Load all objects and combine them

    combined_mask = None

   

    for obj_id in frame_entry["objects"]:

        mask = load_mask(results_dir, frame_number, obj_id)

       

        if mask is not None:

            if combined_mask is None:

                combined_mask = mask.copy()

            else:

                combined_mask = np.logical_or(combined_mask, mask)

   

    return combined_mask

 

def load_mask(results_dir: str, frame_number: int, obj_id: str) -> Optional[np.ndarray]:

    """

    Load a specific mask from saved results.

   

    Args:

        results_dir: Path to the results directory

        frame_number: Frame number to load

        obj_id: Object ID to load

       

    Returns:

        Mask as numpy array, or None if not found

    """

    mask_path = os.path.join(results_dir, "masks", f"obj_{obj_id}", f"frame_{frame_number}.npy")

   

    if os.path.exists(mask_path):

        return np.load(mask_path)

    else:

        print(f"Mask not found: {mask_path}")

        return None

 

 

# CELL 4: Visualization Functions

def visualize_propagation_results(

    propagated_masks: Dict[int, Dict[int, Union[np.ndarray, str]]],

    frames_data: Dict,

    frame_range: List[int],

    frame_dir: str,

    metrics_df: pd.DataFrame,

    max_frames: int = 10,

    sample_strategy: str = "uniform"

):

    """

    Visualize propagation results along with ground truth and metrics.

   

    Args:

        propagated_masks: Dictionary mapping {frame_number: {obj_id: mask}}

        frames_data: Dictionary of frame data from segmentation JSON

        frame_range: Range of frames to evaluate [start, end]

        frame_dir: Directory containing frame images

        metrics_df: DataFrame with evaluation metrics

        max_frames: Maximum number of frames to visualize

        sample_strategy: Strategy for frame sampling ("uniform" or "key_frames")

    """

    # Find all annotated frames in range (from any reviewer)

    annotated_frames = find_annotated_frames(frames_data, frame_range)

   

    if not annotated_frames:

        logger.warning("No annotated frames found")

        return

   

    # Select frames to visualize based on strategy

    frames_to_viz = [annotated_frames[0]]  # Always include initialization frame

   

    # Handle long sequences with different sampling strategies

    if len(annotated_frames) > max_frames:

        logger.info(f"Long sequence detected ({len(annotated_frames)} frames). Using {sample_strategy} sampling.")

       

        if sample_strategy == "uniform":

            # Sample frames uniformly across the sequence

            if len(annotated_frames) > 1:  # Make sure we have more than just the init frame

                step = max(1, (len(annotated_frames) - 1) // (max_frames - 1))

                sampled_frames = annotated_frames[1::step][:max_frames-1]

                frames_to_viz.extend(sampled_frames)

       

        elif sample_strategy == "key_frames":

            # Use frames with metrics data, prioritizing those with largest changes in metrics

            if not metrics_df.empty:

                # Calculate frame-to-frame changes in DICE score

                if len(metrics_df) > 1:

                    metrics_df['dice_change'] = metrics_df['dice'].diff().abs().fillna(0)

                    # Get frames with largest metric changes

                    key_frames = metrics_df.nlargest(min(max_frames-1, len(metrics_df)), 'dice_change')['frame'].tolist()

                    frames_to_viz.extend(key_frames)

                else:

                    # If only one metric frame, include it

                    frames_to_viz.extend(metrics_df['frame'].tolist())

    else:

        # For shorter sequences, include all frames with metrics

        if not metrics_df.empty:

            eval_frames = metrics_df['frame'].tolist()

            frames_to_viz.extend(eval_frames)

   

    # Function to create mask visualization overlay

    def create_mask_overlay(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.5):

        """

        Create a mask overlay that only blends color in the masked regions, leaving other areas unchanged.

        More efficient implementation using array operations.

        """

        overlay = image.copy()

       

        if np.any(mask):

            # Convert color to a numpy array for easier calculation

            color_array = np.array(color, dtype=np.uint8)

           

            # Create a colored overlay for just the masked region

            # Only blend where mask is True

            for c in range(3):  # RGB channels

                overlay[:,:,c] = np.where(

                    mask,

                    (alpha * color_array[c] + (1 - alpha) * overlay[:,:,c]).astype(np.uint8),

                    overlay[:,:,c]

                )

           

            # Add border around the mask

            contours, _ = cv2.findContours(

                mask.astype(np.uint8),

                cv2.RETR_EXTERNAL,

                cv2.CHAIN_APPROX_SIMPLE

            )

            cv2.drawContours(overlay, contours, -1, color, 2)

       

        return overlay

 

    folder_id = CONFIG.get("folder_id", os.path.basename(frame_dir))

    # Visualize each selected frame

    for frame_number in frames_to_viz:

        # Get annotation data for this frame (from any reviewer)

        annotations = extract_annotation_data(frames_data, frame_number)

       

        # Load the frame image

        try:

            frame_img = load_frame_image(frame_dir, frame_number)

        except Exception as e:

            logger.error(f"Error loading frame {frame_number}: {e}")

            continue

       

        # Prepare figure for this frame

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

       

        # Plot original image

        axes[0].imshow(frame_img)

        axes[0].set_title(f"Frame {frame_number}\nVideo: {folder_id}")

        axes[0].axis('off')

       

        # Plot ground truth mask overlay

        if annotations and any(a['contours'] for a in annotations):

            gt_overlay = frame_img.copy()

            for i, annotation in enumerate(annotations):

                if annotation['contours']:

                    gt_mask = contours_to_mask(annotation['contours'], frame_img.shape[:2])

                    # Use different colors for different objects

                    color = plt.cm.tab10(i % 10)[:3]

                    color = tuple(int(c * 255) for c in color)

                    gt_overlay = create_mask_overlay(gt_overlay, gt_mask, color)

            axes[1].imshow(gt_overlay)

        else:

            axes[1].imshow(frame_img)

            axes[1].text(0.5, 0.5, "No ground truth",

                      ha='center', va='center', transform=axes[1].transAxes)

        axes[1].set_title(f"Ground Truth: {frame_number}\nVideo: {folder_id}")

        axes[1].axis('off')

       

        # Plot propagated mask overlay

        if frame_number in propagated_masks and propagated_masks[frame_number]:

            prop_overlay = frame_img.copy()

            for i, (obj_id, mask_or_path) in enumerate(propagated_masks[frame_number].items()):

                # Handle mask stored as file path

                if isinstance(mask_or_path, str) and mask_or_path.endswith('.npy'):

                    mask = np.load(mask_or_path)

                else:

                    mask = mask_or_path

 

                if len(mask.shape) == 3 and mask.shape[0] == 1:

                    mask = mask[0]

                # else:

                #     logger.info(f"GT Mask shape: {mask.shape}, target: {height}x{width}")

 

                # Resize mask if needed

                if mask.shape[:2] != frame_img.shape[:2]:

                    mask = cv2.resize(

                        mask.astype(np.uint8),

                        (frame_img.shape[1], frame_img.shape[0]),

                        interpolation=cv2.INTER_NEAREST

                    ).astype(bool)

               

                # Use different colors for different objects

                color = plt.cm.tab10(i % 10)[:3]

                color = tuple(int(c * 255) for c in color)

                prop_overlay = create_mask_overlay(prop_overlay, mask, color)

            axes[2].imshow(prop_overlay)

        else:

            axes[2].imshow(frame_img)

            axes[2].text(0.5, 0.5, "No propagation result",

                      ha='center', va='center', transform=axes[2].transAxes)

        axes[2].set_title(f"SAM2 Propagation: Frame {frame_number}\nVideo: {folder_id}")

        axes[2].axis('off')

       

        # Add metrics text if available

        if not metrics_df.empty and frame_number in metrics_df['frame'].values:

            metrics_row = metrics_df[metrics_df['frame'] == frame_number].iloc[0]

            metrics_text = (

                f"DICE: {metrics_row['dice']:.4f}\n"

                # f"Precision: {metrics_row['precision']:.4f}\n"

                # f"Recall: {metrics_row['recall']:.4f}"

            )

            fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=12)

       

        # Add title about initialization or evaluation frame

        if frame_number == annotated_frames[0]:

            fig.suptitle(f"Frame {frame_number} - Initialization Frame", fontsize=16)

        else:

            frame_delta = frame_number - annotated_frames[0]

            fig.suptitle(f"Frame {frame_number} - Evaluation Frame (+{frame_delta} from init)", fontsize=16)

       

        plt.tight_layout()

        plt.show()

   

    # Plot metrics over frames if available

    if not metrics_df.empty:

        # Figure 1: DICE and predicted IoU

        plt.figure(figsize=(12, 6))

       

        plt.plot(metrics_df['frames_from_start'], metrics_df['dice'], 'o-', label='DICE (Actual)', color='blue')

        # plt.plot(metrics_df['frames_from_start'], metrics_df['precision'], 's-', label='Precision', color='green')

        # plt.plot(metrics_df['frames_from_start'], metrics_df['recall'], '^-', label='Recall', color='purple')

        # plt.plot(metrics_df['frames_from_start'], metrics_df['f1'], 'D-', label='F1', color='cyan')

       

        # Add predicted IoU line if available

        if 'pred_iou' in metrics_df.columns and not metrics_df['pred_iou'].isna().all():

            plt.plot(metrics_df['frames_from_start'], metrics_df['pred_iou'], '*-',

                   label='Predicted IoU (SAM2)', color='red')

       

        plt.xlabel('Frames from Initialization')

        plt.ylabel('Score')

        plt.title('Propagation Performance and Predicted IoU Over Time')

        plt.grid(True, alpha=0.3)

        plt.legend()

       

        # Add average lines

        avg_dice = metrics_df['dice'].mean()

        plt.axhline(y=avg_dice, color='blue', linestyle='--', alpha=0.5,

                    label=f'Avg DICE: {avg_dice:.4f}')

       

        if 'pred_iou' in metrics_df.columns and not metrics_df['pred_iou'].isna().all():

            avg_pred_iou = metrics_df['pred_iou'].mean()

            plt.axhline(y=avg_pred_iou, color='red', linestyle='--', alpha=0.5,

                      label=f'Avg Pred IoU: {avg_pred_iou:.4f}')

       

        plt.tight_layout()

        plt.show()

       

        # Figure 2: Object scores (confidence/occlusion)

        if 'obj_score' in metrics_df.columns and not metrics_df['obj_score'].isna().all():

            plt.figure(figsize=(12, 6))

           

            plt.plot(metrics_df['frames_from_start'], metrics_df['obj_score'], 'o-',

                   label='Object Score', color='orange')

           

            plt.xlabel('Frames from Initialization')

            plt.ylabel('Object Score')

            plt.title('Object Confidence/Occlusion Score Over Time')

            plt.grid(True, alpha=0.3)

           

            # Add average line

            avg_obj_score = metrics_df['obj_score'].mean()

            plt.axhline(y=avg_obj_score, color='orange', linestyle='--', alpha=0.5,

                        label=f'Avg Score: {avg_obj_score:.4f}')

           

            plt.legend()

            plt.tight_layout()

            plt.show()

 

 

 

       

        # # Plot object-level metrics

        # if 'avg_obj_dice' in metrics_df.columns:

        #     plt.figure(figsize=(12, 6))

        #     plt.plot(metrics_df['frames_from_start'], metrics_df['avg_obj_dice'], 'o-',

        #              label='Avg Object DICE')

        #     plt.plot(metrics_df['frames_from_start'], metrics_df['dice'], 's-',

        #              label='Combined Mask DICE')

           

        #     plt.xlabel('Frames from Initialization')

        #     plt.ylabel('DICE Score')

        #     plt.title('Object-Level vs. Combined Mask Performance')

        #     plt.grid(True, alpha=0.3)

        #     plt.legend()

           

        #     plt.tight_layout()

        #     plt.show()

 

 

 

def visualize_sam2_pipeline_overview(

    propagated_masks: Dict[int, Dict[int, Union[np.ndarray, str]]],

    frames_data: Dict,

    frame_dir: str,

    metrics_df: pd.DataFrame,

    init_annotations: List[Dict],

    frames_to_viz: List[int] = None,

    fig_size: Tuple[int, int] = (24, 8)

):

    """

    Create a comprehensive visualization of the SAM2 pipeline, showing 6 images in a row:

    1. Original image (initialization frame)

    2. Ground truth merged contours (from JSON)

    3. SAM2 inputs (bbox+points or mask inputs)

    4-6. Three SAM2 generated results from different frames with DICE scores

   

    Args:

        propagated_masks: Dictionary of propagated masks

        frames_data: Dictionary of frame data from segmentation JSON

        frame_dir: Directory containing frame images

        metrics_df: DataFrame with evaluation metrics

        init_annotations: Annotations used for initialization

        frames_to_viz: List of frames to visualize [init_frame, frame2, frame3]

                     Must contain at least the initialization frame

        fig_size: Size of the figure (width, height)

    """

    # Verify we have at least the initialization frame

    if not frames_to_viz or len(frames_to_viz) < 1:

        logger.warning("No frames specified for visualization, cannot create pipeline overview")

        return

   

    # The first frame is always the initialization frame

    init_frame = frames_to_viz[0]

   

    # Make sure we have at least 3 frames (init + 2 more)

    # If not enough frames provided, duplicate the last one

    while len(frames_to_viz) < 3:

        frames_to_viz.append(frames_to_viz[-1])

   

    logger.info(f"Creating pipeline visualization with frames: {frames_to_viz}")

    folder_id = CONFIG.get("folder_id", os.path.basename(frame_dir))

    # Create figure

    fig, axes = plt.subplots(1, 6, figsize=fig_size)

   

    # ----------------- Panel 1: Original Image -----------------

    try:

        # Load the initialization frame

        init_img = load_frame_image(frame_dir, init_frame)

        axes[0].imshow(init_img)

        axes[0].set_title(f"Original Image\nFrame {init_frame}")

        axes[0].axis('off')

    except Exception as e:

        logger.error(f"Error loading original image: {e}")

        axes[0].text(0.5, 0.5, "Error loading image", ha='center', va='center')

        axes[0].axis('off')

   

    # ----------------- Panel 2: Ground Truth Contours -----------------

    try:

        # Load the initialization frame again for ground truth overlay

        gt_img = init_img.copy() if 'init_img' in locals() else load_frame_image(frame_dir, init_frame)

       

        # Get annotations for initialization frame

        init_gt_annotations = extract_annotation_data(frames_data, init_frame)

       

        if init_gt_annotations and any(a['contours'] for a in init_gt_annotations):

            # Draw contours on image

            for i, annotation in enumerate(init_gt_annotations):

                if annotation['contours']:

                    # Convert contours to binary mask

                    gt_mask = contours_to_mask(annotation['contours'], gt_img.shape[:2])

                   

                    # Use different colors for different objects

                    color = plt.cm.tab10(i % 10)[:3]

                    color = tuple(int(c * 255) for c in color)

                   

                    # Create mask overlay

                    colored_mask = np.zeros_like(gt_img)

                    colored_mask[gt_mask] = color

                   

                    # Add to image with transparency

                    alpha = 0.2

                    gt_img = cv2.addWeighted(

                        colored_mask, alpha,

                        gt_img, 1 - alpha,

                        0

                    )

                   

                    # Add contour around the mask

                    contours, _ = cv2.findContours(

                        gt_mask.astype(np.uint8),

                        cv2.RETR_EXTERNAL,

                        cv2.CHAIN_APPROX_SIMPLE

                    )

                    cv2.drawContours(gt_img, contours, -1, color, 2)

                   

                    # Add object ID label

                    if len(contours) > 0:

                        # Find a point to place the label (centroid)

                        M = cv2.moments(contours[0])

                        if M["m00"] != 0:

                            cx = int(M["m10"] / M["m00"])

                            cy = int(M["m01"] / M["m00"])

                           

                            # Draw the object ID label

                            cv2.putText(

                                gt_img,

                                f"Obj {annotation['obj_id']}",

                                (cx, cy),

                                cv2.FONT_HERSHEY_SIMPLEX,

                                0.7,

                                (255, 255, 255),

                                2

                            )

           

            axes[1].imshow(gt_img)

        else:

            axes[1].imshow(gt_img)

            axes[1].text(0.5, 0.5, "No ground truth contours",

                      ha='center', va='center', transform=axes[1].transAxes)

       

        axes[1].set_title("Ground Truth Contours")

        axes[1].axis('off')

    except Exception as e:

        logger.error(f"Error creating ground truth visualization: {e}")

        axes[1].text(0.5, 0.5, "Error creating visualization", ha='center', va='center')

        axes[1].axis('off')

   

    # ----------------- Panel 3: SAM2 Inputs -----------------

    try:

        # Load the initialization frame again for input visualization

        input_img = init_img.copy() if 'init_img' in locals() else load_frame_image(frame_dir, init_frame)

       

        # Check if we used mask input or bbox+points

        use_mask_input = CONFIG.get("use_mask_input", False)

       

        if use_mask_input:

            # Visualize mask inputs

            for i, annotation in enumerate(init_annotations):

                if annotation['contours']:

                    # Create mask from contours

                    input_mask = contours_to_mask(annotation['contours'], input_img.shape[:2])

                   

                    # Use different colors for different objects

                    color = plt.cm.tab10(i % 10)[:3]

                    color = tuple(int(c * 255) for c in color)

                   

                    # Create colored mask overlay

                    colored_mask = np.zeros_like(input_img)

                    colored_mask[input_mask] = color

                   

                    # Add to image with transparency

                    alpha = 0.2  # More transparent than ground truth

                    input_img = cv2.addWeighted(

                        colored_mask, alpha,

                        input_img, 1 - alpha,

                        0

                    )

                   

                    # Add mask contour

                    contours, _ = cv2.findContours(

                        input_mask.astype(np.uint8),

                        cv2.RETR_EXTERNAL,

                        cv2.CHAIN_APPROX_SIMPLE

                    )

                    cv2.drawContours(input_img, contours, -1, color, 2)

                   

                    # Label as mask input

                    if len(contours) > 0:

                        M = cv2.moments(contours[0])

                        if M["m00"] != 0:

                            cx = int(M["m10"] / M["m00"])

                            cy = int(M["m01"] / M["m00"])

                           

                            cv2.putText(

                                input_img,

                                f"Mask {annotation['obj_id']}",

                                (cx, cy),

                                cv2.FONT_HERSHEY_SIMPLEX,

                                0.7,

                                (255, 255, 255),

                                2

                            )

        else:

            # Visualize bbox and points inputs

            for i, annotation in enumerate(init_annotations):

                # Draw bounding box

                if 'bbox' in annotation:

                    bbox = annotation['bbox']

                    # Format is [x, y, width, height]

                    x, y, w, h = [int(v) for v in bbox]

                   

                    # Use different colors for different objects

                    color = plt.cm.tab10(i % 10)[:3]

                    color = tuple(int(c * 255) for c in color)

                   

                    # Draw bbox

                    cv2.rectangle(input_img, (x, y), (x + w, y + h), color, 2)

                   

                    # Label the box

                    cv2.putText(

                        input_img,

                        f"Box {annotation['obj_id']}",

                        (x, y - 10),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.7,

                        color,

                        2

                    )

               

                # Draw positive points

                if 'positive_points' in annotation and annotation['positive_points']:

                    for point in annotation['positive_points']:

                        px, py = int(point[0]), int(point[1])

                        cv2.circle(input_img, (px, py), 5, (0, 255, 0), -1)  # Green for positive

               

                # Draw negative points

                if 'negative_points' in annotation and annotation['negative_points']:

                    for point in annotation['negative_points']:

                        px, py = int(point[0]), int(point[1])

                        cv2.circle(input_img, (px, py), 5, (255, 0, 0), -1)  # Red for negative

       

        axes[2].imshow(input_img)

        input_type = "Mask Inputs" if use_mask_input else "BBox + Points Inputs"

        axes[2].set_title(f"SAM2 {input_type}")

        axes[2].axis('off')

    except Exception as e:

        logger.error(f"Error creating input visualization: {e}")

        axes[2].text(0.5, 0.5, "Error creating visualization", ha='center', va='center')

        axes[2].axis('off')

   

    # ----------------- Panels 4-6: SAM2 Results for Three Frames -----------------

    for i, frame_number in enumerate(frames_to_viz):

        ax_idx = i + 3  # Starting at axes[3]

       

        try:

            # Load the frame image

            frame_img = load_frame_image(frame_dir, frame_number)

           

            # Create mask visualization

            if frame_number in propagated_masks and propagated_masks[frame_number]:

                result_img = frame_img.copy()

               

                for j, (obj_id, mask_or_path) in enumerate(propagated_masks[frame_number].items()):

                    # Handle mask stored as file path

                    if isinstance(mask_or_path, str) and mask_or_path.endswith('.npy'):

                        mask = np.load(mask_or_path)

                    else:

                        mask = mask_or_path

 

                    if len(mask.shape) == 3 and mask.shape[0] == 1:

                        mask = mask[0]

                    # else:

                    #     logger.info(f"GT Mask shape: {mask.shape}, target: {height}x{width}")

 

                    # Resize mask if needed

                    if mask.shape[:2] != frame_img.shape[:2]:

                        mask = cv2.resize(

                            mask.astype(np.uint8),

                            (frame_img.shape[1], frame_img.shape[0]),

                            interpolation=cv2.INTER_NEAREST

                        ).astype(bool)

                   

                    # Use different colors for different objects

                    color = plt.cm.tab10(j % 10)[:3]

                    color = tuple(int(c * 255) for c in color)

                   

                    # Create colored mask overlay

                    colored_mask = np.zeros_like(result_img)

                    colored_mask[mask] = color

                   

                    # Add to image with transparency

                    alpha = 0.2

                    result_img = cv2.addWeighted(

                        colored_mask, alpha,

                        result_img, 1 - alpha,

                        0

                    )

                   

                    # Add contour around the mask

                    contours, _ = cv2.findContours(

                        mask.astype(np.uint8),

                        cv2.RETR_EXTERNAL,

                        cv2.CHAIN_APPROX_SIMPLE

                    )

                    cv2.drawContours(result_img, contours, -1, color, 2)

                   

                    # Add object ID label

                    if len(contours) > 0:

                        M = cv2.moments(contours[0])

                        if M["m00"] != 0:

                            cx = int(M["m10"] / M["m00"])

                            cy = int(M["m01"] / M["m00"])

                           

                            cv2.putText(

                                result_img,

                                f"Obj {obj_id}",

                                (cx, cy),

                                cv2.FONT_HERSHEY_SIMPLEX,

                                0.7,

                                (255, 255, 255),

                                2

                            )

               

                axes[ax_idx].imshow(result_img)

            else:

                axes[ax_idx].imshow(frame_img)

                axes[ax_idx].text(0.5, 0.5, "No propagation result",

                               ha='center', va='center', transform=axes[ax_idx].transAxes)

           

            # Get DICE score if available

            dice_score = None

            if not metrics_df.empty and frame_number in metrics_df['frame'].values:

                metrics_row = metrics_df[metrics_df['frame'] == frame_number].iloc[0]

                dice_score = metrics_row['dice']

           

            # Create title with DICE score if available

            if dice_score is not None:

                title = f"Frame {frame_number}\nDICE: {dice_score:.4f}"

            else:

                title = f"Frame {frame_number}"

               

            # Special label for initialization frame

            if frame_number == init_frame:

                title = f"Init {title}"

               

            axes[ax_idx].set_title(title)

            axes[ax_idx].axis('off')

        except Exception as e:

            logger.error(f"Error creating result visualization for frame {frame_number}: {e}")

            axes[ax_idx].text(0.5, 0.5, f"Error with frame {frame_number}", ha='center', va='center')

            axes[ax_idx].axis('off')

   

    # Add overall title

    use_mask_input = CONFIG.get("use_mask_input", False)

    separate_objects = CONFIG.get("separate_objects", True)

   

    init_type = "Mask" if use_mask_input else "BBox+Points"

    obj_mode = "Separate" if separate_objects else "Unified"

   

    fig.suptitle(

        f"SAM2 Pipeline Overview: {init_type} Init, {obj_mode} Objects Mode\nVideo ID: {folder_id}",

        fontsize=16

    )

   

    plt.tight_layout()

    fig.subplots_adjust(top=0.85)  # Make room for the suptitle

    plt.show()

   

    # Save visualization if output directory is specified

    if CONFIG.get("save_intermediates", True) and "output_dir" in CONFIG:

        viz_dir = os.path.join(CONFIG["output_dir"], "visualizations")

        os.makedirs(viz_dir, exist_ok=True)

       

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        viz_path = os.path.join(

            viz_dir,

            f"pipeline_overview_{timestamp}.png"

        )

       

        fig.savefig(viz_path, dpi=150, bbox_inches='tight')

        logger.info(f"Saved pipeline visualization to {viz_path}")

   

    return fig

 

 

def save_evaluation_results(

    propagated_masks: Dict[int, Dict],

    metrics_df: pd.DataFrame,

    config: Dict,

    output_dir: str,

    results_prefix: str = "sam2_eval"

):

    """

    Save propagation results and metrics for future reference with improved organization.

    Now includes IoU predictions and object scores in masks_index.json.

    """

    # Create timestamp for unique directory

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

   

    # Extract key parameters for directory naming

    folder_id = config["folder_id"]

    start_frame, end_frame = config["frame_range"]

    input_type = "mask" if config["use_mask_input"] else "bbox"

    obj_mode = "separate" if config.get("separate_objects", True) else "unified"

    reviewer = config.get("reviewer_id", "any_reviewer").replace(" ", "") if "reviewer_id" in config else "any_reviewer"

 

    # Create descriptive directory name

    dir_name = f"{results_prefix}_{timestamp}_{folder_id.split('_')[0]}_f{start_frame}-{end_frame}_{input_type}_{obj_mode}_{reviewer}"

    results_dir = os.path.join(output_dir, dir_name)

   

    logger.info(f"Creating results directory: {results_dir}")

    os.makedirs(results_dir, exist_ok=True)

   

    # Create subdirectories

    masks_dir = os.path.join(results_dir, "masks")

    metrics_dir = os.path.join(results_dir, "metrics")

    visualization_dir = os.path.join(results_dir, "visualizations")

   

    os.makedirs(masks_dir, exist_ok=True)

    os.makedirs(metrics_dir, exist_ok=True)

    os.makedirs(visualization_dir, exist_ok=True)

   

    # Save configuration

    with open(os.path.join(results_dir, "config.json"), 'w') as f:

        # Create a copy of config and remove any non-serializable items

        save_config = {k: v for k, v in config.items() if isinstance(v, (str, int, float, list, dict, bool)) or v is None}

        json.dump(save_config, f, indent=2)

   

    # Save metrics

    if not metrics_df.empty:

        # Convert object_metrics column to string for serialization

        if 'object_metrics' in metrics_df.columns:

            metrics_df['object_metrics'] = metrics_df['object_metrics'].apply(lambda x: str(x))

       

        # Save as CSV

        metrics_df.to_csv(os.path.join(metrics_dir, "frame_metrics.csv"), index=False)

       

        # Save summary metrics - now including predicted IoU and object scores

        summary_metrics = {

            "mean_dice": metrics_df['dice'].mean(),

            "std_dice": metrics_df['dice'].std(),

            "min_dice": metrics_df['dice'].min(),

            "max_dice": metrics_df['dice'].max(),

            "mean_precision": metrics_df['precision'].mean(),

            "mean_recall": metrics_df['recall'].mean(),

            "mean_f1": metrics_df['f1'].mean(),

            "frame_count": len(metrics_df),

            "object_count": metrics_df['num_objects'].mean()

        }

       

        # Add predicted IoU and object score summaries if available

        if 'pred_iou' in metrics_df.columns and not metrics_df['pred_iou'].isna().all():

            summary_metrics.update({

                "mean_pred_iou": metrics_df['pred_iou'].mean(),

                "std_pred_iou": metrics_df['pred_iou'].std(),

                "min_pred_iou": metrics_df['pred_iou'].min(),

                "max_pred_iou": metrics_df['pred_iou'].max()

            })

           

        if 'obj_score' in metrics_df.columns and not metrics_df['obj_score'].isna().all():

            summary_metrics.update({

                "mean_obj_score": metrics_df['obj_score'].mean(),

                "std_obj_score": metrics_df['obj_score'].std(),

                "min_obj_score": metrics_df['obj_score'].min(),

                "max_obj_score": metrics_df['obj_score'].max()

            })

       

        with open(os.path.join(metrics_dir, "summary_metrics.json"), 'w') as f:

            json.dump(summary_metrics, f, indent=2)

   

    # Save masks with more descriptive filenames

    logger.info(f"Saving masks to {masks_dir}")

   

    # Create subdirectories for each object and for combined masks

    saved_masks_count = 0

    combined_masks_dir = os.path.join(masks_dir, "combined")

    os.makedirs(combined_masks_dir, exist_ok=True)

   

    # Create a dictionary to hold combined masks

    combined_masks = {}

   

    # Generate and save index file for easy loading

    masks_index = []

    for frame_num in sorted(propagated_masks.keys()):

        frame_data = propagated_masks[frame_num]

       

        # Extract special fields

        special_keys = ["iou_predictions", "object_score_logits"]

        obj_ids = [k for k in frame_data.keys() if k not in special_keys]

       

        # Extract IoU and object score data if available

        frame_iou_preds = {}

        frame_obj_scores = {}

       

        if "iou_predictions" in frame_data:

            frame_iou_preds = {str(obj_id): float(iou) for obj_id, iou in frame_data["iou_predictions"].items()}

       

        if "object_score_logits" in frame_data:

            frame_obj_scores = {str(obj_id): float(score) for obj_id, score in frame_data["object_score_logits"].items()}

       

        # Initialize combined mask for this frame (if multiple objects exist)

        if len(obj_ids) > 1:

            combined_masks[frame_num] = None

       

        # Save each object's mask

        for obj_id in obj_ids:

            # Create object directory

            obj_dir = os.path.join(masks_dir, f"obj_{obj_id}")

            os.makedirs(obj_dir, exist_ok=True)

           

            # Prepare mask - handle both in-memory masks and paths

            mask_or_path = frame_data[obj_id]

            if isinstance(mask_or_path, str) and mask_or_path.endswith('.npy'):

                # If mask was saved to disk, load it

                mask = np.load(mask_or_path)

            else:

                mask = mask_or_path

           

            # Save with descriptive filename

            mask_path = os.path.join(obj_dir, f"frame_{frame_num}.npy")

            np.save(mask_path, mask)

            saved_masks_count += 1

           

            # Update combined mask for this frame

            if len(obj_ids) > 1:

                if combined_masks[frame_num] is None:

                    combined_masks[frame_num] = mask.copy()

                else:

                    combined_masks[frame_num] = np.logical_or(combined_masks[frame_num], mask)

       

        # Save combined masks

        combined_path = None

        if frame_num in combined_masks and combined_masks[frame_num] is not None:

            combined_mask_path = os.path.join(combined_masks_dir, f"frame_{frame_num}.npy")

            np.save(combined_mask_path, combined_masks[frame_num])

            combined_path = f"masks/combined/frame_{frame_num}.npy"

       

        # Create paths for masks index

        paths = [f"masks/obj_{obj_id}/frame_{frame_num}.npy" for obj_id in obj_ids]

       

        # Add entry to masks index

        masks_index.append({

            "frame": frame_num,

            "objects": obj_ids,

            "paths": paths,

            "combined_path": combined_path,

            "iou_predictions": frame_iou_preds,  # NEW

            "object_score_logits": frame_obj_scores    # NEW

        })

   

    with open(os.path.join(results_dir, "masks_index.json"), 'w') as f:

        json.dump(masks_index, f, indent=2)

   

    logger.info(f"Saved {saved_masks_count} masks across {len(propagated_masks)} frames")

    logger.info(f"Saved {len(combined_masks)} combined masks for frames with multiple objects")

    logger.info(f"Saved evaluation results to {results_dir}")

   

    return results_dir

 

 

def visualize_saved_results(results_dir: str, max_frames: int = 10, show_metrics: bool = True):

    """

    Visualize results from a previously saved evaluation run.

   

    Args:

        results_dir: Path to the saved results directory

        max_frames: Maximum number of frames to visualize

        show_metrics: Whether to show metrics plots

    """

    # Check if directory exists

    if not os.path.exists(results_dir):

        print(f"Results directory not found: {results_dir}")

        return

   

    # Load configuration

    config_path = os.path.join(results_dir, "config.json")

    if not os.path.exists(config_path):

        print(f"Configuration file not found: {config_path}")

        return

       

    with open(config_path, 'r') as f:

        config = json.load(f)

   

    # Load masks index

    index_path = os.path.join(results_dir, "masks_index.json")

    if not os.path.exists(index_path):

        print(f"Masks index not found: {index_path}")

        return

       

    with open(index_path, 'r') as f:

        masks_index = json.load(f)

   

    # Load metrics

    metrics_path = os.path.join(results_dir, "metrics", "frame_metrics.csv")

    if os.path.exists(metrics_path):

        metrics_df = pd.read_csv(metrics_path)

    else:

        metrics_df = pd.DataFrame()

        print("Metrics file not found, visualizing without metrics")

   

    # Load summary metrics

    summary_path = os.path.join(results_dir, "metrics", "summary_metrics.json")

    if os.path.exists(summary_path):

        with open(summary_path, 'r') as f:

            summary = json.load(f)

        print(f"Summary Metrics:")

        print(f"  Mean DICE: {summary['mean_dice']:.4f}")

        print(f"  Min/Max DICE: {summary['min_dice']:.4f}/{summary['max_dice']:.4f}")

        print(f"  Precision/Recall: {summary['mean_precision']:.4f}/{summary['mean_recall']:.4f}")

        print(f"  Objects per frame: {summary['object_count']:.1f}")

   

    # Select frames to visualize

    frames = sorted(list(set([entry["frame"] for entry in masks_index])))

   

    if len(frames) > max_frames:

        # Sample frames uniformly

        step = max(1, len(frames) // max_frames)

        frames_to_show = frames[::step][:max_frames]

    else:

        frames_to_show = frames

   

    print(f"Visualizing {len(frames_to_show)} frames out of {len(frames)}")

   

    # Visualize each frame

    for frame_number in frames_to_show:

        # Find objects for this frame

        frame_entries = [entry for entry in masks_index if entry["frame"] == frame_number]

       

        if not frame_entries:

            continue

           

        # Load masks for this frame

        masks = {}

        for entry in frame_entries:

            for obj_id, mask_path in zip(entry["objects"], entry["paths"]):

                full_path = os.path.join(results_dir, mask_path)

                if os.path.exists(full_path):

                    masks[obj_id] = np.load(full_path)

       

        # Create visualization

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

       

        # Use one of the masks to get dimensions (all should be the same)

        if masks:

            example_mask = next(iter(masks.values()))

            height, width = example_mask.shape

           

            # Create a blank image as placeholder (would be better with real frame)

            blank_img = np.zeros((height, width, 3), dtype=np.uint8)

            blank_img.fill(200)  # Light gray background

           

            # Draw text on image

            cv2.putText(

                blank_img,

                f"Frame {frame_number}",

                (width//4, height//2),

                cv2.FONT_HERSHEY_SIMPLEX,

                1.5,

                (0, 0, 0),

                2

            )

           

            # Show blank image

            axes[0].imshow(blank_img)

            axes[0].set_title(f"Frame {frame_number}")

            axes[0].axis('off')

           

            # Create combined mask visualization

            prop_overlay = blank_img.copy()

            for i, (obj_id, mask) in enumerate(masks.items()):

                # Use different colors for different objects

                color = plt.cm.tab10(i % 10)[:3]

                color = tuple(int(c * 255) for c in color)

               

                # Create custom mask overlay since we don't have the create_mask_overlay function

                colored_mask = np.zeros_like(prop_overlay)

                colored_mask[mask] = color

               

                # Add to overlay with transparency

                alpha = 0.2

                prop_overlay = cv2.addWeighted(

                    colored_mask, alpha,

                    prop_overlay, 1 - alpha,

                    0

                )

               

                # Add contour around the mask

                contours, _ = cv2.findContours(

                    mask.astype(np.uint8),

                    cv2.RETR_EXTERNAL,

                    cv2.CHAIN_APPROX_SIMPLE

                )

                cv2.drawContours(prop_overlay, contours, -1, color, 2)

               

                # Add object ID label

                if len(contours) > 0:

                    # Find a point to place the label (centroid)

                    M = cv2.moments(contours[0])

                    if M["m00"] != 0:

                        cx = int(M["m10"] / M["m00"])

                        cy = int(M["m01"] / M["m00"])

                       

                        # Draw the object ID label

                        cv2.putText(

                            prop_overlay,

                            f"Obj {obj_id}",

                            (cx, cy),

                            cv2.FONT_HERSHEY_SIMPLEX,

                            0.7,

                            (255, 255, 255),

                            2

                        )

           

            # Show mask overlay

            axes[1].imshow(prop_overlay)

            axes[1].set_title(f"Propagated Masks")

            axes[1].axis('off')

           

            # Add metrics if available

            if not metrics_df.empty and frame_number in metrics_df['frame'].values:

                metrics_row = metrics_df[metrics_df['frame'] == frame_number].iloc[0]

                metrics_text = (

                    f"DICE: {metrics_row['dice']:.4f}, "

                    f"Prec: {metrics_row['precision']:.4f}, "

                    f"Rec: {metrics_row['recall']:.4f}"

                )

                fig.text(0.5, 0.01, metrics_text, ha='center', fontsize=12)

       

        plt.tight_layout()

        plt.show()

   

    # Plot metrics trends if available

    if show_metrics and not metrics_df.empty:

        plt.figure(figsize=(12, 6))

       

        plt.plot(metrics_df['frames_from_start'], metrics_df['dice'], 'o-', label='DICE')

        plt.plot(metrics_df['frames_from_start'], metrics_df['precision'], 's-', label='Precision')

        plt.plot(metrics_df['frames_from_start'], metrics_df['recall'], '^-', label='Recall')

       

        plt.xlabel('Frames from Initialization')

        plt.ylabel('Score')

        plt.title('Propagation Performance Over Time')

        plt.grid(True, alpha=0.3)

        plt.legend()

       

        # Add average line

        avg_dice = metrics_df['dice'].mean()

        plt.axhline(y=avg_dice, color='r', linestyle='--', alpha=0.5,

                    label=f'Avg DICE: {avg_dice:.4f}')

       

        plt.tight_layout()

        plt.show()

 

def load_evaluation_results(results_dir: str) -> Dict:

    """

    Load results from a previously saved evaluation run.

   

    Args:

        results_dir: Path to the saved results directory

       

    Returns:

        Dictionary containing configuration, metrics, and masks index

    """

    # Check if directory exists

    if not os.path.exists(results_dir):

        print(f"Results directory not found: {results_dir}")

        return {}

   

    results = {

        "config": None,

        "metrics_df": None,

        "summary": None,

        "masks_index": None,

        "results_dir": results_dir

    }

   

    # Load configuration

    config_path = os.path.join(results_dir, "config.json")

    if os.path.exists(config_path):

        with open(config_path, 'r') as f:

            results["config"] = json.load(f)

   

    # Load metrics

    metrics_path = os.path.join(results_dir, "metrics", "frame_metrics.csv")

    if os.path.exists(metrics_path):

        results["metrics_df"] = pd.read_csv(metrics_path)

   

    # Load summary metrics

    summary_path = os.path.join(results_dir, "metrics", "summary_metrics.json")

    if os.path.exists(summary_path):

        with open(summary_path, 'r') as f:

            results["summary"] = json.load(f)

   

    # Load masks index

    index_path = os.path.join(results_dir, "masks_index.json")

    if os.path.exists(index_path):

        with open(index_path, 'r') as f:

            results["masks_index"] = json.load(f)

   

    return results

 

# CELL 5: The Main Function that Brings it all together

def run_propagation_evaluation(

    folder_id: str,

    frame_range: List[int],

    start_frame_index: int = 0,  # Which annotated frame to use (0 = first)

    use_mask_input: bool = False,

    separate_objects: bool = True,

    save_to_disk: bool = False,

    visualization_frames: List[int] = None,  # New parameter for user-specified frames

    sample_visualization: str = "uniform"

):

    """

    Main function to run the propagation evaluation pipeline.

   

    Args:

        folder_id: Video folder ID

        frame_range: [start_frame, end_frame] to evaluate

        start_frame_index: Which annotated frame to use as starting point (0 = first)

        use_mask_input: If True, use contour-based mask initialization

        separate_objects: If True, process each object separately

        save_to_disk: If True, save masks to disk instead of keeping in memory

        visualization_frames: Optional list of frame numbers to visualize (after init frame)

        sample_visualization: Strategy for visualizing frames ("uniform" or "key_frames")

    """

    # Update configuration

    CONFIG["folder_id"] = folder_id

    CONFIG["frame_range"] = frame_range

    CONFIG["start_frame_index"] = start_frame_index

    CONFIG["use_mask_input"] = use_mask_input

    CONFIG["separate_objects"] = separate_objects

    CONFIG["save_to_disk"] = save_to_disk

    CONFIG["sample_visualization"] = sample_visualization

    CONFIG["visualization_frames"] = visualization_frames

   

    # Remove reviewer_id from CONFIG as it's no longer needed

    if "reviewer_id" in CONFIG:

        del CONFIG["reviewer_id"]

   

    # Set custom results directory if provided

    if 'results_dir' in globals() and results_dir is not None:

        CONFIG["output_dir"] = results_dir

   

    logger.info(f"Starting propagation evaluation for video {folder_id}")

    logger.info(f"Frame range: {frame_range}")

    logger.info(f"Using {'mask' if use_mask_input else 'bbox+points'} initialization")

    logger.info(f"Object handling mode: {'separate inference states' if separate_objects else 'single inference state'}")

   

    try:

        # 1. Load segmentation data

        segmentation_data = load_segmentation_json(CONFIG["segmentation_json_path"])

       

        # 2. Get video data

        video_data, frames_data = get_video_data(segmentation_data, folder_id)

       

        # 3. Find annotated frames (now from any reviewer)

        annotated_frames = find_annotated_frames(frames_data, frame_range)

       

        if not annotated_frames:

            logger.error("No annotated frames found in the specified range")

            return

           

        logger.info(f"Found {len(annotated_frames)} annotated frames: {annotated_frames}")

       

        # 4. Select initialization frame

        if start_frame_index >= len(annotated_frames):

            logger.error(f"Start frame index {start_frame_index} out of range (max {len(annotated_frames)-1})")

            return

           

        init_frame = annotated_frames[start_frame_index]

        logger.info(f"Using frame {init_frame} as initialization frame")

       

        # 5. Extract annotations for initialization frame (from any reviewer)

        init_annotations = extract_annotation_data(frames_data, init_frame)

       

        if not init_annotations:

            logger.error(f"No annotations found for initialization frame {init_frame}")

            return

           

        logger.info(f"Found {len(init_annotations)} annotations for initialization frame")

       

        # 6. Resolve frame directory

        frame_dir = resolve_frame_directory(video_data['metadata']['preprocessing_directory'])

       

        # 7. Initialize SAM2

        predictor = initialize_sam2()

       

        # 8. Create frame sequence (all frames in range)

        frame_sequence = list(range(frame_range[0], frame_range[1] + 1))

       

        # 9. Propagate masks

        logger.info("Starting mask propagation...")

        propagated_masks = propagate_masks_from_annotation(

            predictor,

            frame_dir,

            frame_sequence,

            init_annotations,

            use_mask_input=CONFIG["use_mask_input"],

            separate_objects=CONFIG["separate_objects"]

        )

        logger.info(f"Completed propagation with {len(propagated_masks)} frames")

       

        # 10. Evaluate propagation

        logger.info("Evaluating propagation results...")

        metrics_df = evaluate_propagation(

            propagated_masks,

            frames_data,

            frame_range,

            frame_dir

        )

       

        # 11. Visualize results

        logger.info("Visualizing propagation results...")

        visualize_propagation_results(

            propagated_masks,

            frames_data,

            frame_range,

            frame_dir,

            metrics_df,

            max_frames=10,

            sample_strategy=CONFIG.get("sample_visualization", "uniform")

        )

       

        # Process visualization frames

        visualization_frame_indices = []

       

        # Handle specified visualization frames

        if visualization_frames:

            # Filter to include only frames that exist in annotated_frames

            valid_viz_frames = [f for f in visualization_frames if f in annotated_frames]

           

            # Make sure we don't include the initialization frame twice

            valid_viz_frames = [f for f in valid_viz_frames if f != init_frame]

           

            # Take up to 2 specified frames

            visualization_frame_indices = valid_viz_frames[:2]

           

            # If we don't have 2 frames, add more from annotated_frames

            if len(visualization_frame_indices) < 2:

                additional_frames = [

                    f for f in annotated_frames

                    if f != init_frame and f not in visualization_frame_indices

                ]

                visualization_frame_indices.extend(additional_frames[:2 - len(visualization_frame_indices)])

        else:

            # Default: use the next 2 annotated frames after init_frame

            additional_frames = [f for f in annotated_frames if f > init_frame]

            visualization_frame_indices = additional_frames[:2]

       

        # If we still don't have enough frames, just use what we have

        visualization_frame_indices = visualization_frame_indices[:2]  # Ensure max 2 frames

       

        # Create list with init frame first, then viz frames

        frames_to_viz = [init_frame] + visualization_frame_indices

       

        logger.info(f"Using frames {frames_to_viz} for pipeline visualization")

       

        # 12. Create pipeline overview visualization

        logger.info("Creating pipeline overview visualization...")

        visualize_sam2_pipeline_overview(

            propagated_masks,

            frames_data,

            frame_dir,

            metrics_df,

            init_annotations,

            frames_to_viz

        )

       

        # 13. Save results

        if CONFIG["save_intermediates"]:

            results_prefix = "sam2_eval"  # Default prefix

            results_path = save_evaluation_results(

                propagated_masks,

                metrics_df,

                CONFIG,

                CONFIG["output_dir"],

                results_prefix

            )

            logger.info(f"Results saved to: {results_path}")

       

        logger.info("Propagation evaluation completed successfully")

       

        # Return results for further analysis

        return {

            "propagated_masks": propagated_masks,

            "metrics_df": metrics_df,

            "annotated_frames": annotated_frames,

            "init_frame": init_frame,

            "init_annotations": init_annotations,

            "visualization_frames": frames_to_viz

        }

       

    except Exception as e:

        logger.error(f"Error in propagation evaluation: {str(e)}")

        import traceback

        traceback.print_exc()

        return None

    finally:

        # Clean up resources

        if 'predictor' in locals():

            # Clean up predictor if it has a cleanup method

            if hasattr(predictor, 'cleanup') and callable(predictor.cleanup):

                predictor.cleanup()

       

        # Force garbage collection

        import gc

        gc.collect()

       

        # Clear CUDA cache

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

 

 

 

# # Example usage with different configurations

# if __name__ == "__main__":

#     # Example 1: Run with BBox+Points initialization and separate object states

#     results_separate = run_propagation_evaluation(

#         folder_id="Janssen_Site-IT10002_IT10002005_Time-Early_Termination_2019-10-07_10_05_15",

#         frame_range=[16000, 16500],

#         reviewer_id="Reviewer 5",

#         start_frame_index=0,

#         use_mask_input=False,  # Use bbox+points initialization

#         separate_objects=True  # Process each object with separate inference state

#     )

 

#     # Example 2: Run with BBox+Points initialization and unified object state

#     results_unified = run_propagation_evaluation(

#         folder_id="Janssen_Site-IT10002_IT10002005_Time-Early_Termination_2019-10-07_10_05_15",

#         frame_range=[16000, 16500],

#         reviewer_id="Reviewer 5",

#         start_frame_index=0,

#         use_mask_input=False,  # Use bbox+points initialization

#         separate_objects=False  # Process all objects in a single inference state

#     )

 

#     # Example 3: Run with mask-based initialization

#     results_mask = run_propagation_evaluation(

#         folder_id="Janssen_Site-IT10002_IT10002005_Time-Early_Termination_2019-10-07_10_05_15",

#         frame_range=[16000, 16500],

#         reviewer_id="Reviewer 5",

#         start_frame_index=0,

#         use_mask_input=True,  # Use mask initialization

#         separate_objects=True  # Process each object with separate inference state

#     )

 

#     # Example 4: Longer sequence with disk storage for memory management

#     results_long = run_propagation_evaluation(

#         folder_id="Janssen_Site-IT10002_IT10002005_Time-Early_Termination_2019-10-07_10_05_15",

#         frame_range=[16000, 16800],  # 800 frame range - longer sequence

#         reviewer_id="Reviewer 5",

#         start_frame_index=0,

#         use_mask_input=False,

#         separate_objects=True,

#         save_to_disk=True  # Save masks to disk instead of keeping in memory

#     )

 

#     # Compare object handling approaches

#     if results_separate is not None and results_unified is not None:

#         plt.figure(figsize=(12, 6))

#         plt.plot(

#             results_separate['metrics_df']['frames_from_start'], results_separate['metrics_df']['dice'],

#             'o-', label='Separate Object States'

#         )

#         plt.plot(

#             results_unified['metrics_df']['frames_from_start'], results_unified['metrics_df']['dice'],

#             's-', label='Unified Object State'

#         )

#         plt.xlabel('Frames from Initialization')

#         plt.ylabel('DICE Score')

#         plt.title('Comparison of Object Handling Approaches')

#         plt.grid(True, alpha=0.3)

#         plt.legend()

#         plt.tight_layout()

#         plt.show()

 

#     # Compare initialization methods

#     if results_separate is not None and results_mask is not None:

#         plt.figure(figsize=(12, 6))

#         plt.plot(

#             results_separate['metrics_df']['frames_from_start'], results_separate['metrics_df']['dice'],

#             'o-', label='BBox+Points Init'

#         )

#         plt.plot(

#             results_mask['metrics_df']['frames_from_start'], results_mask['metrics_df']['dice'],

#             's-', label='Mask Init'

#         )

#         plt.xlabel('Frames from Initialization')

#         plt.ylabel('DICE Score')

#         plt.title('Comparison of Initialization Methods')

#         plt.grid(True, alpha=0.3)

#         plt.legend()

#         plt.tight_layout()

#         plt.show()

 

    # Example of loading previous results

 

# CELL 6 EXECUTION

# Example usage with different configurations

if __name__ == "__main__":

    # Example 1: Run with BBox+Points initialization and separate object states

 

    results_separate = run_propagation_evaluation(

        folder_id="67586d5f38ca792feb727a9b",

        frame_range=[1, 25],

        start_frame_index=0,

        use_mask_input=False,  # Use bbox+points initialization

        separate_objects=True  # Process each object with separate inference state

    )

    results_separate = run_propagation_evaluation(

        folder_id="67586d5f38ca792feb727a9b",

        frame_range=[1, 25],

        start_frame_index=0,

        use_mask_input=False,  # Use bbox+points initialization

        separate_objects=False  # Process each object with separate inference state

    )

 

 