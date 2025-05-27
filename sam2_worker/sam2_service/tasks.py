"""
Celery tasks for SAM 2 video segmentation
"""

import json
import logging
from typing import List, Dict, Any

from celery import Celery
from minio import Minio

from sam2_service.predictor import ThermalSegmentationAPI

logger = logging.getLogger(__name__)

# Initialize Celery app
app = Celery('sam2_worker')
app.config_from_object('celeryconfig')

# Initialize MinIO client
minio_client = Minio(
    endpoint=os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
    access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
    secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin123'),
    secure=False
)

# Initialize SAM 2 service
sam2_service = ThermalSegmentationAPI(minio_client=minio_client)


@app.task(bind=True, queue='sam2_queue')
def initialize_video_analysis(self, analysis_session_id: str, video_file_path: str) -> Dict[str, Any]:
    """
    Initialize SAM 2 analysis session for a video.
    
    Args:
        analysis_session_id: Django AnalysisSession ID
        video_file_path: Path to video file in MinIO
        
    Returns:
        Dictionary with sam2_session_id and status
    """
    try:
        # Download video file from MinIO to local storage
        local_video_path = f"/tmp/{analysis_session_id}.mp4"
        
        # Get video from MinIO
        minio_client.fget_object(
            bucket_name="videos",
            object_name=video_file_path,
            file_path=local_video_path
        )
        
        # Initialize SAM 2 session
        sam2_session_id = sam2_service.start_session(
            video_path=local_video_path,
            analysis_session_id=analysis_session_id
        )
        
        logger.info(f"Initialized SAM 2 session {sam2_session_id} for analysis {analysis_session_id}")
        
        return {
            "sam2_session_id": sam2_session_id,
            "status": "initialized",
            "analysis_session_id": analysis_session_id
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize video analysis: {e}")
        self.retry(countdown=60, max_retries=3)


@app.task(bind=True, queue='sam2_queue')
def add_segmentation_points(
    self,
    sam2_session_id: str,
    frame_index: int,
    object_id: int,
    points: List[List[float]],
    labels: List[int],
    clear_old_points: bool = False
) -> Dict[str, Any]:
    """
    Add segmentation points to a specific frame.
    
    Args:
        sam2_session_id: SAM 2 session ID
        frame_index: Frame number
        object_id: Object identifier
        points: List of [x, y] coordinates
        labels: List of point labels (1 for positive, 0 for negative)
        clear_old_points: Whether to clear existing points
        
    Returns:
        Dictionary with segmentation results
    """
    try:
        result = sam2_service.add_points(
            session_id=sam2_session_id,
            frame_index=frame_index,
            object_id=object_id,
            points=points,
            labels=labels,
            clear_old_points=clear_old_points
        )
        
        logger.info(f"Added points to frame {frame_index} in session {sam2_session_id}")
        
        return {
            "sam2_session_id": sam2_session_id,
            "frame_index": result["frame_index"],
            "masks": result["results"],
            "status": "segmented"
        }
        
    except Exception as e:
        logger.error(f"Failed to add segmentation points: {e}")
        self.retry(countdown=30, max_retries=3)


@app.task(bind=True, queue='sam2_queue')
def propagate_segmentation(
    self,
    sam2_session_id: str,
    start_frame_index: int,
    websocket_group: str = None
) -> Dict[str, Any]:
    """
    Propagate segmentation across the entire video.
    
    Args:
        sam2_session_id: SAM 2 session ID
        start_frame_index: Frame to start propagation from
        websocket_group: WebSocket group name for real-time updates
        
    Returns:
        Dictionary with propagation summary
    """
    try:
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync
        
        channel_layer = get_channel_layer()
        frames_processed = 0
        total_objects = 0
        
        # Send initial status
        if websocket_group and channel_layer:
            async_to_sync(channel_layer.group_send)(
                websocket_group,
                {
                    "type": "propagation_started",
                    "sam2_session_id": sam2_session_id,
                    "start_frame": start_frame_index
                }
            )
        
        # Process propagation results as they come in
        for result in sam2_service.propagate_in_video(sam2_session_id, start_frame_index):
            frames_processed += 1
            frame_index = result["frame_index"]
            masks = result["results"]
            total_objects = max(total_objects, len(masks))
            
            # Send real-time update via WebSocket
            if websocket_group and channel_layer:
                async_to_sync(channel_layer.group_send)(
                    websocket_group,
                    {
                        "type": "propagation_update",
                        "sam2_session_id": sam2_session_id,
                        "frame_index": frame_index,
                        "masks": masks,
                        "progress": {
                            "frames_processed": frames_processed,
                            "current_frame": frame_index
                        }
                    }
                )
            
            # Update progress every 10 frames
            if frames_processed % 10 == 0:
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'frames_processed': frames_processed,
                        'current_frame': frame_index,
                        'total_objects': total_objects
                    }
                )
        
        # Send completion status
        if websocket_group and channel_layer:
            async_to_sync(channel_layer.group_send)(
                websocket_group,
                {
                    "type": "propagation_completed",
                    "sam2_session_id": sam2_session_id,
                    "frames_processed": frames_processed,
                    "total_objects": total_objects
                }
            )
        
        logger.info(f"Completed propagation for session {sam2_session_id}: {frames_processed} frames, {total_objects} objects")
        
        return {
            "sam2_session_id": sam2_session_id,
            "status": "completed",
            "frames_processed": frames_processed,
            "total_objects": total_objects
        }
        
    except Exception as e:
        logger.error(f"Failed to propagate segmentation: {e}")
        
        # Send error status via WebSocket
        if websocket_group and channel_layer:
            async_to_sync(channel_layer.group_send)(
                websocket_group,
                {
                    "type": "propagation_error",
                    "sam2_session_id": sam2_session_id,
                    "error": str(e)
                }
            )
        
        self.retry(countdown=60, max_retries=2)


@app.task(bind=True, queue='sam2_queue')
def analyze_thermal_data(
    self,
    analysis_session_id: str,
    segmentation_results: List[Dict[str, Any]],
    thermal_video_path: str = None
) -> Dict[str, Any]:
    """
    Analyze thermal data for segmented regions.
    
    Args:
        analysis_session_id: Django AnalysisSession ID
        segmentation_results: List of segmentation results with masks
        thermal_video_path: Path to thermal video file (if different from visual)
        
    Returns:
        Dictionary with thermal analysis results
    """
    try:
        import cv2
        import numpy as np
        from pycocotools.mask import decode as decode_masks
        
        thermal_data = []
        
        # Process each segmentation result
        for result in segmentation_results:
            frame_index = result["frame_index"]
            masks = result["masks"]
            
            # Load thermal frame (this is simplified - real implementation
            # would need proper thermal video decoding)
            if thermal_video_path:
                cap = cv2.VideoCapture(thermal_video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, thermal_frame = cap.read()
                cap.release()
                
                if ret:
                    # Convert to grayscale if needed (thermal might be single channel)
                    if len(thermal_frame.shape) == 3:
                        thermal_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Analyze thermal data for each mask
                    for mask_data in masks:
                        # Decode RLE mask
                        rle_mask = {
                            "counts": mask_data["mask"]["counts"],
                            "size": mask_data["mask"]["size"]
                        }
                        mask = decode_masks(rle_mask)
                        
                        # Extract thermal values within mask
                        masked_thermal = thermal_frame[mask > 0]
                        
                        if len(masked_thermal) > 0:
                            thermal_analysis = {
                                "frame_index": frame_index,
                                "object_id": mask_data["object_id"],
                                "avg_temperature": float(np.mean(masked_thermal)),
                                "min_temperature": float(np.min(masked_thermal)),
                                "max_temperature": float(np.max(masked_thermal)),
                                "std_temperature": float(np.std(masked_thermal)),
                                "pixel_count": len(masked_thermal)
                            }
                            
                            # Detect thermal anomalies
                            temp_deviation = abs(thermal_analysis["avg_temperature"] - np.mean(thermal_frame))
                            thermal_analysis["anomaly_score"] = min(temp_deviation / 10.0, 1.0)  # Simplified scoring
                            
                            thermal_data.append(thermal_analysis)
        
        logger.info(f"Analyzed thermal data for {len(thermal_data)} segmented regions in session {analysis_session_id}")
        
        return {
            "analysis_session_id": analysis_session_id,
            "thermal_data": thermal_data,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze thermal data: {e}")
        self.retry(countdown=60, max_retries=3)


@app.task(bind=True, queue='sam2_queue')
def close_analysis_session(self, sam2_session_id: str) -> Dict[str, Any]:
    """
    Close SAM 2 analysis session and cleanup resources.
    
    Args:
        sam2_session_id: SAM 2 session ID to close
        
    Returns:
        Dictionary with closure status
    """
    try:
        success = sam2_service.close_session(sam2_session_id)
        
        logger.info(f"Closed SAM 2 session {sam2_session_id}")
        
        return {
            "sam2_session_id": sam2_session_id,
            "status": "closed" if success else "not_found"
        }
        
    except Exception as e:
        logger.error(f"Failed to close analysis session: {e}")
        return {
            "sam2_session_id": sam2_session_id,
            "status": "error",
            "error": str(e)
        }


# Task routing configuration
app.conf.update(
    task_routes={
        'sam2_worker.tasks.initialize_video_analysis': {'queue': 'sam2_queue'},
        'sam2_worker.tasks.add_segmentation_points': {'queue': 'sam2_queue'},
        'sam2_worker.tasks.propagate_segmentation': {'queue': 'sam2_queue'},
        'sam2_worker.tasks.analyze_thermal_data': {'queue': 'sam2_queue'},
        'sam2_worker.tasks.close_analysis_session': {'queue': 'sam2_queue'},
    }
)