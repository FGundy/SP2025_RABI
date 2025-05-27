"""
Celery tasks for SAM 2 video segmentation
"""
# sam2_worker/sam2_service/tasks.py
import json
import logging
import os  # ADD THIS MISSING IMPORT
from typing import List, Dict, Any

from celery import Celery
from minio import Minio

from sam2_service.predictor import ThermalSegmentationAPI

logger = logging.getLogger(__name__)

# Initialize MinIO client
minio_client = Minio(
    endpoint=os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
    access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),  
    secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin123'),
    secure=False
)

# Initialize SAM 2 service
sam2_service = ThermalSegmentationAPI(minio_client=minio_client)

# Get celery app from worker module
from worker import celery as celery_app

@celery_app.task(bind=True, queue='sam2_queue')
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


@celery_app.task(bind=True, queue='sam2_queue')
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


@celery_app.task(bind=True, queue='sam2_queue')
def test_sam2_basic(self) -> Dict[str, Any]:
    """
    Basic test task to verify SAM 2 is working
    
    Returns:
        Dictionary with test results
    """
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        
        result = {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "sam2_import": "success",
            "status": "ready"
        }
        
        if torch.cuda.is_available():
            result["cuda_device"] = torch.cuda.get_device_name()
            
        logger.info("SAM 2 basic test completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"SAM 2 basic test failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }