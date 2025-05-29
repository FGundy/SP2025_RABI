# sam2_worker/sam2_service/tasks.py - Updated with model selection
import json
import logging
import os
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

# Get celery app from worker module
from worker import celery as celery_app

@celery_app.task(bind=True, queue='sam2_queue')
def initialize_video_analysis(
    self, 
    analysis_session_id: str, 
    video_file_path: str, 
    sam2_model: str = 'base_plus'
) -> Dict[str, Any]:
    """
    Initialize SAM 2 analysis session for a video with user-selected model.
    
    Args:
        analysis_session_id: Django AnalysisSession ID
        video_file_path: Path to video file in MinIO
        sam2_model: SAM 2 model size (tiny, small, base_plus, large)
        
    Returns:
        Dictionary with sam2_session_id and status
    """
    try:
        logger.info(f"🚀 Starting SAM 2 {sam2_model.upper()} initialization for session {analysis_session_id}")
        
        # Download video file from MinIO to local storage
        local_video_path = f"/tmp/{analysis_session_id}.mp4"
        
        # Get video from MinIO
        minio_client.fget_object(
            bucket_name="videos",
            object_name=video_file_path,
            file_path=local_video_path
        )
        
        logger.info(f"📥 Downloaded video to {local_video_path}")
        
        # Initialize SAM 2 service with user-selected model
        sam2_service_instance = ThermalSegmentationAPI(
            minio_client=minio_client,
            model_size=sam2_model  # Pass user selection
        )
        
        # Initialize SAM 2 session with memory-efficient settings
        sam2_session_id = sam2_service_instance.start_session(
            video_path=local_video_path,
            analysis_session_id=analysis_session_id
        )
        
        logger.info(f"✅ Initialized SAM 2 ({sam2_model}) session {sam2_session_id} for analysis {analysis_session_id}")
        
        # Update Django model status (you'll need to import your models)
        # from apps.analysis.models import AnalysisSession
        # try:
        #     session = AnalysisSession.objects.get(id=analysis_session_id)
        #     session.sam2_session_id = sam2_session_id
        #     session.status = 'active'
        #     session.save()
        # except AnalysisSession.DoesNotExist:
        #     logger.error(f"Analysis session {analysis_session_id} not found")
        
        return {
            "sam2_session_id": sam2_session_id,
            "status": "initialized",
            "analysis_session_id": analysis_session_id,
            "model_used": sam2_model,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize video analysis: {e}")
        
        # Update Django model with error
        # try:
        #     session = AnalysisSession.objects.get(id=analysis_session_id)
        #     session.status = 'error'
        #     session.error_message = str(e)
        #     session.save()
        # except:
        #     pass
        
        # Retry with smaller model if current model failed due to memory
        if "out of memory" in str(e).lower() or "exit status 137" in str(e).lower():
            smaller_models = {
                'large': 'base_plus',
                'base_plus': 'small', 
                'small': 'tiny'
            }
            
            if sam2_model in smaller_models:
                logger.info(f"🔄 Retrying with smaller model: {smaller_models[sam2_model]}")
                return initialize_video_analysis(
                    analysis_session_id, 
                    video_file_path, 
                    smaller_models[sam2_model]
                )
        
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
        # Initialize SAM 2 service (you might want to cache this)
        sam2_service_instance = ThermalSegmentationAPI(minio_client=minio_client)
        
        result = sam2_service_instance.add_points(
            session_id=sam2_session_id,
            frame_index=frame_index,
            object_id=object_id,
            points=points,
            labels=labels,
            clear_old_points=clear_old_points
        )
        
        logger.info(f"✅ Added points to frame {frame_index} in session {sam2_session_id}")
        
        # TODO: Save results to Django database
        # Create SegmentationResult objects from the RLE masks
        
        return {
            "sam2_session_id": sam2_session_id,
            "frame_index": result["frame_index"],
            "masks": result["results"],
            "status": "segmented",
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to add segmentation points: {e}")
        self.retry(countdown=30, max_retries=3)


@celery_app.task(bind=True, queue='sam2_queue')
def propagate_segmentation(
    self,
    sam2_session_id: str,
    start_frame_index: int = 0
) -> Dict[str, Any]:
    """
    Propagate segmentation across the entire video.
    
    Args:
        sam2_session_id: SAM 2 session ID
        start_frame_index: Frame to start propagation from
        
    Returns:
        Dictionary with propagation results
    """
    try:
        logger.info(f"🎯 Starting propagation for session {sam2_session_id} from frame {start_frame_index}")
        
        # Initialize SAM 2 service
        sam2_service_instance = ThermalSegmentationAPI(minio_client=minio_client)
        
        frame_count = 0
        total_objects = 0
        
        # Process propagation results
        for result in sam2_service_instance.propagate_in_video(
            session_id=sam2_session_id,
            start_frame_index=start_frame_index
        ):
            frame_idx = result["frame_index"]
            masks = result["results"]
            
            frame_count += 1
            total_objects += len(masks)
            
            # TODO: Save each frame's results to Django database
            # Create SegmentationResult objects for each mask
            
            # Update progress every 50 frames
            if frame_count % 50 == 0:
                logger.info(f"📈 Processed {frame_count} frames, {total_objects} objects")
                
                # TODO: Update Django model progress
                # try:
                #     session = AnalysisSession.objects.get(sam2_session_id=sam2_session_id)
                #     session.total_frames_analyzed = frame_count
                #     session.total_objects_tracked = total_objects
                #     session.save()
                # except:
                #     pass
        
        logger.info(f"✅ Propagation completed: {frame_count} frames, {total_objects} objects")
        
        # TODO: Update final Django model status
        # try:
        #     session = AnalysisSession.objects.get(sam2_session_id=sam2_session_id)
        #     session.status = 'completed'
        #     session.total_frames_analyzed = frame_count
        #     session.total_objects_tracked = total_objects
        #     session.save()
        # except:
        #     pass
        
        return {
            "sam2_session_id": sam2_session_id,
            "frames_processed": frame_count,
            "objects_tracked": total_objects,
            "status": "completed",
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Propagation failed: {e}")
        
        # TODO: Update Django model with error
        # try:
        #     session = AnalysisSession.objects.get(sam2_session_id=sam2_session_id)
        #     session.status = 'error'
        #     session.error_message = str(e)
        #     session.save()
        # except:
        #     pass
        
        self.retry(countdown=60, max_retries=2)


@celery_app.task(bind=True, queue='sam2_queue')
def test_sam2_with_model(self, model_size: str = 'tiny') -> Dict[str, Any]:
    """
    Test SAM 2 with a specific model size
    
    Args:
        model_size: Model size to test (tiny, small, base_plus, large)
        
    Returns:
        Dictionary with test results
    """
    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        from pathlib import Path
        
        model_configs = {
            'tiny': ('configs/sam2.1/sam2.1_hiera_t.yaml', 'sam2.1_hiera_tiny.pt'),
            'small': ('configs/sam2.1/sam2.1_hiera_s.yaml', 'sam2.1_hiera_small.pt'),
            'base_plus': ('configs/sam2.1/sam2.1_hiera_b+.yaml', 'sam2.1_hiera_base_plus.pt'),
            'large': ('configs/sam2.1/sam2.1_hiera_l.yaml', 'sam2.1_hiera_large.pt'),
        }
        
        if model_size not in model_configs:
            return {"error": f"Invalid model size: {model_size}"}
        
        config_file, checkpoint_file = model_configs[model_size]
        checkpoint_path = Path(f"/opt/sam2_worker/sam2/checkpoints/{checkpoint_file}")
        
        result = {
            "model_size": model_size,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "checkpoint_exists": checkpoint_path.exists(),
            "status": "testing"
        }
        
        if torch.cuda.is_available():
            result["cuda_device"] = torch.cuda.get_device_name()
            free_memory, total_memory = torch.cuda.mem_get_info()
            result["gpu_memory_free"] = f"{free_memory / 1e9:.1f}GB"
            result["gpu_memory_total"] = f"{total_memory / 1e9:.1f}GB"
        
        # Try to load the model
        if checkpoint_path.exists():
            predictor = build_sam2_video_predictor(config_file, checkpoint_path, device='cuda')
            result["model_loaded"] = True
            result["status"] = "success"
            
            if torch.cuda.is_available():
                free_after, _ = torch.cuda.mem_get_info()
                model_memory = (free_memory - free_after) / 1e9
                result["model_memory_usage"] = f"{model_memory:.1f}GB"
        else:
            result["model_loaded"] = False
            result["error"] = f"Checkpoint not found: {checkpoint_path}"
            
        logger.info(f"SAM 2 {model_size} test completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"SAM 2 {model_size} test failed: {e}")
        return {
            "model_size": model_size,
            "status": "error",
            "error": str(e)
        }