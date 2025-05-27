"""
SAM 2 Service for Thermal Building Analysis
Adapted from Meta's SAM 2 demo predictor.py
"""

import contextlib
import logging
import os
import pickle
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import torch
from minio import Minio
from pycocotools.mask import decode as decode_masks, encode as encode_masks
from sam2.build_sam import build_sam2_video_predictor

logger = logging.getLogger(__name__)


class ThermalSegmentationAPI:
    """
    SAM 2 predictor service adapted for thermal building analysis.
    Handles video segmentation with persistent state management.
    """

    def __init__(self, minio_client: Optional[Minio] = None):
        super(ThermalSegmentationAPI, self).__init__()
        
        # In-memory session cache (for active sessions)
        self.session_states: Dict[str, Any] = {}
        self.score_thresh = 0
        self.inference_lock = Lock()
        
        # MinIO client for persistent state storage
        self.minio_client = minio_client
        self.state_bucket = "sam2-states"
        
        # Initialize SAM 2 model (adapted from SAM 2's predictor.py)
        self._initialize_sam2_model()
        
        # Ensure MinIO bucket exists
        if self.minio_client:
            self._ensure_bucket_exists()

    def _initialize_sam2_model(self):
        """Initialize SAM 2 model with proper device selection"""
        model_size = os.getenv("MODEL_SIZE", "base_plus")
        
        # Model configuration (from SAM 2's predictor.py)
        if model_size == "tiny":
            checkpoint = Path("/opt/sam2_worker/sam2/checkpoints/sam2.1_hiera_tiny.pt")
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        elif model_size == "small":
            checkpoint = Path("/opt/sam2_worker/sam2/checkpoints/sam2.1_hiera_small.pt")
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif model_size == "large":
            checkpoint = Path("/opt/sam2_worker/sam2/checkpoints/sam2.1_hiera_large.pt")
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        else:  # base_plus (default)
            checkpoint = Path("/opt/sam2_worker/sam2/checkpoints/sam2.1_hiera_base_plus.pt")
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

        # Device selection (from SAM 2's predictor.py)
        force_cpu_device = os.environ.get("SAM2_DEMO_FORCE_CPU_DEVICE", "0") == "1"
        if force_cpu_device:
            logger.info("forcing CPU device for SAM 2")
        
        if torch.cuda.is_available() and not force_cpu_device:
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and not force_cpu_device:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        logger.info(f"using device: {device}")

        # Device-specific optimizations (from SAM 2's predictor.py)
        if device.type == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            logging.warning(
                "Support for MPS devices is preliminary. SAM 2 might give numerically "
                "different outputs and sometimes degraded performance on MPS."
            )

        self.device = device
        self.predictor = build_sam2_video_predictor(
            model_cfg, checkpoint, device=device
        )

    def _ensure_bucket_exists(self):
        """Ensure MinIO bucket exists for state storage"""
        if not self.minio_client.bucket_exists(self.state_bucket):
            self.minio_client.make_bucket(self.state_bucket)

    def autocast_context(self):
        """Device-appropriate autocast context"""
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            return contextlib.nullcontext()

    def start_session(self, video_path: str, analysis_session_id: str) -> str:
        """
        Initialize a new SAM 2 session for a video.
        
        Args:
            video_path: Path to the video file
            analysis_session_id: Django AnalysisSession ID for persistence
            
        Returns:
            session_id: Unique session identifier
        """
        with self.autocast_context(), self.inference_lock:
            session_id = str(uuid.uuid4())
            
            # Initialize SAM 2 inference state
            offload_video_to_cpu = self.device.type == "mps"
            inference_state = self.predictor.init_state(
                video_path,
                offload_video_to_cpu=offload_video_to_cpu,
            )
            
            # Store session state
            self.session_states[session_id] = {
                "canceled": False,
                "state": inference_state,
                "analysis_session_id": analysis_session_id,
                "video_path": video_path,
            }
            
            # Save initial state to MinIO
            self._save_session_state(session_id)
            
            logger.info(f"Started SAM 2 session {session_id} for analysis {analysis_session_id}")
            return session_id

    def add_points(
        self,
        session_id: str,
        frame_index: int,
        object_id: int,
        points: List[List[float]],
        labels: List[int],
        clear_old_points: bool = False,
    ) -> Dict[str, Any]:
        """
        Add points to a frame and get immediate segmentation result.
        
        Args:
            session_id: SAM 2 session ID
            frame_index: Frame number
            object_id: Object identifier
            points: List of [x, y] coordinates
            labels: List of point labels (1 for positive, 0 for negative)
            clear_old_points: Whether to clear existing points
            
        Returns:
            Dictionary with frame_index and RLE masks
        """
        with self.autocast_context(), self.inference_lock:
            session = self._get_session(session_id)
            inference_state = session["state"]

            # Add points using SAM 2's API
            frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_index,
                obj_id=object_id,
                points=points,
                labels=labels,
                clear_old_points=clear_old_points,
                normalize_coords=False,
            )

            masks_binary = (masks > self.score_thresh)[:, 0].cpu().numpy()
            rle_mask_list = self._get_rle_mask_list(
                object_ids=object_ids, masks=masks_binary
            )
            
            # Save updated state
            self._save_session_state(session_id)
            
            return {
                "frame_index": frame_idx,
                "results": rle_mask_list,
            }

    def propagate_in_video(
        self, session_id: str, start_frame_index: int
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Propagate segmentation across the entire video.
        
        Args:
            session_id: SAM 2 session ID
            start_frame_index: Frame to start propagation from
            
        Yields:
            Dictionary with frame_index and RLE masks for each frame
        """
        with self.autocast_context(), self.inference_lock:
            session = self._get_session(session_id)
            session["canceled"] = False
            inference_state = session["state"]

            try:
                # Forward propagation
                for outputs in self.predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=start_frame_index,
                    reverse=False,
                ):
                    if session["canceled"]:
                        return

                    frame_idx, obj_ids, video_res_masks = outputs
                    masks_binary = (
                        (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()
                    )

                    rle_mask_list = self._get_rle_mask_list(
                        object_ids=obj_ids, masks=masks_binary
                    )

                    yield {
                        "frame_index": frame_idx,
                        "results": rle_mask_list,
                    }

                # Backward propagation
                for outputs in self.predictor.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=start_frame_index,
                    reverse=True,
                ):
                    if session["canceled"]:
                        return

                    frame_idx, obj_ids, video_res_masks = outputs
                    masks_binary = (
                        (video_res_masks > self.score_thresh)[:, 0].cpu().numpy()
                    )

                    rle_mask_list = self._get_rle_mask_list(
                        object_ids=obj_ids, masks=masks_binary
                    )

                    yield {
                        "frame_index": frame_idx,
                        "results": rle_mask_list,
                    }

            finally:
                # Save final state
                self._save_session_state(session_id)
                logger.info(f"Propagation completed for session {session_id}")

    def close_session(self, session_id: str) -> bool:
        """Close a SAM 2 session and cleanup resources"""
        session = self.session_states.pop(session_id, None)
        if session is None:
            logger.warning(f"Session {session_id} not found for closure")
            return False
        
        logger.info(f"Closed SAM 2 session {session_id}")
        return True

    def _get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session state, loading from MinIO if necessary"""
        session = self.session_states.get(session_id)
        if session is None:
            # Try to load from MinIO
            session = self._load_session_state(session_id)
            if session is None:
                raise RuntimeError(f"Session {session_id} not found")
        return session

    def _save_session_state(self, session_id: str):
        """Save session state to MinIO for persistence"""
        if not self.minio_client:
            return
            
        session = self.session_states.get(session_id)
        if not session:
            return
            
        # Serialize the inference state (this is the tricky part)
        # For now, we'll save the session metadata and reconstruct state as needed
        state_data = {
            "analysis_session_id": session["analysis_session_id"],
            "video_path": session["video_path"],
            "canceled": session["canceled"],
        }
        
        state_bytes = pickle.dumps(state_data)
        object_name = f"session_{session_id}.pkl"
        
        try:
            from io import BytesIO
            self.minio_client.put_object(
                self.state_bucket,
                object_name,
                BytesIO(state_bytes),
                length=len(state_bytes),
            )
        except Exception as e:
            logger.error(f"Failed to save session state: {e}")

    def _load_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session state from MinIO"""
        if not self.minio_client:
            return None
            
        object_name = f"session_{session_id}.pkl"
        
        try:
            response = self.minio_client.get_object(self.state_bucket, object_name)
            state_data = pickle.loads(response.read())
            
            # Reconstruct the session (this is simplified - real implementation
            # would need to properly restore the SAM 2 inference state)
            video_path = state_data["video_path"]
            
            offload_video_to_cpu = self.device.type == "mps"
            inference_state = self.predictor.init_state(
                video_path,
                offload_video_to_cpu=offload_video_to_cpu,
            )
            
            session = {
                "canceled": state_data["canceled"],
                "state": inference_state,
                "analysis_session_id": state_data["analysis_session_id"],
                "video_path": video_path,
            }
            
            self.session_states[session_id] = session
            return session
            
        except Exception as e:
            logger.error(f"Failed to load session state: {e}")
            return None

    def _get_rle_mask_list(self, object_ids: List[int], masks: np.ndarray) -> List[Dict]:
        """Convert masks to RLE format (from SAM 2's predictor.py)"""
        return [
            self._get_mask_for_object(object_id=object_id, mask=mask)
            for object_id, mask in zip(object_ids, masks)
        ]

    def _get_mask_for_object(self, object_id: int, mask: np.ndarray) -> Dict:
        """Create RLE mask data for an object (from SAM 2's predictor.py)"""
        mask_rle = encode_masks(np.array(mask, dtype=np.uint8, order="F"))
        mask_rle["counts"] = mask_rle["counts"].decode()
        
        return {
            "object_id": object_id,
            "mask": {
                "size": mask_rle["size"],
                "counts": mask_rle["counts"],
            },
        }