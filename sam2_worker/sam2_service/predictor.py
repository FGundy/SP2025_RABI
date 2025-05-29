# sam2_worker/sam2_service/predictor.py - UNIFIED VERSION
"""
Unified SAM 2 Service with Common Base Class
Supports both standard and hybrid GPU/CPU processing
"""
import contextlib
import logging
import os
import pickle
import uuid
import gc
import subprocess
import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Generator, List, Optional
from abc import ABC, abstractmethod

import numpy as np
import torch
from minio import Minio
from pycocotools.mask import decode as decode_masks, encode as encode_masks
from sam2.build_sam import build_sam2_video_predictor

logger = logging.getLogger(__name__)

class BaseThermalSegmentationAPI(ABC):
    """
    Base class for all SAM 2 thermal segmentation APIs
    Provides common functionality and interface
    """
    
    def __init__(self, minio_client: Optional[Minio] = None, model_size: str = 'base_plus'):
        self.model_size = model_size
        self.minio_client = minio_client
        self.session_states: Dict[str, Any] = {}
        self.score_thresh = 0
        self.inference_lock = Lock()
        self.state_bucket = "sam2-states"
        
        # Initialize SAM 2 model
        self._initialize_sam2_model()
        
        # Ensure MinIO bucket exists
        if self.minio_client:
            self._ensure_bucket_exists()
    
    @abstractmethod
    def _initialize_sam2_model(self):
        """Initialize SAM 2 model - implemented by subclasses"""
        pass
    
    @abstractmethod
    def start_session(self, video_path: str, analysis_session_id: str) -> str:
        """Start SAM 2 session - implemented by subclasses"""
        pass
    
    # Common utility methods that all subclasses can use
    def get_video_info_safe(self, video_path: str) -> Dict[str, Any]:
        """Get video information safely without loading the entire video."""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', video_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
                
                if video_stream:
                    return {
                        'width': int(video_stream.get('width', 0)),
                        'height': int(video_stream.get('height', 0)),
                        'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                        'duration': float(data['format']['duration']),
                        'frame_count': int(float(data['format']['duration']) * eval(video_stream.get('r_frame_rate', '0/1'))),
                        'codec': video_stream.get('codec_name'),
                        'is_4k': int(video_stream.get('width', 0)) >= 3840,
                    }
            
            logger.warning(f"Could not get video info for {video_path}")
            return {}
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}
    
    # Alias for backward compatibility
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Alias for get_video_info_safe for backward compatibility"""
        return self.get_video_info_safe(video_path)
    
    def clear_gpu_cache(self):
        """Clear GPU memory cache to free up VRAM"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get memory info
            free_memory, total_memory = torch.cuda.mem_get_info()
            used_memory = total_memory - free_memory
            
            logger.debug(f"🔧 GPU Memory: {used_memory / 1e9:.1f}GB used / {total_memory / 1e9:.1f}GB total")
    
    def cleanup_old_frame_outputs(self, inference_state: Dict[str, Any], max_frames: int = 5):
        """Clean up old frame outputs to prevent memory accumulation."""
        for obj_idx, obj_output_dict in inference_state.get("output_dict_per_obj", {}).items():
            non_cond_outputs = obj_output_dict.get("non_cond_frame_outputs", {})
            
            if len(non_cond_outputs) > max_frames:
                frame_indices = sorted(non_cond_outputs.keys())
                frames_to_remove = frame_indices[:-max_frames]
                
                for frame_idx in frames_to_remove:
                    del non_cond_outputs[frame_idx]
                
                logger.debug(f"🧹 Cleaned {len(frames_to_remove)} old frame outputs for object {obj_idx}")
    
    def autocast_context(self):
        """Device-appropriate autocast context"""
        if hasattr(self, 'device') and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            return contextlib.nullcontext()
    
    def _get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session state"""
        session = self.session_states.get(session_id)
        if session is None:
            raise RuntimeError(f"Session {session_id} not found")
        return session
    
    def _get_rle_mask_list(self, object_ids: List[int], masks: np.ndarray) -> List[Dict]:
        """Convert masks to RLE format"""
        return [
            self._get_mask_for_object(object_id=object_id, mask=mask)
            for object_id, mask in zip(object_ids, masks)
        ]

    def _get_mask_for_object(self, object_id: int, mask: np.ndarray) -> Dict:
        """Create RLE mask data for an object"""
        mask_rle = encode_masks(np.array(mask, dtype=np.uint8, order="F"))
        mask_rle["counts"] = mask_rle["counts"].decode()
        
        return {
            "object_id": object_id,
            "mask": {
                "size": mask_rle["size"],
                "counts": mask_rle["counts"],
            },
        }
    
    def _ensure_bucket_exists(self):
        """Ensure MinIO bucket exists for state storage"""
        if not self.minio_client.bucket_exists(self.state_bucket):
            self.minio_client.make_bucket(self.state_bucket)

class ThermalSegmentationAPI(BaseThermalSegmentationAPI):
    """
    Standard SAM 2 predictor service adapted for thermal building analysis.
    """

    def _initialize_sam2_model(self):
        """Initialize SAM 2 model with user-selected model size"""
        model_size = self.model_size or os.getenv("MODEL_SIZE", "base_plus")
        
        logger.info(f"🚀 Initializing Standard SAM 2 with model size: {model_size}")
        
        # Model configuration paths
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

        # Device selection
        force_cpu_device = os.environ.get("SAM2_DEMO_FORCE_CPU_DEVICE", "0") == "1"
        
        if torch.cuda.is_available() and not force_cpu_device:
            device = torch.device("cuda")
        elif torch.backends.mps.is_available() and not force_cpu_device:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
        self.device = device
        
        logger.info(f"📱 Device: {device}")
        logger.info(f"📂 Checkpoint: {checkpoint.name}")

        # Device-specific optimizations
        if device.type == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        try:
            self.predictor = build_sam2_video_predictor(
                model_cfg, checkpoint, device=device
            )
            logger.info(f"✅ Standard SAM 2 {model_size.upper()} model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load SAM 2 {model_size} model: {e}")
            raise RuntimeError(f"Failed to load SAM 2 {model_size} model: {e}")

    def start_session(self, video_path: str, analysis_session_id: str) -> str:
        """Initialize a new SAM 2 session with memory optimization"""
        with self.autocast_context(), self.inference_lock:
            session_id = str(uuid.uuid4())
            
            # Get video info for optimization decisions
            video_info = self.get_video_info_safe(video_path)
            if video_info:
                logger.info(f"🎬 Standard SAM 2 session {session_id}:")
                logger.info(f"   📏 Resolution: {video_info.get('width')}x{video_info.get('height')}")
                logger.info(f"   📊 Frames: {video_info.get('frame_count', 0)}")
            
            # Clear GPU cache before initialization
            self.clear_gpu_cache()
            gc.collect()
            
            # Monitor memory
            if torch.cuda.is_available():
                free_before, total = torch.cuda.mem_get_info()
                logger.info(f"📊 GPU memory before init: {(total - free_before) / 1e9:.1f}GB used")
            
            # Memory-efficient settings
            is_4k = video_info.get('is_4k', False)
            frame_count = video_info.get('frame_count', 0)
            
            # Always use aggressive CPU offloading for large videos
            offload_video_to_cpu = True
            offload_state_to_cpu = is_4k or frame_count > 2000
            
            logger.info(f"🚀 Standard SAM 2 with memory optimization:")
            logger.info(f"   💾 Async loading: True")
            logger.info(f"   🔄 Video CPU offload: {offload_video_to_cpu}")
            logger.info(f"   📱 State CPU offload: {offload_state_to_cpu}")
            
            try:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    inference_state = self.predictor.init_state(
                        video_path=video_path,
                        offload_video_to_cpu=offload_video_to_cpu,
                        offload_state_to_cpu=offload_state_to_cpu,
                        async_loading_frames=True,
                    )
                    
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Monitor memory after initialization
                if torch.cuda.is_available():
                    free_after, total = torch.cuda.mem_get_info()
                    used_memory = (total - free_after) / 1e9
                    memory_diff = (free_before - free_after) / 1e9
                    logger.info(f"📊 GPU memory after init: {used_memory:.1f}GB used ({memory_diff:+.1f}GB change)")
                
                # Store session state
                self.session_states[session_id] = {
                    "canceled": False,
                    "state": inference_state,
                    "analysis_session_id": analysis_session_id,
                    "video_path": video_path,
                    "video_info": video_info,
                }
                
                logger.info(f"✅ Standard SAM 2 session {session_id} initialized")
                return session_id
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"💥 OOM with standard mode - trying emergency CPU fallback")
                    
                    try:
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        inference_state = self.predictor.init_state(
                            video_path=video_path,
                            offload_video_to_cpu=True,
                            offload_state_to_cpu=True,
                            async_loading_frames=True,
                        )
                        
                        logger.info(f"✅ Recovered with emergency CPU fallback")
                        
                        self.session_states[session_id] = {
                            "canceled": False,
                            "state": inference_state,
                            "analysis_session_id": analysis_session_id,
                            "video_path": video_path,
                            "video_info": video_info,
                            "emergency_cpu": True
                        }
                        
                        return session_id
                        
                    except Exception as fallback_error:
                        logger.error(f"❌ Emergency CPU fallback failed: {fallback_error}")
                        raise
                else:
                    logger.error(f"❌ Non-OOM error: {e}")
                    raise
                    
            except Exception as e:
                logger.error(f"❌ Unexpected error: {e}")
                self.clear_gpu_cache()
                gc.collect()
                raise

    def add_points(
        self,
        session_id: str,
        frame_index: int,
        object_id: int,
        points: List[List[float]],
        labels: List[int],
        clear_old_points: bool = False,
    ) -> Dict[str, Any]:
        """Add points to a frame and get immediate segmentation result."""
        with self.autocast_context(), self.inference_lock:
            session = self._get_session(session_id)
            inference_state = session["state"]

            # Memory cleanup before processing
            self.cleanup_old_frame_outputs(inference_state)

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
            
            # Memory cleanup after processing
            if frame_index % 10 == 0:
                self.cleanup_old_frame_outputs(inference_state)
                
            if frame_index % 50 == 0:
                self.clear_gpu_cache()
            
            return {
                "frame_index": frame_idx,
                "results": rle_mask_list,
            }

    def close_session(self, session_id: str) -> bool:
        """Close a SAM 2 session and cleanup resources"""
        session = self.session_states.pop(session_id, None)
        if session is None:
            logger.warning(f"Session {session_id} not found for closure")
            return False
        
        # Final memory cleanup
        if "state" in session:
            self.cleanup_old_frame_outputs(session["state"])
        self.clear_gpu_cache()
        
        logger.info(f"🔒 Closed standard SAM 2 session {session_id}")
        return True

class HybridThermalSegmentationAPI(BaseThermalSegmentationAPI):
    """
    Hybrid SAM 2 predictor with GPU/CPU processing
    - GPU: SAM 2 model inference (fast)
    - CPU: Video frame storage (memory efficient)
    """

    def __init__(self, minio_client: Optional[Minio] = None, model_size: str = 'tiny', 
                 force_cpu_frames: bool = True, chunk_size: int = 50):
        self.force_cpu_frames = force_cpu_frames
        self.chunk_size = chunk_size
        super().__init__(minio_client, model_size)
        
        logger.info(f"🎯 Hybrid SAM 2 initialized:")
        logger.info(f"   🧠 Model: {model_size} on GPU")
        logger.info(f"   📹 Frames: {'CPU' if force_cpu_frames else 'GPU'}")
        logger.info(f"   📦 Chunk size: {chunk_size} frames")

    def _initialize_sam2_model(self):
        """Initialize SAM 2 model with hybrid configuration"""
        model_size = self.model_size or os.getenv("MODEL_SIZE", "tiny")
        
        # Model configuration paths
        model_configs = {
            "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
            "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
            "base_plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
            "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
        }
        
        if model_size not in model_configs:
            model_size = "tiny"
            logger.warning(f"Unknown model size, defaulting to {model_size}")
        
        model_cfg, checkpoint_file = model_configs[model_size]
        checkpoint = Path(f"/opt/sam2_worker/sam2/checkpoints/{checkpoint_file}")

        # Keep model on GPU for speed
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        self.device = device
        logger.info(f"🚀 Loading Hybrid SAM 2 {model_size.upper()} on {device}")

        # Device-specific optimizations
        if device.type == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        try:
            # Apply memory optimization overrides for hybrid mode
            # hydra_overrides = [
            #     '++model.max_cond_frames_in_attn=3',  # Reduce conditioning frames
            #     '++model.num_maskmem=5'  # Reduce mask memory slots
            # ]
            
            self.predictor = build_sam2_video_predictor(
                model_cfg, 
                checkpoint, 
                device=device,
                # hydra_overrides_extra=hydra_overrides
            )
            
            logger.info(f"✅ Hybrid SAM 2 {model_size.upper()} loaded with optimizations")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Hybrid SAM 2 {model_size} model: {e}")
            raise RuntimeError(f"Failed to load Hybrid SAM 2 {model_size} model: {e}")

    def start_session(self, video_path: str, analysis_session_id: str) -> str:
        """Initialize SAM 2 session with aggressive CPU offloading"""
        with self.autocast_context(), self.inference_lock:
            session_id = str(uuid.uuid4())
            
            # Get video info
            video_info = self.get_video_info_safe(video_path)
            if video_info:
                logger.info(f"🎬 Hybrid processing for session {session_id}:")
                logger.info(f"   📏 Resolution: {video_info.get('width')}x{video_info.get('height')}")
                logger.info(f"   📊 Frames: {video_info.get('frame_count', 0)}")
            
            # Clear GPU cache
            self.clear_gpu_cache()
            gc.collect()
            
            # Monitor memory
            if torch.cuda.is_available():
                free_before, total = torch.cuda.mem_get_info()
                logger.info(f"📊 GPU memory before init: {(total - free_before) / 1e9:.1f}GB used")
            
            # HYBRID: Maximum CPU offloading
            offload_video_to_cpu = True
            offload_state_to_cpu = True
            
            logger.info(f"🎯 Hybrid SAM 2 initialization:")
            logger.info(f"   💾 Async frame loading: True")
            logger.info(f"   🔄 Video frames → CPU: {offload_video_to_cpu}")
            logger.info(f"   📱 Inference state → CPU: {offload_state_to_cpu}")
            
            try:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    inference_state = self.predictor.init_state(
                        video_path=video_path,
                        offload_video_to_cpu=offload_video_to_cpu,
                        offload_state_to_cpu=offload_state_to_cpu,
                        async_loading_frames=True,
                    )
                    
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Monitor memory after initialization
                if torch.cuda.is_available():
                    free_after, total = torch.cuda.mem_get_info()
                    used_memory = (total - free_after) / 1e9
                    memory_diff = (free_before - free_after) / 1e9
                    logger.info(f"📊 GPU memory after init: {used_memory:.1f}GB used ({memory_diff:+.1f}GB change)")
                
                # Store session state
                self.session_states[session_id] = {
                    "canceled": False,
                    "state": inference_state,
                    "analysis_session_id": analysis_session_id,
                    "video_path": video_path,
                    "video_info": video_info,
                    "hybrid_mode": True,
                    "cpu_offload": offload_video_to_cpu
                }
                
                logger.info(f"✅ Hybrid SAM 2 session {session_id} initialized successfully")
                return session_id
                
            except Exception as e:
                logger.error(f"❌ Hybrid initialization failed: {e}")
                self.clear_gpu_cache()
                gc.collect()
                raise

    def add_points(
        self,
        session_id: str,
        frame_index: int,
        object_id: int,
        points: List[List[float]],
        labels: List[int],
        clear_old_points: bool = False,
    ) -> Dict[str, Any]:
        """Add points with hybrid processing"""
        with self.autocast_context(), self.inference_lock:
            session = self._get_session(session_id)
            inference_state = session["state"]

            # Aggressive memory cleanup for hybrid mode
            self.cleanup_old_frame_outputs(inference_state, max_frames=3)

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
            
            # Frequent memory cleanup in hybrid mode
            if frame_index % 5 == 0:
                self.cleanup_old_frame_outputs(inference_state, max_frames=3)
                
            if frame_index % 20 == 0:
                self.clear_gpu_cache()
            
            return {
                "frame_index": frame_idx,
                "results": rle_mask_list,
                "hybrid_mode": session.get("hybrid_mode", False)
            }

    def close_session(self, session_id: str) -> bool:
        """Close session with hybrid cleanup"""
        session = self.session_states.pop(session_id, None)
        if session is None:
            logger.warning(f"Session {session_id} not found")
            return False
        
        if "state" in session:
            self.cleanup_old_frame_outputs(session["state"], max_frames=0)
        
        self.clear_gpu_cache()
        logger.info(f"🔒 Closed hybrid session {session_id}")
        return True

# For backward compatibility, keep the original global functions
def get_video_info_safe(video_path: str) -> Dict[str, Any]:
    """Legacy function for backward compatibility"""
    api = BaseThermalSegmentationAPI(model_size='tiny')
    return api.get_video_info_safe(video_path)

def clear_gpu_cache():
    """Legacy function for backward compatibility"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def cleanup_old_frame_outputs(inference_state: Dict[str, Any], max_frames: int = 5):
    """Legacy function for backward compatibility"""
    api = BaseThermalSegmentationAPI(model_size='tiny')
    api.cleanup_old_frame_outputs(inference_state, max_frames)