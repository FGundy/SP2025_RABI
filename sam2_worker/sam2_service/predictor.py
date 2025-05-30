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

import decord
from collections import OrderedDict, deque
from threading import Thread
from sam2.sam2_video_predictor import SAM2VideoPredictor
from .config import (
    USE_LAZY_LOADING, 
    LAZY_LOADER_LRU_CACHE_SIZE, 
    LAZY_LOADER_SEQUENTIAL_PREFETCH,
    LAZY_LOADER_MAX_CACHE_MEMORY_GB
)

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



class DecordLazyFrameLoader:
    """
    Memory-efficient video frame loader that only keeps a sliding window of frames in memory.
    Replaces SAM2's default frame loading for 4GB VRAM constraint.
    """
    
    def __init__(
        self,
        video_path: str,
        image_size: int,
        offload_to_cpu: bool = True,
        img_mean: tuple = (0.485, 0.456, 0.406),
        img_std: tuple = (0.229, 0.224, 0.225),
        compute_device: torch.device = None,
        lru_cache_size: int = LAZY_LOADER_LRU_CACHE_SIZE,
        sequential_prefetch: int = LAZY_LOADER_SEQUENTIAL_PREFETCH,
        max_cache_memory_gb: float = LAZY_LOADER_MAX_CACHE_MEMORY_GB
    ):
        self.video_path = video_path
        self.image_size = image_size
        self.offload_to_cpu = offload_to_cpu
        self.compute_device = compute_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lru_cache_size = lru_cache_size
        self.sequential_prefetch = sequential_prefetch
        self.max_cache_memory_gb = max_cache_memory_gb
        
        # Normalization parameters
        self.img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        
        # Initialize decord video reader with safety checks
        try:
            decord.bridge.set_bridge("torch")
            self.video_reader = decord.VideoReader(video_path)
            
            # Get video metadata safely
            self.num_frames = len(self.video_reader)
            
            # Get first frame to determine dimensions
            test_frame = self.video_reader[0]
            if hasattr(test_frame, 'shape'):
                self.video_height = test_frame.shape[0]
                self.video_width = test_frame.shape[1]
            else:
                # Fallback for different frame types
                frame_array = test_frame.asnumpy() if hasattr(test_frame, 'asnumpy') else test_frame
                self.video_height = frame_array.shape[0] 
                self.video_width = frame_array.shape[1]
                
        except Exception as e:
            logger.error(f"Failed to initialize video reader for {video_path}: {e}")
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        # Video metadata
        # self.num_frames = len(self.video_reader)
        # self.video_height = self.video_reader[0].shape[0]
        # self.video_width = self.video_reader[0].shape[1]
        
        # Cache structures
        self.lru_cache = OrderedDict()  # frame_idx -> tensor
        self.lru_order = deque()        # Track access order
        self.prefetch_queue = set()     # Frames being prefetched
        self.cache_lock = Lock()
        
        # Device placement for cache
        if self.offload_to_cpu:
            self.cache_device = torch.device("cpu")
            self.img_mean = self.img_mean.to("cpu")
            self.img_std = self.img_std.to("cpu")
        else:
            self.cache_device = self.compute_device
            self.img_mean = self.img_mean.to(self.compute_device)
            self.img_std = self.img_std.to(self.compute_device)
        
        logger.info(f"🎬 Lazy frame loader initialized:")
        logger.info(f"   📏 Video: {self.video_width}x{self.video_height}, {self.num_frames} frames")
        logger.info(f"   💾 Cache: {self.lru_cache_size} frames LRU + {self.sequential_prefetch} prefetch")
        logger.info(f"   📱 Device: {'CPU' if self.offload_to_cpu else self.compute_device}")
        logger.info(f"   🔧 Max cache memory: {self.max_cache_memory_gb}GB")
    
    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, frame_idx: int) -> torch.Tensor:
        """Get frame tensor, loading on-demand with caching and prefetch"""
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.num_frames-1}]")
        
        with self.cache_lock:
            # Check if frame is already in cache
            if frame_idx in self.lru_cache:
                # Move to end of LRU order (most recently used)
                self.lru_order.remove(frame_idx)
                self.lru_order.append(frame_idx)
                frame_tensor = self.lru_cache[frame_idx]
                
                # Move to target device if needed
                if not self.offload_to_cpu and frame_tensor.device != self.compute_device:
                    frame_tensor = frame_tensor.to(self.compute_device, non_blocking=True)
                    self.lru_cache[frame_idx] = frame_tensor
                
                # Trigger prefetch for sequential access
                self._trigger_sequential_prefetch(frame_idx)
                
                return frame_tensor
            
            # Cache miss - load frame
            frame_tensor = self._load_frame(frame_idx)
            
            # Add to cache with LRU eviction
            self._add_to_cache(frame_idx, frame_tensor)
            
            # Trigger prefetch for sequential access
            self._trigger_sequential_prefetch(frame_idx)
            
            return frame_tensor
    
    def _load_frame(self, frame_idx: int) -> torch.Tensor:
        """Load and process a single frame from video"""
        try:
            # Load frame using decord
            frame = self.video_reader[frame_idx]  # Shape: (H, W, 3)
            
            # Convert NDArray to PyTorch tensor safely
            if hasattr(frame, 'asnumpy'):
                # It's a decord NDArray - convert to numpy then to tensor
                frame_np = frame.asnumpy()
                frame = torch.from_numpy(frame_np.copy())  # Copy to avoid memory issues
            elif not isinstance(frame, torch.Tensor):
                # It's some other array type - convert to tensor
                frame = torch.tensor(frame)
            
            # Ensure it's the right data type and normalize to [0, 1]
            frame = frame.float() / 255.0
            
            # Resize to model input size
            if frame.shape[0] != self.image_size or frame.shape[1] != self.image_size:
                frame = torch.nn.functional.interpolate(
                    frame.permute(2, 0, 1).unsqueeze(0),  # (1, 3, H, W)
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True
                ).squeeze(0)  # (3, H, W)
            else:
                frame = frame.permute(2, 0, 1)  # (3, H, W)
            
            # Normalize by mean and std
            frame -= self.img_mean
            frame /= self.img_std
            
            # Move to appropriate device
            frame = frame.to(self.cache_device, non_blocking=True)
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to load frame {frame_idx}: {e}")
            # Return placeholder frame (black tensor with correct dimensions)
            placeholder = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            placeholder = placeholder.to(self.cache_device)
            return placeholder
    
    def _add_to_cache(self, frame_idx: int, frame_tensor: torch.Tensor):
        """Add frame to LRU cache with memory limit enforcement"""
        # Check if we need to evict frames to stay under memory limit
        while len(self.lru_cache) >= self.lru_cache_size or self._estimate_cache_memory_gb() > self.max_cache_memory_gb:
            if not self.lru_order:
                break
            
            # Evict least recently used frame
            lru_frame_idx = self.lru_order.popleft()
            if lru_frame_idx in self.lru_cache:
                del self.lru_cache[lru_frame_idx]
                logger.debug(f"🧹 Evicted frame {lru_frame_idx} from cache")
        
        # Add new frame
        self.lru_cache[frame_idx] = frame_tensor
        self.lru_order.append(frame_idx)
        
        logger.debug(f"💾 Cached frame {frame_idx} ({len(self.lru_cache)}/{self.lru_cache_size} frames)")
    
    def _trigger_sequential_prefetch(self, frame_idx: int):
        """Trigger background prefetch of sequential frames"""
        for offset in range(1, self.sequential_prefetch + 1):
            next_frame_idx = frame_idx + offset
            
            if (next_frame_idx < self.num_frames and 
                next_frame_idx not in self.lru_cache and 
                next_frame_idx not in self.prefetch_queue):
                
                # Add to prefetch queue and start background loading
                self.prefetch_queue.add(next_frame_idx)
                
                def prefetch_frame(idx):
                    try:
                        frame_tensor = self._load_frame(idx)
                        with self.cache_lock:
                            if idx not in self.lru_cache:  # Still not loaded by main thread
                                self._add_to_cache(idx, frame_tensor)
                    except Exception as e:
                        logger.debug(f"Prefetch failed for frame {idx}: {e}")
                    finally:
                        self.prefetch_queue.discard(idx)
                
                # Start prefetch in background thread
                Thread(target=prefetch_frame, args=(next_frame_idx,), daemon=True).start()
                logger.debug(f"🔄 Prefetching frame {next_frame_idx}")
    
    def _estimate_cache_memory_gb(self) -> float:
        """Estimate current cache memory usage in GB"""
        if not self.lru_cache:
            return 0.0
        
        # Estimate memory per frame: 3 channels * image_size^2 * 4 bytes (float32)
        bytes_per_frame = 3 * self.image_size * self.image_size * 4
        total_bytes = len(self.lru_cache) * bytes_per_frame
        return total_bytes / 1e9
    
    def get_cache_size_in_bytes(self) -> int:
        """Get current cache size in bytes for monitoring"""
        bytes_per_frame = 3 * self.image_size * self.image_size * 4
        return len(self.lru_cache) * bytes_per_frame
    
    def clear_cache(self):
        """Clear all cached frames"""
        with self.cache_lock:
            self.lru_cache.clear()
            self.lru_order.clear()
            self.prefetch_queue.clear()
            logger.info("🧹 Cleared lazy loader cache")


class CustomHybridSAM2VideoPredictor(SAM2VideoPredictor):
    """
    Custom SAM2 Video Predictor that uses lazy frame loading to prevent OOM errors.
    Replaces the standard init_state() with memory-efficient frame loading.
    """
    
    def __init__(self, **kwargs):
        # Pass all arguments to the parent class using keywords.
        # This is compatible with SAM2's hydra build system.
        super().__init__(**kwargs)
        logger.info("🎯 Custom Hybrid SAM2 Video Predictor initialized")
    
    @torch.inference_mode()
    def init_state(
        self,
        video_path,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize inference state with lazy frame loading"""
        compute_device = self.device  # device of the model
        
        # Create lazy frame loader instead of loading all frames
        lazy_loader = DecordLazyFrameLoader(
            video_path=video_path,
            image_size=self.image_size,
            offload_to_cpu=offload_video_to_cpu,
            compute_device=compute_device
        )
        
        # Initialize inference state (similar to original init_state)
        inference_state = {}
        inference_state["images"] = lazy_loader  # Use lazy loader instead of full tensor
        inference_state["num_frames"] = len(lazy_loader)
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["video_height"] = lazy_loader.video_height
        inference_state["video_width"] = lazy_loader.video_width
        inference_state["device"] = compute_device
        
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        
        # Initialize state structures (same as original)
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["cached_features"] = {}
        inference_state["constants"] = {}
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["frames_tracked_per_obj"] = {}
        
        # Warm up the visual backbone and cache the image feature on frame 0
        # This ensures the lazy loader works correctly with SAM2's expectations
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        
        logger.info(f"✅ Custom SAM2 init_state completed with lazy loading")
        logger.info(f"   📊 Video: {inference_state['num_frames']} frames")
        logger.info(f"   💾 Lazy cache: {lazy_loader.lru_cache_size} frames max")
        
        return inference_state


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

    def propagate_in_video(self, session_id: str, **kwargs):
        """Expose SAM2's native propagate_in_video method directly"""
        session = self._get_session(session_id)
        inference_state = session["state"]
        
        # Just call the original method - it's already perfect!
        return self.predictor.propagate_in_video(
            inference_state=inference_state, 
            **kwargs
        )


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
        """Initialize SAM 2 model with optional lazy loading configuration"""
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
        
        model_cfg_path, checkpoint_file = model_configs[model_size]
        checkpoint_path = Path(f"/opt/sam2_worker/sam2/checkpoints/{checkpoint_file}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        use_lazy_loading = USE_LAZY_LOADING
        logger.info(f"🚀 Loading Hybrid SAM 2 {model_size.upper()} on {device}")
        logger.info(f"   🔧 Lazy loading: {'enabled' if use_lazy_loading else 'disabled'}")

        # Device-specific optimizations
        if device.type == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        try:
            hydra_overrides = []
            
            if use_lazy_loading:
                # Use hydra override to swap in our custom class
                # IMPORTANT: Full Python import path to our custom class
                hydra_overrides.append(
                    "++model._target_=sam2_service.predictor.CustomHybridSAM2VideoPredictor"
                )
                logger.info("   🔧 Overriding model target to use CustomHybridSAM2VideoPredictor")

            # Use the standard SAM2 builder with our override
            self.predictor = build_sam2_video_predictor(
                model_cfg_path,
                checkpoint_path,
                device=device,
                hydra_overrides_extra=hydra_overrides,
            )
            
            if use_lazy_loading:
                logger.info(f"✅ Hybrid SAM 2 {model_size.upper()} loaded with LAZY LOADING")
            else:
                logger.info(f"✅ Hybrid SAM 2 {model_size.upper()} loaded with STANDARD loading")
                
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

    def propagate_in_video(self, session_id: str, **kwargs):
        """Expose SAM2's native propagate_in_video method directly"""
        session = self._get_session(session_id)
        inference_state = session["state"]
        
        # Just call the original method - it's already perfect!
        return self.predictor.propagate_in_video(
            inference_state=inference_state, 
            **kwargs
        )


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