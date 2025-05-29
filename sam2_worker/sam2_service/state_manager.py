"""
Memory Management Utilities for SAM 2 Video Processing
Handles large 4K+ videos by implementing memory-efficient processing strategies
"""
# sam2_worker/sam2_service/state_manager.py
import gc
import logging
from typing import Dict, Any, Optional
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages GPU memory for SAM 2 video processing to prevent OOM errors.
    Based on solutions from SAM2 GitHub issues #264 and #545.
    """
    
    def __init__(self, max_memory_frames: int = 5):
        """
        Initialize memory manager.
        
        Args:
            max_memory_frames: Maximum number of frame outputs to keep in memory
        """
        self.max_memory_frames = max_memory_frames
        
    def init_state_memory_efficient(
        self,
        predictor: SAM2VideoPredictor,
        video_path: str,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
    ) -> Dict[str, Any]:
        """
        Initialize SAM 2 inference state with memory-efficient settings.
        
        This is the key fix: using async_loading_frames=True prevents loading
        all video frames into memory at once.
        
        Args:
            predictor: SAM2VideoPredictor instance
            video_path: Path to video file
            offload_video_to_cpu: Whether to offload video frames to CPU
            offload_state_to_cpu: Whether to offload inference state to CPU
            
        Returns:
            inference_state: SAM 2 inference state dictionary
        """
        # Clear GPU cache before initialization
        self.clear_gpu_cache()
        
        # The critical fix: async_loading_frames=True
        # This prevents loading all frames into memory at initialization
        inference_state = predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=True,  # KEY FIX: Load frames on-demand
        )
        
        # Add memory management metadata
        inference_state["_memory_manager"] = {
            "max_memory_frames": self.max_memory_frames,
            "frame_cleanup_enabled": True,
        }
        
        logger.info(f"🚀 Initialized memory-efficient SAM 2 state")
        logger.info(f"   📊 Max memory frames: {self.max_memory_frames}")
        logger.info(f"   💾 Async loading: enabled")
        logger.info(f"   🔄 GPU offload: {offload_video_to_cpu}")
        
        return inference_state
    
    def cleanup_old_frame_outputs(self, inference_state: Dict[str, Any]) -> None:
        """
        Clean up old frame outputs to prevent memory accumulation.
        
        Based on GitHub issue #196 and #264 solutions.
        This removes old non-conditioning frame outputs to prevent memory buildup.
        
        Args:
            inference_state: SAM 2 inference state
        """
        if not inference_state.get("_memory_manager", {}).get("frame_cleanup_enabled", False):
            return
            
        max_frames = self.max_memory_frames
        
        # Clean up per-object frame outputs
        for obj_idx, obj_output_dict in inference_state.get("output_dict_per_obj", {}).items():
            # Clean non-conditioning frame outputs (these build up over time)
            non_cond_outputs = obj_output_dict.get("non_cond_frame_outputs", {})
            
            if len(non_cond_outputs) > max_frames:
                # Keep only the most recent frames
                frame_indices = sorted(non_cond_outputs.keys())
                frames_to_remove = frame_indices[:-max_frames]
                
                for frame_idx in frames_to_remove:
                    del non_cond_outputs[frame_idx]
                
                logger.debug(f"🧹 Cleaned {len(frames_to_remove)} old frame outputs for object {obj_idx}")
            
            # Optional: Also limit conditioning frame outputs if they get too large
            cond_outputs = obj_output_dict.get("cond_frame_outputs", {})
            if len(cond_outputs) > max_frames * 2:  # Allow more cond frames as they're user inputs
                frame_indices = sorted(cond_outputs.keys())
                frames_to_remove = frame_indices[:-max_frames * 2]
                
                for frame_idx in frames_to_remove:
                    del cond_outputs[frame_idx]
                
                logger.debug(f"🧹 Cleaned {len(frames_to_remove)} old conditioning frames for object {obj_idx}")
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU memory cache to free up VRAM"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get memory info
            free_memory, total_memory = torch.cuda.mem_get_info()
            used_memory = total_memory - free_memory
            
            logger.debug(f"🔧 GPU Memory: {used_memory / 1e9:.1f}GB used / {total_memory / 1e9:.1f}GB total")
    
    def monitor_memory_usage(self, context: str = "") -> Dict[str, float]:
        """
        Monitor current memory usage for debugging.
        
        Args:
            context: Description of current operation
            
        Returns:
            Dictionary with memory statistics
        """
        memory_info = {}
        
        if torch.cuda.is_available():
            free_memory, total_memory = torch.cuda.mem_get_info()
            used_memory = total_memory - free_memory
            
            memory_info = {
                "gpu_used_gb": used_memory / 1e9,
                "gpu_total_gb": total_memory / 1e9,
                "gpu_free_gb": free_memory / 1e9,
                "gpu_usage_percent": (used_memory / total_memory) * 100,
            }
            
            if context:
                logger.info(f"📊 Memory usage ({context}): {memory_info['gpu_used_gb']:.1f}GB / {memory_info['gpu_total_gb']:.1f}GB ({memory_info['gpu_usage_percent']:.1f}%)")
        
        return memory_info
    
    def memory_cleanup_hook(self, inference_state: Dict[str, Any], frame_idx: int) -> None:
        """
        Hook to call after processing each frame for memory management.
        
        Args:
            inference_state: SAM 2 inference state
            frame_idx: Current frame index
        """
        # Clean up old frames every 10 frames to avoid performance impact
        if frame_idx % 10 == 0:
            self.cleanup_old_frame_outputs(inference_state)
            
        # Clear GPU cache every 50 frames
        if frame_idx % 50 == 0:
            self.clear_gpu_cache()

# Global memory manager instance
memory_manager = MemoryManager()

def get_video_info_safe(video_path: str) -> Dict[str, Any]:
    """
    Get video information safely without loading the entire video.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video metadata
    """
    try:
        import subprocess
        import json
        
        # Use ffprobe to get video info (same approach as your current code)
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