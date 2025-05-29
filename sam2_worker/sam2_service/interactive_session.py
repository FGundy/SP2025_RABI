# sam2_worker/sam2_service/interactive_session.py - FIXED VERSION
"""
Interactive Video Session Manager - Updated for Unified API
Handles video playback and on-demand SAM 2 segmentation activation
"""
import logging
import uuid
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import json

logger = logging.getLogger(__name__)

class SessionMode(Enum):
    """Video session modes"""
    VIEWING = "viewing"           # Just watching video, no SAM 2
    SEGMENTING = "segmenting"     # Active SAM 2 segmentation
    PAUSED = "paused"             # Segmentation paused, can resume

class PromptType(Enum):
    """Available prompt types for segmentation"""
    POSITIVE_POINT = "positive_point"
    NEGATIVE_POINT = "negative_point"
    BOUNDING_BOX = "bounding_box"
    CLEAR_MASKS = "clear_masks"

class InteractiveVideoSession:
    """
    Manages interactive video sessions with on-demand SAM 2 activation
    Compatible with both ThermalSegmentationAPI and HybridThermalSegmentationAPI
    """
    
    def __init__(self, thermal_api, django_session_id: str):
        """
        Initialize interactive session
        
        Args:
            thermal_api: ThermalSegmentationAPI or HybridThermalSegmentationAPI instance
            django_session_id: Django AnalysisSession ID
        """
        self.session_id = str(uuid.uuid4())
        self.thermal_api = thermal_api
        self.django_session_id = django_session_id
        
        # Session state
        self.mode = SessionMode.VIEWING
        self.video_path = None
        self.video_info = {}
        self.current_frame = 0
        self.playback_speed = 1.0
        self.is_playing = False
        
        # SAM 2 state (only active during segmentation)
        self.sam2_session_id = None
        self.active_chunk_info = None
        self.current_masks = {}  # frame_index -> masks
        self.segmentation_history = []  # For undo/redo
        
        # User preferences
        self.selected_model = "base_plus"
        self.confidence_threshold = 0.5
        
        logger.info(f"🎬 Created interactive session {self.session_id}")
    
    def load_video(self, video_path: str) -> Dict[str, Any]:
        """
        Load video for viewing (no SAM 2 initialization)
        
        Args:
            video_path: Path to video file
            
        Returns:
            Video information and session status
        """
        self.video_path = video_path
        self.video_info = self.thermal_api.get_video_info_safe(video_path)
        self.current_frame = 0
        self.mode = SessionMode.VIEWING
        
        logger.info(f"📽️  Loaded video: {self.video_info.get('width')}x{self.video_info.get('height')}, {self.video_info.get('frame_count')} frames")
        
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "video_info": self.video_info,
            "current_frame": self.current_frame,
            "playback_controls": {
                "can_play": True,
                "can_seek": True,
                "can_change_speed": True
            },
            "segmentation_available": True,
            "recommended_sam2_model": self._get_recommended_model()
        }
    
    def play_video(self, speed: float = 1.0) -> Dict[str, Any]:
        """Start video playback at specified speed"""
        if self.mode == SessionMode.SEGMENTING:
            return {"error": "Cannot play video during active segmentation. Pause segmentation first."}
        
        self.is_playing = True
        self.playback_speed = speed
        
        return {
            "session_id": self.session_id,
            "status": "playing",
            "speed": speed,
            "current_frame": self.current_frame
        }
    
    def pause_video(self) -> Dict[str, Any]:
        """Pause video playback"""
        self.is_playing = False
        
        return {
            "session_id": self.session_id,
            "status": "paused",
            "current_frame": self.current_frame
        }
    
    def seek_to_frame(self, frame_index: int) -> Dict[str, Any]:
        """
        Seek to specific frame
        
        Args:
            frame_index: Target frame number
        """
        max_frame = self.video_info.get('frame_count', 0) - 1
        self.current_frame = max(0, min(frame_index, max_frame))
        
        response = {
            "session_id": self.session_id,
            "current_frame": self.current_frame,
            "mode": self.mode.value
        }
        
        # Note: Chunking management removed for simplicity
        # The unified API handles memory management internally
        
        return response
    
    def enter_segmentation_mode(self, model_size: Optional[str] = None, 
                               confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Enter segmentation mode - THIS IS WHERE SAM 2 INITIALIZES
        
        Args:
            model_size: SAM 2 model to use (user choice)
            confidence_threshold: Segmentation confidence threshold
        """
        if self.mode == SessionMode.SEGMENTING:
            return {"message": "Already in segmentation mode", "session_id": self.session_id}
        
        # Update user preferences
        if model_size:
            self.selected_model = model_size
        if confidence_threshold:
            self.confidence_threshold = confidence_threshold
        
        # Pause video playback
        self.is_playing = False
        
        logger.info(f"🎯 Entering segmentation mode with {self.selected_model} model")
        logger.info(f"   📍 Current frame: {self.current_frame}")
        
        try:
            # Initialize SAM 2 - FIXED: Remove force_chunking parameter
            self._initialize_sam2_for_current_frame()
            
            self.mode = SessionMode.SEGMENTING
            
            return {
                "session_id": self.session_id,
                "sam2_session_id": self.sam2_session_id,
                "mode": self.mode.value,
                "current_frame": self.current_frame,
                "model_used": self.selected_model,
                "confidence_threshold": self.confidence_threshold,
                "active_chunk": self.active_chunk_info,
                "available_tools": [
                    "positive_point",
                    "negative_point", 
                    "bounding_box",
                    "clear_masks",
                    "save_masks",
                    "load_masks"
                ],
                "status": "ready_for_segmentation"
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to enter segmentation mode: {e}")
            return {
                "error": f"Failed to initialize segmentation: {str(e)}",
                "recommended_action": "Try a smaller model size or reduce video quality"
            }
    
    def exit_segmentation_mode(self, save_masks: bool = False) -> Dict[str, Any]:
        """
        Exit segmentation mode and return to viewing
        
        Args:
            save_masks: Whether to save current masks to database
        """
        if self.mode != SessionMode.SEGMENTING:
            return {"message": "Not in segmentation mode"}
        
        result = {"session_id": self.session_id, "masks_saved": False}
        
        # Save masks if requested
        if save_masks and self.current_masks:
            try:
                saved_count = self._save_masks_to_database()
                result["masks_saved"] = True
                result["saved_mask_count"] = saved_count
                logger.info(f"💾 Saved {saved_count} masks to database")
            except Exception as e:
                logger.error(f"Failed to save masks: {e}")
                result["save_error"] = str(e)
        
        # Clean up SAM 2 resources
        if self.sam2_session_id:
            try:
                self.thermal_api.close_session(self.sam2_session_id)
                logger.info(f"🔒 Closed SAM 2 session {self.sam2_session_id}")
            except Exception as e:
                logger.warning(f"Error closing SAM 2 session: {e}")
        
        # Reset segmentation state
        self.sam2_session_id = None
        self.active_chunk_info = None
        self.mode = SessionMode.VIEWING
        
        result.update({
            "mode": self.mode.value,
            "current_frame": self.current_frame,
            "playback_controls_restored": True
        })
        
        return result
    
    def add_prompt(self, prompt_type: str, coordinates: List[float], 
                   object_id: int = 1) -> Dict[str, Any]:
        """
        Add segmentation prompt (point, box, etc.)
        
        Args:
            prompt_type: Type of prompt (positive_point, negative_point, bounding_box)
            coordinates: [x, y] for points, [x1, y1, x2, y2] for boxes
            object_id: Object identifier
        """
        if self.mode != SessionMode.SEGMENTING:
            return {"error": "Must be in segmentation mode to add prompts"}
        
        if not self.sam2_session_id:
            return {"error": "SAM 2 session not initialized"}
        
        try:
            # Convert prompt type to SAM 2 format
            if prompt_type == "positive_point":
                points = [coordinates]
                labels = [1]  # Positive
            elif prompt_type == "negative_point":
                points = [coordinates]
                labels = [0]  # Negative
            elif prompt_type == "bounding_box":
                # Handle bounding box (would need different SAM 2 API call)
                return self._add_bounding_box(coordinates, object_id)
            else:
                return {"error": f"Unknown prompt type: {prompt_type}"}
            
            # Add points to SAM 2 - FIXED: Use current frame directly
            result = self.thermal_api.add_points(
                session_id=self.sam2_session_id,
                frame_index=self.current_frame,  # Use current frame directly
                object_id=object_id,
                points=points,
                labels=labels,
                clear_old_points=False
            )
            
            # Store masks for this frame
            self.current_masks[self.current_frame] = result["results"]
            
            # Add to history for undo/redo
            self.segmentation_history.append({
                "action": "add_prompt",
                "prompt_type": prompt_type,
                "coordinates": coordinates,
                "object_id": object_id,
                "frame": self.current_frame,
                "result": result
            })
            
            logger.info(f"✅ Added {prompt_type} at {coordinates} to frame {self.current_frame}")
            
            return {
                "session_id": self.session_id,
                "prompt_added": True,
                "prompt_type": prompt_type,
                "frame_index": self.current_frame,
                "masks": result["results"],
                "object_count": len(result["results"]),
                "can_undo": len(self.segmentation_history) > 0
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to add prompt: {e}")
            return {"error": f"Failed to add prompt: {str(e)}"}
    
    def clear_masks(self, frame_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Clear masks for current frame or specific frame
        
        Args:
            frame_index: Frame to clear (current frame if None)
        """
        if self.mode != SessionMode.SEGMENTING:
            return {"error": "Must be in segmentation mode to clear masks"}
        
        target_frame = frame_index or self.current_frame
        
        # Remove from current masks
        if target_frame in self.current_masks:
            del self.current_masks[target_frame]
        
        # Add to history
        self.segmentation_history.append({
            "action": "clear_masks",
            "frame": target_frame
        })
        
        return {
            "session_id": self.session_id,
            "masks_cleared": True,
            "frame_index": target_frame,
            "remaining_frames_with_masks": len(self.current_masks)
        }
    
    def save_masks_to_database(self) -> Dict[str, Any]:
        """Save current masks to Django database"""
        if not self.current_masks:
            return {"message": "No masks to save"}
        
        try:
            saved_count = self._save_masks_to_database()
            
            return {
                "session_id": self.session_id,
                "masks_saved": True,
                "saved_count": saved_count,
                "frames_saved": list(self.current_masks.keys())
            }
            
        except Exception as e:
            logger.error(f"Failed to save masks: {e}")
            return {"error": f"Failed to save masks: {str(e)}"}
    
    def load_masks_from_database(self) -> Dict[str, Any]:
        """Load previously saved masks from Django database"""
        try:
            # This would query your Django SegmentationResult model
            loaded_masks = self._load_masks_from_database()
            
            self.current_masks.update(loaded_masks)
            
            return {
                "session_id": self.session_id,
                "masks_loaded": True,
                "loaded_count": len(loaded_masks),
                "frames_with_masks": list(self.current_masks.keys())
            }
            
        except Exception as e:
            logger.error(f"Failed to load masks: {e}")
            return {"error": f"Failed to load masks: {str(e)}"}
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status"""
        return {
            "session_id": self.session_id,
            "mode": self.mode.value,
            "current_frame": self.current_frame,
            "is_playing": self.is_playing,
            "playback_speed": self.playback_speed,
            "video_info": self.video_info,
            "segmentation_active": self.sam2_session_id is not None,
            "masks_count": len(self.current_masks),
            "can_undo": len(self.segmentation_history) > 0,
            "selected_model": self.selected_model
        }
    
    # Private helper methods
    def _initialize_sam2_for_current_frame(self):
        """Initialize SAM 2 for the current frame location - FIXED"""
        # Create SAM 2 session - FIXED: Remove force_chunking parameter
        self.sam2_session_id = self.thermal_api.start_session(
            video_path=self.video_path,
            analysis_session_id=f"{self.django_session_id}_interactive_{self.session_id}"
        )
        
        # Store session info
        self.active_chunk_info = {
            "session_id": self.sam2_session_id,
            "current_frame": self.current_frame,
            "video_path": self.video_path
        }
        
        logger.info(f"🚀 SAM 2 initialized for frame {self.current_frame}")
    
    def _save_masks_to_database(self) -> int:
        """Save masks to Django SegmentationResult model"""
        # TODO: Implement Django model integration
        saved_count = 0
        
        for frame_index, masks in self.current_masks.items():
            for mask_data in masks:
                # Create SegmentationResult object
                # segmentation_result = SegmentationResult.objects.create(...)
                saved_count += 1
        
        return saved_count
    
    def _load_masks_from_database(self) -> Dict[int, List[Dict]]:
        """Load masks from Django SegmentationResult model"""
        # TODO: Implement Django model integration
        # return {frame_index: [mask_data, ...], ...}
        return {}
    
    def _get_recommended_model(self) -> str:
        """Get recommended SAM 2 model based on video characteristics"""
        frame_count = self.video_info.get('frame_count', 0)
        is_4k = self.video_info.get('is_4k', False)
        
        if is_4k and frame_count > 2000:
            return 'tiny'
        elif is_4k or frame_count > 3000:
            return 'small'
        else:
            return 'base_plus'
    
    def _add_bounding_box(self, coordinates: List[float], object_id: int) -> Dict[str, Any]:
        """Handle bounding box prompts"""
        # TODO: Implement bounding box support
        return {"error": "Bounding box prompts not yet implemented"}