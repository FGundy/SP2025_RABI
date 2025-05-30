# ui_components/segment-anything-ui/segment_anything_ui/thermal_api_adapter.py

import requests
import numpy as np
import torch
import cv2
import tempfile
import os
import json
from typing import Optional, Dict, Any, Tuple
import logging

from segment_anything_ui.config import THERMAL_API_BASE_URL, USE_THERMAL_BACKEND

logger = logging.getLogger(__name__)

class ThermalAPIAdapter:
    """
    Adapter to connect PyQt UI to your Django thermal analysis API.
    Mimics the SAM predictor interface but calls your API instead.
    """
    
    def __init__(self, base_url: str = THERMAL_API_BASE_URL):
        self.base_url = base_url.rstrip('/') + '/'
        self.session = requests.Session()
        self.interactive_session_id = None
        self.current_frame = 0
        self.analysis_session_id = None
        self.is_authenticated = False
        
        # Login automatically (you might want to add a login dialog)
        self.auto_login()
        
    def auto_login(self):
        """Auto-login for development. In production, you'd want a proper login dialog."""
        try:
            # Try to get CSRF token first
            csrf_response = self.session.get(f"{self.base_url}auth/csrf/")
            if csrf_response.status_code == 200:
                csrf_token = csrf_response.json().get('csrf_token')
                self.session.headers.update({'X-CSRFToken': csrf_token})
            
            # For development, you might want to create a test user or use session auth
            # This is a placeholder - implement proper authentication
            logger.info("🔐 Using session-based authentication")
            self.is_authenticated = True
            
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            self.is_authenticated = False
    
    def set_image(self, image: np.ndarray) -> bool:
        """
        Initialize thermal analysis session with an image.
        For video workflows, this creates an interactive session.
        """
        if not self.is_authenticated:
            logger.error("❌ Not authenticated with thermal API")
            return False
            
        try:
            logger.info("🚀 Starting thermal analysis session...")
            
            # Save image temporarily
            temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(temp_image.name, image)
            temp_image.close()
            
            # For single image analysis, we need to create an analysis session first
            # In your UI, you might want to let users select existing analysis sessions
            # For now, we'll create a minimal session
            
            # Create analysis session (simplified for demo)
            analysis_data = {
                'sam2_model': 'tiny',  # Matches your optimal config
                'description': 'PyQt UI Session',
                'building_project': None  # You might want to add building selection
            }
            
            response = self.session.post(
                f"{self.base_url}analysis/sessions/",
                json=analysis_data
            )
            
            if response.status_code == 201:
                session_data = response.json()
                self.analysis_session_id = session_data['id']
                logger.info(f"✅ Analysis session created: {self.analysis_session_id}")
                
                # Start interactive session
                interactive_response = self.session.post(
                    f"{self.base_url}analysis/interactive/start/",
                    json={'analysis_session_id': self.analysis_session_id}
                )
                
                if interactive_response.status_code == 201:
                    interactive_data = interactive_response.json()
                    self.interactive_session_id = interactive_data['interactive_session_id']
                    logger.info(f"✅ Interactive session started: {self.interactive_session_id}")
                    return True
                    
            logger.error(f"❌ Failed to create session: {response.status_code}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error setting image: {e}")
            return False
        finally:
            # Cleanup temp file
            if 'temp_image' in locals():
                try:
                    os.unlink(temp_image.name)
                except:
                    pass
    
    def predict(
        self, 
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None, 
        box: Optional[np.ndarray] = None,
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make segmentation prediction - mimics SAM predictor interface.
        Calls your Django API instead of direct SAM inference.
        """
        if not self.interactive_session_id:
            logger.error("❌ No interactive session - call set_image() first")
            return None, None, None
            
        try:
            logger.info("🎯 Making segmentation prediction...")
            
            # Enter segmentation mode if not already active
            seg_mode_response = self.session.post(
                f"{self.base_url}analysis/interactive/{self.interactive_session_id}/segmentation/enter/",
                json={'model_size': 'tiny'}
            )
            
            if seg_mode_response.status_code != 200:
                logger.warning(f"⚠️ Segmentation mode response: {seg_mode_response.status_code}")
            
            # Prepare prompt data
            prompt_data = {
                'prompt_type': 'points',
                'coordinates': point_coords.tolist() if point_coords is not None else [],
                'object_id': 1,  # Single object for now
            }
            
            # Add point labels if provided
            if point_labels is not None:
                prompt_data['labels'] = point_labels.tolist()
            
            # Add bounding box if provided
            if box is not None:
                prompt_data['prompt_type'] = 'box'
                prompt_data['coordinates'] = box.tolist()
            
            # Send segmentation prompt
            response = self.session.post(
                f"{self.base_url}analysis/interactive/{self.interactive_session_id}/prompt/",
                json=prompt_data
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract mask data from response
                if 'masks' in result:
                    masks_data = result['masks']
                    
                    # Convert masks back to numpy arrays
                    masks = []
                    scores = []
                    logits = []
                    
                    for mask_info in masks_data:
                        # Your API should return mask as base64 or coordinates
                        # This is a placeholder - adapt based on your actual API response
                        mask = np.array(mask_info.get('mask', []))
                        if mask.size > 0:
                            masks.append(mask)
                            scores.append(mask_info.get('score', 0.9))
                            logits.append(mask_info.get('logits', mask))
                    
                    if masks:
                        masks = np.array(masks)
                        scores = np.array(scores)
                        logits = np.array(logits)
                        
                        logger.info(f"✅ Prediction successful: {masks.shape}")
                        return masks, scores, logits
                
                logger.warning("⚠️ No masks in API response")
                
            else:
                logger.error(f"❌ Prediction failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
        
        # Return empty results if prediction fails
        return np.array([]), np.array([]), np.array([])
    
    def close_session(self):
        """Clean up the thermal analysis session."""
        if self.interactive_session_id:
            try:
                response = self.session.delete(
                    f"{self.base_url}analysis/interactive/{self.interactive_session_id}/"
                )
                logger.info(f"🧹 Session closed: {response.status_code}")
            except Exception as e:
                logger.error(f"❌ Error closing session: {e}")
            finally:
                self.interactive_session_id = None
                self.analysis_session_id = None


class ThermalSAMPredictor:
    """
    Drop-in replacement for SAM predictor that uses thermal API.
    Maintains exact same interface as SAM predictors.
    """
    
    def __init__(self, thermal_api: ThermalAPIAdapter):
        self.thermal_api = thermal_api
        self.is_image_set = False
    
    def set_image(self, image: np.ndarray):
        """Set the image for analysis - required before prediction."""
        self.is_image_set = self.thermal_api.set_image(image)
        if not self.is_image_set:
            logger.error("❌ Failed to set image in thermal API")
    
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make prediction using thermal API."""
        if not self.is_image_set:
            logger.error("❌ Image not set - call set_image() first")
            return np.array([]), np.array([]), np.array([])
        
        return self.thermal_api.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'thermal_api'):
            self.thermal_api.close_session()


def create_thermal_predictor() -> ThermalSAMPredictor:
    """Factory function to create thermal predictor."""
    thermal_api = ThermalAPIAdapter()
    return ThermalSAMPredictor(thermal_api)


# Mock SAM model class for thermal backend
class ThermalSAMModel:
    """Mock SAM model that represents thermal API backend."""
    
    def __init__(self):
        self.device = "api"  # Special device identifier
        self.model_type = "thermal_api"
    
    def to(self, device):
        """No-op for device movement since this is API-based."""
        return self
    
    def eval(self):
        """No-op for eval mode since this is API-based."""
        return self