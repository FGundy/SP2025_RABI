# backend/apps/analysis/models.py
# 
from django.contrib.auth.models import User
from django.contrib.gis.db import models
from django.contrib.gis.geos import Point, Polygon
from django.utils import timezone  # ADD THIS IMPORT
from apps.buildings.models import BuildingProject, VideoFile
import uuid


class AnalysisSession(models.Model):
    """Represents a video analysis session using SAM 2"""
    
    # Session statuses
    SESSION_STATUSES = [
        ('initializing', 'Initializing'),
        ('active', 'Active'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('error', 'Error'),
    ]
    
    # SAM 2 model choices
    SAM2_MODEL_CHOICES = [
        ('tiny', 'SAM 2 Tiny (38.9M params) - Fast, low memory'),
        ('small', 'SAM 2 Small (46.0M params) - Balanced'),
        ('base_plus', 'SAM 2 Base Plus (80.8M params) - High quality'),
        ('large', 'SAM 2 Large (224.4M params) - Best quality, high memory'),
    ]
    
    # Analysis type choices
    ANALYSIS_TYPE_CHOICES = [
        ('thermal_anomaly', 'Thermal Anomaly Detection'),
        ('material_mapping', 'Material Mapping'),
        ('insulation_analysis', 'Insulation Analysis'),
        ('air_leakage', 'Air Leakage Detection'),
        ('structural_analysis', 'Structural Analysis'),
    ]

    # Primary key and relationships
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    building_project = models.ForeignKey(
        BuildingProject, 
        on_delete=models.CASCADE, 
        related_name='analysis_sessions'
    )
    video_file = models.ForeignKey(
        VideoFile, 
        on_delete=models.CASCADE, 
        related_name='analysis_sessions'
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Session metadata
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    status = models.CharField(
        max_length=20, 
        choices=SESSION_STATUSES, 
        default='initializing'
    )
    
    # SAM 2 configuration
    sam2_model = models.CharField(
        max_length=20,
        choices=SAM2_MODEL_CHOICES,
        default='base_plus',
        help_text='SAM 2 model size affects quality vs memory usage'
    )
    
    # SAM 2 session info
    sam2_session_id = models.CharField(
        max_length=100, 
        blank=True,
        help_text='UUID from SAM 2 service'
    )
    
    # Analysis parameters
    analysis_type = models.CharField(
        max_length=50, 
        choices=ANALYSIS_TYPE_CHOICES,
        default='thermal_anomaly'
    )
    target_materials = models.JSONField(
        default=list,
        help_text='List of materials to focus on'
    )
    
    # Progress tracking
    total_frames_analyzed = models.IntegerField(default=0)
    total_objects_tracked = models.IntegerField(default=0)
    
    # Processing parameters
    confidence_threshold = models.FloatField(
        default=0.5,
        help_text='Minimum confidence threshold for detections'
    )
    enable_tracking = models.BooleanField(
        default=True,
        help_text='Enable object tracking across frames'
    )
    
    # Memory management settings
    offload_to_cpu = models.BooleanField(
        default=True,
        help_text='Offload video frames to CPU to save GPU memory'
    )
    max_memory_frames = models.IntegerField(
        default=5,
        help_text='Maximum number of frames to keep in GPU memory'
    )
    
    # Results and error handling
    error_message = models.TextField(
        blank=True,
        help_text='Error message if analysis fails'
    )
    results_summary = models.JSONField(
        default=dict,
        blank=True,
        help_text='Summary of analysis results'
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Analysis Session'
        verbose_name_plural = 'Analysis Sessions'
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['sam2_model']),
            models.Index(fields=['analysis_type']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        return f"{self.building_project.name} - {self.name}"
    
    @property
    def is_active(self):
        """Check if the analysis session is currently active"""
        return self.status in ['initializing', 'active', 'processing']
    
    @property
    def progress_percentage(self):
        """Calculate progress percentage based on frames analyzed"""
        if not self.video_file or not hasattr(self.video_file, 'frame_count'):
            return 0
        
        total_frames = getattr(self.video_file, 'frame_count', 0)
        if total_frames == 0:
            return 0
            
        return min(100, (self.total_frames_analyzed / total_frames) * 100)
    
    @property
    def duration_seconds(self):
        """Get the duration of the analysis session in seconds"""
        if not self.started_at:
            return 0
        
        end_time = self.completed_at or timezone.now()
        return (end_time - self.started_at).total_seconds()
    
    def get_model_info(self):
        """Get detailed information about the selected SAM 2 model"""
        model_info = {
            'tiny': {
                'parameters': '38.9M',
                'memory_usage': 'Low (~1-2GB)',
                'speed': 'Fastest',
                'quality': 'Good'
            },
            'small': {
                'parameters': '46.0M', 
                'memory_usage': 'Medium (~2-4GB)',
                'speed': 'Fast',
                'quality': 'Better'
            },
            'base_plus': {
                'parameters': '80.8M',
                'memory_usage': 'High (~4-6GB)', 
                'speed': 'Medium',
                'quality': 'High'
            },
            'large': {
                'parameters': '224.4M',
                'memory_usage': 'Very High (~6-8GB)',
                'speed': 'Slower',
                'quality': 'Best'
            }
        }
        return model_info.get(self.sam2_model, {})
    
    def get_recommended_model(self):
        """Get recommended model based on video characteristics"""
        if not self.video_file:
            return 'base_plus'
        
        # Get video info if available
        video_info = getattr(self.video_file, 'metadata', {})
        frame_count = video_info.get('frame_count', 0)
        is_4k = video_info.get('width', 0) >= 3840
        
        # Recommendation logic
        if is_4k and frame_count > 2000:
            return 'tiny'  # Large 4K videos
        elif is_4k and frame_count > 1000:
            return 'small'  # Medium 4K videos
        elif frame_count > 5000:
            return 'tiny'  # Very long videos
        elif frame_count > 2000:
            return 'small'  # Long videos
        else:
            return 'base_plus'  # Default for smaller videos
    
    def get_memory_estimate(self):
        """Estimate memory usage based on model and video characteristics"""
        base_memory = {
            'tiny': 1.5,    # GB
            'small': 2.5,   # GB
            'base_plus': 4.0,  # GB
            'large': 6.0    # GB
        }
        
        model_memory = base_memory.get(self.sam2_model, 4.0)
        
        # Add video-specific memory overhead
        if self.video_file:
            video_info = getattr(self.video_file, 'metadata', {})
            is_4k = video_info.get('width', 0) >= 3840
            frame_count = video_info.get('frame_count', 0)
            
            if is_4k:
                model_memory += 2.0  # Additional overhead for 4K
            if frame_count > 3000:
                model_memory += 1.0  # Additional overhead for long videos
        
        return model_memory
    
    def save(self, *args, **kwargs):
        """Override save to set timestamps and validate model selection"""
        # Set started_at when status changes to active/processing
        if self.status in ['active', 'processing'] and not self.started_at:
            self.started_at = timezone.now()
        
        # Set completed_at when status changes to completed/error
        if self.status in ['completed', 'error'] and not self.completed_at:
            self.completed_at = timezone.now()
        
        super().save(*args, **kwargs)
    
    def clean(self):
        """Validate the model selection"""
        from django.core.exceptions import ValidationError
        
        recommended_model = self.get_recommended_model()
        estimated_memory = self.get_memory_estimate()
        
        # Warn if using a large model for a large video
        if self.sam2_model == 'large' and estimated_memory > 8.0:
            raise ValidationError(
                f"Large model may require {estimated_memory:.1f}GB GPU memory. "
                f"Consider using '{recommended_model}' model instead."
            )


class ClickPrompt(models.Model):
    """Represents user clicks/prompts for segmentation"""
    PROMPT_TYPES = [
        ('positive', 'Positive Point'),
        ('negative', 'Negative Point'),
        ('box', 'Bounding Box'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    analysis_session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name='prompts')
    
    # Prompt details
    frame_index = models.IntegerField()
    object_id = models.IntegerField()  # SAM 2 object ID
    prompt_type = models.CharField(max_length=20, choices=PROMPT_TYPES)
    
    # Spatial coordinates
    coordinates = models.JSONField()  # [x, y] for points, [x1, y1, x2, y2] for boxes
    
    # Metadata
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['frame_index', 'created_at']

    def __str__(self):
        return f"Frame {self.frame_index} - {self.get_prompt_type_display()}"


class SegmentationResult(models.Model):
    """Stores segmentation masks and analysis results"""
    RESULT_TYPES = [
        ('initial', 'Initial Segmentation'),
        ('propagated', 'Propagated Segmentation'),
        ('manual_refined', 'Manually Refined'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    analysis_session = models.ForeignKey(AnalysisSession, on_delete=models.CASCADE, related_name='results')
    
    # Frame and object info
    frame_index = models.IntegerField()
    object_id = models.IntegerField()  # SAM 2 object ID
    result_type = models.CharField(max_length=20, choices=RESULT_TYPES)
    
    # Mask data
    mask_rle_counts = models.TextField()  # RLE compressed mask
    mask_rle_size = models.JSONField()  # [height, width]
    mask_area_pixels = models.IntegerField()  # Number of pixels in mask
    
    # Material and thermal analysis
    identified_material = models.CharField(max_length=100, blank=True)  # brick, concrete, glass, etc.
    material_confidence = models.FloatField(null=True, blank=True)  # 0.0 to 1.0
    
    # Thermal data (if available)
    avg_temperature = models.FloatField(null=True, blank=True)
    min_temperature = models.FloatField(null=True, blank=True)
    max_temperature = models.FloatField(null=True, blank=True)
    thermal_anomaly_score = models.FloatField(null=True, blank=True)  # 0.0 to 1.0
    
    # Geometric properties
    centroid = models.PointField(null=True, blank=True)  # Center point of segmented area
    bounding_box = models.PolygonField(null=True, blank=True)  # Bounding rectangle
    
    # Quality metrics
    prediction_confidence = models.FloatField(null=True, blank=True)  # SAM 2 confidence
    edge_quality_score = models.FloatField(null=True, blank=True)  # Edge detection quality
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['frame_index', 'object_id']
        indexes = [
            models.Index(fields=['analysis_session', 'frame_index']),
            models.Index(fields=['analysis_session', 'object_id']),
            models.Index(fields=['identified_material']),
        ]

    def __str__(self):
        return f"Frame {self.frame_index}, Object {self.object_id} - {self.identified_material or 'Unknown'}"

    def get_mask_as_numpy(self):
        """Convert RLE mask back to numpy array"""
        from pycocotools.mask import decode as decode_masks
        rle_mask = {
            "counts": self.mask_rle_counts,
            "size": self.mask_rle_size,
        }
        return decode_masks(rle_mask)


class ThermalAnomalyDetection(models.Model):
    """Stores thermal anomaly detection results"""
    ANOMALY_TYPES = [
        ('hot_spot', 'Hot Spot'),
        ('cold_spot', 'Cold Spot'),
        ('thermal_bridge', 'Thermal Bridge'),
        ('insulation_defect', 'Insulation Defect'),
        ('air_leakage', 'Air Leakage'),
        ('moisture_intrusion', 'Moisture Intrusion'),
    ]
    
    SEVERITY_LEVELS = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    segmentation_result = models.ForeignKey(SegmentationResult, on_delete=models.CASCADE, related_name='anomalies')
    
    # Anomaly classification
    anomaly_type = models.CharField(max_length=30, choices=ANOMALY_TYPES)
    severity = models.CharField(max_length=20, choices=SEVERITY_LEVELS)
    confidence_score = models.FloatField()  # 0.0 to 1.0
    
    # Temperature analysis
    temperature_deviation = models.FloatField()  # Deviation from expected temperature
    baseline_temperature = models.FloatField()  # Expected temperature for this material/condition
    
    # Spatial analysis
    affected_area_sqm = models.FloatField(null=True, blank=True)  # Area in square meters
    location_description = models.CharField(max_length=200, blank=True)  # "North wall, 3rd floor window"
    
    # Recommendations
    recommended_action = models.TextField(blank=True)
    priority_level = models.IntegerField(default=3)  # 1 (highest) to 5 (lowest)
    estimated_cost_impact = models.CharField(max_length=20, blank=True)  # low, medium, high
    
    # Tracking
    acknowledged = models.BooleanField(default=False)
    resolved = models.BooleanField(default=False)
    resolution_notes = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['priority_level', '-confidence_score']

    def __str__(self):
        return f"{self.get_anomaly_type_display()} - {self.get_severity_display()}"


class BuildingMaterial(models.Model):
    """Database of building materials and their thermal properties"""
    MATERIAL_CATEGORIES = [
        ('masonry', 'Masonry'),
        ('metal', 'Metal'),
        ('wood', 'Wood'),
        ('glass', 'Glass'),
        ('insulation', 'Insulation'),
        ('roofing', 'Roofing'),
        ('concrete', 'Concrete'),
        ('composite', 'Composite'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=100, unique=True)
    category = models.CharField(max_length=20, choices=MATERIAL_CATEGORIES)
    description = models.TextField(blank=True)
    
    # Thermal properties
    thermal_conductivity = models.FloatField(null=True, blank=True)  # W/mK
    specific_heat = models.FloatField(null=True, blank=True)  # J/kgK
    density = models.FloatField(null=True, blank=True)  # kg/m³
    emissivity = models.FloatField(null=True, blank=True)  # 0.0 to 1.0
    
    # Expected temperature ranges
    typical_temp_min = models.FloatField(null=True, blank=True)
    typical_temp_max = models.FloatField(null=True, blank=True)
    
    # Visual identification features
    visual_features = models.JSONField(default=dict)  # Color, texture, pattern descriptors
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['category', 'name']

    def __str__(self):
        return f"{self.name} ({self.get_category_display()})"