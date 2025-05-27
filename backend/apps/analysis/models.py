from django.contrib.auth.models import User
from django.contrib.gis.db import models
from django.contrib.gis.geos import Point, Polygon
from apps.buildings.models import BuildingProject, VideoFile
import uuid


class AnalysisSession(models.Model):
    """Represents a video analysis session using SAM 2"""
    SESSION_STATUSES = [
        ('initializing', 'Initializing'),
        ('active', 'Active'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('error', 'Error'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    building_project = models.ForeignKey(BuildingProject, on_delete=models.CASCADE, related_name='analysis_sessions')
    video_file = models.ForeignKey(VideoFile, on_delete=models.CASCADE, related_name='analysis_sessions')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Session metadata
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    status = models.CharField(max_length=20, choices=SESSION_STATUSES, default='initializing')
    
    # SAM 2 session info
    sam2_session_id = models.CharField(max_length=100, blank=True)  # UUID from SAM 2 service
    
    # Analysis parameters
    target_materials = models.JSONField(default=list)  # List of materials to focus on
    analysis_type = models.CharField(max_length=50, default='thermal_anomaly')  # thermal_anomaly, material_mapping, etc.
    
    # Progress tracking
    total_frames_analyzed = models.IntegerField(default=0)
    total_objects_tracked = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.building_project.name} - {self.name}"


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