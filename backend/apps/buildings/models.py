# backend/apps/buildings/models.py
from django.contrib.auth.models import User
from django.contrib.gis.db import models
from django.contrib.gis.geos import Point, Polygon
import uuid


class BuildingProject(models.Model):
    """Represents a building being analyzed"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='building_projects')
    location = models.PointField(null=True, blank=True)  # Building GPS coordinates
    building_footprint = models.PolygonField(null=True, blank=True)  # Building outline
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return self.name


class VideoFile(models.Model):
    """Represents uploaded video files (visual/thermal)"""
    VIDEO_TYPES = [
        ('visual', 'Visual/RGB Video'),
        ('thermal', 'Thermal/IRX Video'),
        ('combined', 'Combined Visual+Thermal'),
    ]
    
    PROCESSING_STATUSES = [
        ('pending', 'Upload Pending'),
        ('processing', 'Processing'),
        ('ready', 'Ready for Analysis'),
        ('error', 'Processing Error'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    building_project = models.ForeignKey(BuildingProject, on_delete=models.CASCADE, related_name='videos')
    video_type = models.CharField(max_length=20, choices=VIDEO_TYPES)
    original_filename = models.CharField(max_length=255)
    file_key = models.CharField(max_length=500)  # MinIO object key
    file_size = models.BigIntegerField()  # Size in bytes
    
    # Video metadata (extracted after upload)
    duration_seconds = models.FloatField(null=True, blank=True)
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    fps = models.FloatField(null=True, blank=True)
    total_frames = models.IntegerField(null=True, blank=True)
    
    # Processing status
    processing_status = models.CharField(max_length=20, choices=PROCESSING_STATUSES, default='pending')
    processing_error = models.TextField(blank=True)
    
    # HLS/DASH streaming info
    streaming_manifest_key = models.CharField(max_length=500, blank=True)  # MinIO key for .m3u8/.mpd
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        # Better display: "MAX_5360.MP4 (RWH - Visual/RGB)"
        return f"{self.original_filename} ({self.building_project.name} - {self.get_video_type_display()})"
