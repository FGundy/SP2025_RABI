from django.contrib.auth.models import User
from django.db import models
from apps.buildings.models import BuildingProject
import uuid


class UploadSession(models.Model):
    """Tracks chunked upload progress"""
    UPLOAD_STATUSES = [
        ('active', 'Upload Active'),
        ('completed', 'Upload Completed'),
        ('failed', 'Upload Failed'),
        ('expired', 'Upload Expired'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    building_project = models.ForeignKey(BuildingProject, on_delete=models.CASCADE)
    
    # Upload metadata
    original_filename = models.CharField(max_length=255)
    total_size = models.BigIntegerField()
    chunk_size = models.IntegerField(default=5 * 1024 * 1024)  # 5MB chunks
    total_chunks = models.IntegerField()
    
    # Progress tracking
    uploaded_chunks = models.JSONField(default=list)  # List of uploaded chunk numbers
    status = models.CharField(max_length=20, choices=UPLOAD_STATUSES, default='active')
    
    # MinIO multipart upload info
    upload_id = models.CharField(max_length=500, blank=True)  # MinIO upload ID
    temp_key_prefix = models.CharField(max_length=500, blank=True)  # MinIO temp storage prefix
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField()  # Upload session expiration

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Upload: {self.original_filename} ({self.status})"

    @property
    def progress_percentage(self):
        """Calculate upload progress percentage"""
        return (len(self.uploaded_chunks) / self.total_chunks) * 100 if self.total_chunks > 0 else 0

    @property
    def is_complete(self):
        """Check if all chunks have been uploaded"""
        return len(self.uploaded_chunks) == self.total_chunks