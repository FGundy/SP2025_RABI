# backend/apps/uploads/serializers.py
from rest_framework import serializers
from .models import UploadSession


class UploadSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadSession
        fields = [
            'id', 'original_filename', 'total_size', 'progress_percentage',
            'status', 'created_at'
        ]
        read_only_fields = ['id', 'progress_percentage', 'created_at']