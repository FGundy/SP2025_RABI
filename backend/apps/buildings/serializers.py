# backend/apps/buildings/serializers.py
from rest_framework import serializers
from .models import BuildingProject, VideoFile


class BuildingProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = BuildingProject
        fields = ['id', 'name', 'description', 'location', 'created_at', 'updated_at']
        read_only_fields = ['id', 'created_at', 'updated_at']


class VideoFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoFile
        fields = [
            'id', 'building_project', 'video_type', 'original_filename',
            'file_size', 'duration_seconds', 'width', 'height', 'fps',
            'processing_status', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']
