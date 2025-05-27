# backend/apps/analysis/serializers.py
from rest_framework import serializers
from .models import AnalysisSession, SegmentationResult


class AnalysisSessionSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisSession
        fields = [
            'id', 'building_project', 'video_file', 'name', 'description',
            'status', 'target_materials', 'analysis_type', 'created_at'
        ]
        read_only_fields = ['id', 'created_at']


class SegmentationResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = SegmentationResult
        fields = [
            'id', 'analysis_session', 'frame_index', 'object_id',
            'identified_material', 'avg_temperature', 'thermal_anomaly_score',
            'created_at'
        ]
        read_only_fields = ['id', 'created_at']
