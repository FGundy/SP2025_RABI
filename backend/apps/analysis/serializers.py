# backend/apps/analysis/serializers.py
from rest_framework import serializers
from .models import AnalysisSession, SegmentationResult, ClickPrompt


class AnalysisSessionSerializer(serializers.ModelSerializer):
    # Add read-only fields for additional info
    model_info = serializers.SerializerMethodField()
    recommended_model = serializers.SerializerMethodField()
    memory_estimate = serializers.SerializerMethodField()
    progress_percentage = serializers.ReadOnlyField()
    duration_seconds = serializers.ReadOnlyField()
    is_active = serializers.ReadOnlyField()
    
    class Meta:
        model = AnalysisSession
        fields = [
            'id', 'building_project', 'video_file', 'name', 'description',
            'sam2_model', 'analysis_type', 'status', 'target_materials',
            'confidence_threshold', 'enable_tracking', 'offload_to_cpu',
            'max_memory_frames', 'error_message', 'results_summary',
            'total_frames_analyzed', 'total_objects_tracked',
            'created_at', 'updated_at', 'started_at', 'completed_at',
            # Read-only computed fields
            'model_info', 'recommended_model', 'memory_estimate',
            'progress_percentage', 'duration_seconds', 'is_active'
        ]
        read_only_fields = [
            'id', 'sam2_session_id', 'total_frames_analyzed', 'total_objects_tracked',
            'created_at', 'updated_at', 'started_at', 'completed_at'
        ]
        
    def get_model_info(self, obj):
        """Get detailed model information"""
        return obj.get_model_info()
    
    def get_recommended_model(self, obj):
        """Get recommended model for this video"""
        return obj.get_recommended_model()
    
    def get_memory_estimate(self, obj):
        """Get estimated memory usage"""
        return obj.get_memory_estimate()
        
    def validate_sam2_model(self, value):
        """Validate model selection based on video characteristics"""
        # Get video file from initial data or instance
        video_file = None
        if self.instance:
            video_file = self.instance.video_file
        elif 'video_file' in self.initial_data:
            # If creating new instance, you might need to fetch the video file
            try:
                from apps.buildings.models import VideoFile
                video_file = VideoFile.objects.get(pk=self.initial_data['video_file'])
            except VideoFile.DoesNotExist:
                pass
        
        if video_file:
            # Create a temporary instance to get recommendations
            temp_session = AnalysisSession(video_file=video_file, sam2_model=value)
            recommended = temp_session.get_recommended_model()
            estimated_memory = temp_session.get_memory_estimate()
            
            # Add warning if memory usage is high
            if estimated_memory > 8.0:
                raise serializers.ValidationError(
                    f"Selected model may require {estimated_memory:.1f}GB GPU memory. "
                    f"Consider using '{recommended}' model for better performance."
                )
        
        return value
    
    def validate(self, attrs):
        """Validate the entire analysis session"""
        # Ensure required fields are present
        if not attrs.get('name'):
            attrs['name'] = f"Analysis - {attrs.get('sam2_model', 'Unknown').title()}"
            
        return attrs


class ClickPromptSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClickPrompt
        fields = [
            'id', 'analysis_session', 'frame_index', 'object_id',
            'prompt_type', 'coordinates', 'user', 'created_at'
        ]
        read_only_fields = ['id', 'user', 'created_at']


class SegmentationResultSerializer(serializers.ModelSerializer):
    # Add method to get mask as base64 encoded image
    mask_preview = serializers.SerializerMethodField()
    
    class Meta:
        model = SegmentationResult
        fields = [
            'id', 'analysis_session', 'frame_index', 'object_id', 'result_type',
            'mask_rle_counts', 'mask_rle_size', 'mask_area_pixels',
            'identified_material', 'material_confidence',
            'avg_temperature', 'min_temperature', 'max_temperature', 'thermal_anomaly_score',
            'centroid', 'bounding_box', 'prediction_confidence', 'edge_quality_score',
            'created_at', 'updated_at', 'mask_preview'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at', 'mask_preview']
    
    def get_mask_preview(self, obj):
        """Get a base64 encoded preview of the mask (optional)"""
        # This could generate a small preview image of the mask
        # For now, return None - implement if needed
        return None
