# backend/apps/analysis/admin.py
from django.contrib import admin
from .models import (
    AnalysisSession, SegmentationResult, ClickPrompt, 
    ThermalAnomalyDetection, BuildingMaterial
)


@admin.register(AnalysisSession)
class AnalysisSessionAdmin(admin.ModelAdmin):
    list_display = ['name', 'building_project', 'user', 'status', 'created_at']
    list_filter = ['status', 'analysis_type', 'created_at']
    search_fields = ['name', 'building_project__name']


@admin.register(SegmentationResult)
class SegmentationResultAdmin(admin.ModelAdmin):
    list_display = ['analysis_session', 'frame_index', 'object_id', 'identified_material', 'created_at']
    list_filter = ['identified_material', 'result_type', 'created_at']


@admin.register(BuildingMaterial)
class BuildingMaterialAdmin(admin.ModelAdmin):
    list_display = ['name', 'category', 'thermal_conductivity', 'emissivity']
    list_filter = ['category']
    search_fields = ['name', 'description']