# backend/apps/buildings/admin.py - Enhanced GIS Admin
from django.contrib import admin
from django.contrib.gis import admin as gis_admin
from .models import BuildingProject, VideoFile


@admin.register(BuildingProject)
class BuildingProjectAdmin(gis_admin.GISModelAdmin):
    list_display = ['name', 'owner', 'created_at']
    list_filter = ['created_at', 'owner']
    search_fields = ['name', 'description']
    readonly_fields = ['id', 'created_at', 'updated_at']
    
    # Better map settings
    gis_widget_kwargs = {
        'attrs': {
            'default_zoom': 15,
            'default_lat': 40.7128,  # New York as default, adjust to your area
            'default_lon': -74.0060,
        },
    }
    
    # Use higher quality map tiles
    openlayers_url = 'https://cdnjs.cloudflare.com/ajax/libs/openlayers/2.13.1/OpenLayers.js'
    
    # Optional: Use custom map settings
    map_template = 'gis/admin/openlayers.html'
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('name', 'description', 'owner')
        }),
        ('Location', {
            'fields': ('location', 'building_footprint'),
            'classes': ('wide',),
        }),
        ('Metadata', {
            'fields': ('id', 'created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )


@admin.register(VideoFile) 
class VideoFileAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'building_project', 'video_type', 'processing_status', 'created_at']
    list_filter = ['video_type', 'processing_status', 'created_at']
    search_fields = ['original_filename', 'building_project__name']
    readonly_fields = ['id', 'created_at', 'updated_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('building_project', 'video_type', 'original_filename')
        }),
        ('File Storage', {
            'fields': ('file_key', 'file_size', 'streaming_manifest_key')
        }),
        ('Video Properties', {
            'fields': ('duration_seconds', 'width', 'height', 'fps', 'total_frames'),
            'classes': ('collapse',),
        }),
        ('Processing', {
            'fields': ('processing_status', 'processing_error')
        }),
        ('Metadata', {
            'fields': ('id', 'created_at', 'updated_at'),
            'classes': ('collapse',),
        }),
    )