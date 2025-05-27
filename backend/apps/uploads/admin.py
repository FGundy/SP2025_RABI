# backend/apps/uploads/admin.py
from django.contrib import admin
from .models import UploadSession


@admin.register(UploadSession)
class UploadSessionAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'user', 'progress_percentage', 'status', 'created_at']
    list_filter = ['status', 'created_at']
    search_fields = ['original_filename', 'user__username']