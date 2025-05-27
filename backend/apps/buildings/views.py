# backend/apps/buildings/views.py
from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from .models import BuildingProject, VideoFile
from .serializers import BuildingProjectSerializer, VideoFileSerializer


class BuildingProjectViewSet(viewsets.ModelViewSet):
    serializer_class = BuildingProjectSerializer
    permission_classes = [IsAuthenticated]
    queryset = BuildingProject.objects.all()  # Define base queryset
    
    def get_queryset(self):
        # Filter by current user
        return BuildingProject.objects.filter(owner=self.request.user)

    def perform_create(self, serializer):
        # Automatically set the owner to the current user
        serializer.save(owner=self.request.user)


class VideoFileViewSet(viewsets.ModelViewSet):
    serializer_class = VideoFileSerializer
    permission_classes = [IsAuthenticated]
    queryset = VideoFile.objects.all()  # Define base queryset
    
    def get_queryset(self):
        # Filter by videos belonging to projects owned by current user
        return VideoFile.objects.filter(building_project__owner=self.request.user)

