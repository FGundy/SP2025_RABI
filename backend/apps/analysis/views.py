# backend/apps/analysis/views.py
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from .models import AnalysisSession, SegmentationResult
from .serializers import AnalysisSessionSerializer, SegmentationResultSerializer


class AnalysisSessionViewSet(viewsets.ModelViewSet):
    serializer_class = AnalysisSessionSerializer
    permission_classes = [IsAuthenticated]
    queryset = AnalysisSession.objects.all()  # Define base queryset
    
    def get_queryset(self):
        # Filter by current user
        return AnalysisSession.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        # Automatically set the user to the current user
        serializer.save(user=self.request.user)


class SegmentationResultViewSet(viewsets.ModelViewSet):
    serializer_class = SegmentationResultSerializer
    permission_classes = [IsAuthenticated]
    queryset = SegmentationResult.objects.all()  # Define base queryset
    
    def get_queryset(self):
        # Filter by results belonging to sessions owned by current user
        return SegmentationResult.objects.filter(
            analysis_session__user=self.request.user
        )


class StartAnalysisView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """Start a new SAM 2 analysis session"""
        return Response({"message": "Analysis endpoint - coming soon"})


class AddPointsView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """Add segmentation points"""
        return Response({"message": "Add points endpoint - coming soon"})


class PropagateView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        """Propagate segmentation"""
        return Response({"message": "Propagate endpoint - coming soon"})
