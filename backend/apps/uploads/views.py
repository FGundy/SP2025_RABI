# backend/apps/uploads/views.py
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from .models import UploadSession


class InitiateUploadView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        return Response({"message": "Initiate upload endpoint - coming soon"})


class UploadChunkView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        return Response({"message": "Upload chunk endpoint - coming soon"})


class FinalizeUploadView(APIView):
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        return Response({"message": "Finalize upload endpoint - coming soon"})


class UploadStatusView(APIView):
    permission_classes = [IsAuthenticated]
    
    def get(self, request, session_id):
        return Response({"message": f"Upload status for {session_id} - coming soon"})