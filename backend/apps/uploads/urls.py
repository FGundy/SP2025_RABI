# backend/apps/uploads/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('initiate/', views.InitiateUploadView.as_view(), name='initiate-upload'),
    path('chunk/', views.UploadChunkView.as_view(), name='upload-chunk'),
    path('finalize/', views.FinalizeUploadView.as_view(), name='finalize-upload'),
    path('status/<uuid:session_id>/', views.UploadStatusView.as_view(), name='upload-status'),
]
