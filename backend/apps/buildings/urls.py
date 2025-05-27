# backend/apps/buildings/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'projects', views.BuildingProjectViewSet)
router.register(r'videos', views.VideoFileViewSet)

urlpatterns = [
    path('', include(router.urls)),
]

