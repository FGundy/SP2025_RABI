# backend/apps/analysis/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'sessions', views.AnalysisSessionViewSet)
router.register(r'results', views.SegmentationResultViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('start-session/', views.StartAnalysisView.as_view(), name='start-analysis'),
    path('add-points/', views.AddPointsView.as_view(), name='add-points'),
    path('propagate/', views.PropagateView.as_view(), name='propagate'),
]
