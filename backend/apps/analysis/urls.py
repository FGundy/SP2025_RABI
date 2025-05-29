# backend/apps/analysis/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'sessions', views.AnalysisSessionViewSet, basename='analysissession')
router.register(r'results', views.SegmentationResultViewSet, basename='segmentationresult')
router.register(r'prompts', views.ClickPromptViewSet, basename='clickprompt')

urlpatterns = [
    # REST API routes
    path('', include(router.urls)),

    # Custom analysis endpoints
    path('start-session/', views.start_analysis_session, name='start-analysis-session'),
    path('sessions/<uuid:session_id>/add-points/', views.add_segmentation_points, name='add-points'),
    path('sessions/<uuid:session_id>/propagate/', views.propagate_segmentation, name='propagate'),
    path('sessions/<uuid:session_id>/recommendations/', views.get_model_recommendations, name='model-recommendations'),
    path('sessions/<uuid:session_id>/update-model/', views.update_model_selection, name='update-model'),

    # Interactive video endpoints
    path('interactive/start/', views.start_interactive_session, name='start-interactive-session'),
    path('interactive/<str:session_id>/info/', views.get_interactive_session_info, name='get-interactive-session-info'),
    path('interactive/<str:session_id>/close/', views.close_interactive_session, name='close-interactive-session'),
    path('interactive/<str:session_id>/playback/', views.video_playback_control, name='video-playback-control'),
    path('interactive/<str:session_id>/enter-segmentation/', views.enter_segmentation_mode, name='enter-segmentation-mode'),
    path('interactive/<str:session_id>/exit-segmentation/', views.exit_segmentation_mode, name='exit-segmentation-mode'),
    path('interactive/<str:session_id>/add-prompt/', views.add_segmentation_prompt, name='add-segmentation-prompt'),
    path('interactive/<str:session_id>/clear-masks/', views.clear_masks, name='clear-masks'),
    path('interactive/<str:session_id>/save-masks/', views.save_masks, name='save-masks'),
    path('interactive/<str:session_id>/load-masks/', views.load_masks, name='load-masks'),
    path('interactive/<str:session_id>/status/', views.get_session_status, name='get-session-status'),
    path('interactive/<str:session_id>/close/', views.close_interactive_session, name='close-interactive-session'),

    # Legacy endpoints (for backward compatibility)
    path('start-analysis/', views.StartAnalysisView.as_view(), name='start-analysis'),
    path('add-points/', views.AddPointsView.as_view(), name='add-points-legacy'),
    path('propagate/', views.PropagateView.as_view(), name='propagate-legacy'),
]
