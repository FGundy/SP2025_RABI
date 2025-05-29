
# backend/apps/analysis/views.py
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from .models import AnalysisSession, SegmentationResult, ClickPrompt
from .serializers import AnalysisSessionSerializer, SegmentationResultSerializer, ClickPromptSerializer



from sam2_worker.sam2_service.predictor import ThermalSegmentationAPI, HybridThermalSegmentationAPI
from sam2_worker.sam2_service.interactive_session import InteractiveVideoSession
from minio import Minio
import os
import logging
import tempfile

logger = logging.getLogger(__name__)

# Global session manager (in production, use Redis or database)
interactive_sessions = {}

# MinIO client
minio_client = Minio(
    endpoint=os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
    access_key=os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
    secret_key=os.getenv('MINIO_SECRET_KEY', 'minioadmin123'),
    secure=False
)

def get_dynamic_chunk_size(video_info: dict, model_size: str) -> int:
    """
    Dynamically determine optimal chunk size based on video characteristics and model size
    
    Args:
        video_info: Video metadata (width, height, frame_count, duration)
        model_size: SAM 2 model size (tiny, small, base_plus, large)
        
    Returns:
        Optimal chunk size in frames
    """
    width = video_info.get('width', 1920)
    height = video_info.get('height', 1080)
    frame_count = video_info.get('frame_count', 1000)
    is_4k = width >= 3840
    is_long_video = frame_count > 3000
    
    # Base chunk sizes by model
    base_chunks = {
        'tiny': 50,      # Default 50 for tiny model
        'small': 40,     # Slightly smaller for small model
        'base_plus': 30, # Smaller for base_plus model
        'large': 20      # Very small for large model
    }
    
    base_chunk = base_chunks.get(model_size, 50)
    
    # Adjust based on video characteristics
    if is_4k:
        if model_size == 'tiny':
            chunk_size = 25  # Very small chunks for 4K with tiny model
        elif model_size == 'small':
            chunk_size = 20  # Even smaller for small model
        else:
            chunk_size = 15  # Ultra-small for larger models
    elif width >= 2560:  # 2K video
        chunk_size = int(base_chunk * 0.8)  # 80% of base
    else:  # 1080p or lower
        chunk_size = base_chunk
    
    # Adjust for very long videos
    if is_long_video:
        chunk_size = max(10, int(chunk_size * 0.7))  # Reduce by 30%, minimum 10
    
    # Ensure reasonable bounds
    chunk_size = max(10, min(100, chunk_size))
    
    logger.info(f"🎯 Dynamic chunk size calculation:")
    logger.info(f"   📏 Resolution: {width}x{height}")
    logger.info(f"   🤖 Model: {model_size}")
    logger.info(f"   📊 Frames: {frame_count}")
    logger.info(f"   📦 Chunk size: {chunk_size} frames")
    
    return chunk_size

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_interactive_session(request):
    """
    Start an interactive video session with dynamic chunking and enhanced error handling
    """
    analysis_session_id = request.data.get('analysis_session_id')
    if not analysis_session_id:
        return Response({
            'error': 'analysis_session_id required',
            'code': 'MISSING_SESSION_ID'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Get Django analysis session
    try:
        analysis_session = get_object_or_404(
            AnalysisSession.objects.filter(user=request.user),
            id=analysis_session_id
        )
    except Exception as e:
        return Response({
            'error': 'Analysis session not found or access denied',
            'code': 'SESSION_NOT_FOUND'
        }, status=status.HTTP_404_NOT_FOUND)
    
    temp_video_path = None
    
    try:
        logger.info(f"🚀 Starting interactive session for analysis {analysis_session_id}")
        logger.info(f"   👤 User: {request.user.username}")
        logger.info(f"   🎬 Video: {analysis_session.video_file.file_path}")
        logger.info(f"   🤖 Model: {analysis_session.sam2_model}")
        
        # Step 1: Download video from MinIO to analyze
        logger.info("📥 Step 1: Downloading video for analysis...")
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video_path = temp_video.name
        temp_video.close()
        
        minio_client.fget_object(
            bucket_name="videos",
            object_name=analysis_session.video_file.file_path,
            file_path=temp_video_path
        )
        
        # Step 2: Analyze video characteristics for dynamic chunking
        logger.info("📊 Step 2: Analyzing video for optimal settings...")
        from sam2_service.predictor import BaseThermalSegmentationAPI
        
        # Create temporary API instance just to get video info
        temp_api = BaseThermalSegmentationAPI(model_size='tiny')
        video_info = temp_api.get_video_info_safe(temp_video_path)
        
        if not video_info:
            raise ValueError("Could not analyze video file - may be corrupted")
        
        # Step 3: Calculate dynamic chunk size
        dynamic_chunk_size = get_dynamic_chunk_size(video_info, analysis_session.sam2_model)
        
        # Step 4: Create Hybrid API with dynamic chunk size
        logger.info(f"🎯 Step 3: Creating Hybrid API with {dynamic_chunk_size}-frame chunks...")
        thermal_api = HybridThermalSegmentationAPI(
            minio_client=minio_client,
            model_size=analysis_session.sam2_model,
            force_cpu_frames=True,  # Always use CPU offloading
            chunk_size=dynamic_chunk_size  # Dynamic chunk sizing!
        )
        
        # Step 5: Create interactive session
        logger.info("🎬 Step 4: Creating interactive session...")
        interactive_session = InteractiveVideoSession(thermal_api, str(analysis_session.id))
        
        # Step 6: Load video for viewing
        logger.info("📽️  Step 5: Loading video for viewing...")
        result = interactive_session.load_video(temp_video_path)
        
        # Step 7: Store session globally (use Redis in production)
        interactive_sessions[interactive_session.session_id] = {
            'session': interactive_session,
            'analysis_session_id': str(analysis_session.id),
            'user_id': request.user.id,
            'temp_video_path': temp_video_path,
            'created_at': timezone.now().isoformat(),
            'video_info': video_info,
            'chunk_size': dynamic_chunk_size
        }
        
        # Step 8: Enhanced response with all the details
        enhanced_result = {
            **result,
            'analysis_session_id': str(analysis_session.id),
            'user_id': request.user.id,
            'interactive_session_id': interactive_session.session_id,
            'sam2_model': analysis_session.sam2_model,
            'chunk_size': dynamic_chunk_size,
            'memory_optimization': {
                'cpu_offloading': True,
                'dynamic_chunking': True,
                'model_memory_estimate': f"{thermal_api.chunk_size * 0.3:.1f}MB per chunk"
            },
            'capabilities': {
                'can_segment': True,
                'supports_points': True,
                'supports_boxes': False,  # Not implemented yet
                'can_save_masks': True,
                'can_load_masks': True
            },
            'performance_info': {
                'processing_mode': 'hybrid_cpu_gpu',
                'chunk_strategy': 'dynamic',
                'memory_efficient': True,
                'recommended_for_4k': video_info.get('is_4k', False)
            }
        }
        
        logger.info(f"✅ Interactive session created successfully!")
        logger.info(f"   🆔 Session ID: {interactive_session.session_id}")
        logger.info(f"   📦 Chunk size: {dynamic_chunk_size} frames")
        logger.info(f"   💾 Memory mode: Hybrid CPU/GPU")
        
        return Response(enhanced_result, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        logger.error(f"❌ Failed to start interactive session: {e}")
        
        # Cleanup on failure
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except:
                pass
        
        # Enhanced error response
        error_response = {
            'error': str(e),
            'code': 'SESSION_CREATION_FAILED',
            'analysis_session_id': str(analysis_session_id),
            'suggested_actions': []
        }
        
        # Add specific suggestions based on error type
        error_str = str(e).lower()
        if 'memory' in error_str or 'out of memory' in error_str:
            error_response['suggested_actions'].extend([
                'Try using a smaller SAM 2 model (tiny instead of base_plus)',
                'Close other applications to free up GPU memory',
                'Consider using a shorter video clip for testing'
            ])
        elif 'video' in error_str or 'corrupted' in error_str:
            error_response['suggested_actions'].extend([
                'Check if the video file is valid and not corrupted',
                'Try uploading the video file again',
                'Ensure the video format is supported (MP4, AVI, MOV)'
            ])
        else:
            error_response['suggested_actions'].append(
                'Check the server logs for more detailed error information'
            )
        
        return Response(error_response, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_interactive_session_info(request, session_id):
    """
    Get detailed information about an interactive session
    """
    if session_id not in interactive_sessions:
        return Response({
            'error': 'Interactive session not found',
            'code': 'SESSION_NOT_FOUND'
        }, status=status.HTTP_404_NOT_FOUND)
    
    session_data = interactive_sessions[session_id]
    
    # Check if user has access to this session
    if session_data['user_id'] != request.user.id:
        return Response({
            'error': 'Access denied to this session',
            'code': 'ACCESS_DENIED'
        }, status=status.HTTP_403_FORBIDDEN)
    
    interactive_session = session_data['session']
    
    # Get current session status
    status_info = interactive_session.get_session_status()
    
    # Enhanced session info
    enhanced_info = {
        **status_info,
        'analysis_session_id': session_data['analysis_session_id'],
        'created_at': session_data['created_at'],
        'chunk_size': session_data['chunk_size'],
        'video_characteristics': session_data['video_info'],
        'memory_optimization': {
            'mode': 'hybrid_cpu_gpu',
            'chunk_strategy': 'dynamic',
            'cpu_offloading': True
        }
    }
    
    return Response(enhanced_info)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def close_interactive_session(request, session_id):
    """
    Close interactive session with enhanced cleanup
    """
    if session_id not in interactive_sessions:
        return Response({
            'error': 'Interactive session not found',
            'code': 'SESSION_NOT_FOUND'
        }, status=status.HTTP_404_NOT_FOUND)
    
    session_data = interactive_sessions[session_id]
    
    # Check user access
    if session_data['user_id'] != request.user.id:
        return Response({
            'error': 'Access denied to this session',
            'code': 'ACCESS_DENIED'
        }, status=status.HTTP_403_FORBIDDEN)
    
    try:
        interactive_session = session_data['session']
        
        # Exit segmentation mode if active
        if interactive_session.mode.value == 'segmenting':
            interactive_session.exit_segmentation_mode(save_masks=False)
        
        # Cleanup temporary video file
        temp_video_path = session_data.get('temp_video_path')
        if temp_video_path and os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
            logger.info(f"🧹 Cleaned up temp video: {temp_video_path}")
        
        # Remove from global sessions
        del interactive_sessions[session_id]
        
        logger.info(f"✅ Interactive session {session_id} closed successfully")
        
        return Response({
            'session_id': session_id,
            'status': 'closed',
            'message': 'Interactive session closed and cleaned up successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ Error closing session {session_id}: {e}")
        return Response({
            'error': str(e),
            'code': 'CLEANUP_ERROR'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def video_playback_control(request, session_id):
    """
    Control video playback (play, pause, seek)
    """
    session = interactive_sessions.get(session_id)
    if not session:
        return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
    
    action = request.data.get('action')
    
    if action == 'play':
        speed = request.data.get('speed', 1.0)
        result = session.play_video(speed)
    elif action == 'pause':
        result = session.pause_video()
    elif action == 'seek':
        frame_index = request.data.get('frame_index')
        if frame_index is None:
            return Response({'error': 'frame_index required for seek'}, status=status.HTTP_400_BAD_REQUEST)
        result = session.seek_to_frame(frame_index)
    else:
        return Response({'error': 'Invalid action. Use: play, pause, seek'}, status=status.HTTP_400_BAD_REQUEST)
    
    return Response(result)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def enter_segmentation_mode(request, session_id):
    """
    Enter segmentation mode - SAM 2 initializes here!
    """
    session = interactive_sessions.get(session_id)
    if not session:
        return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
    
    # User can override model choice
    model_size = request.data.get('model_size')  # User's choice!
    confidence_threshold = request.data.get('confidence_threshold', 0.5)
    
    result = session.enter_segmentation_mode(model_size, confidence_threshold)
    
    if 'error' in result:
        return Response(result, status=status.HTTP_400_BAD_REQUEST)
    
    return Response(result)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def exit_segmentation_mode(request, session_id):
    """
    Exit segmentation mode and return to viewing
    """
    session = interactive_sessions.get(session_id)
    if not session:
        return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
    
    save_masks = request.data.get('save_masks', False)
    result = session.exit_segmentation_mode(save_masks)
    
    return Response(result)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def add_segmentation_prompt(request, session_id):
    """
    Add segmentation prompt (points, bounding box)
    """
    session = interactive_sessions.get(session_id)
    if not session:
        return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
    
    prompt_type = request.data.get('prompt_type')
    coordinates = request.data.get('coordinates')
    object_id = request.data.get('object_id', 1)
    
    if not prompt_type or not coordinates:
        return Response(
            {'error': 'prompt_type and coordinates required'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    result = session.add_prompt(prompt_type, coordinates, object_id)
    
    if 'error' in result:
        return Response(result, status=status.HTTP_400_BAD_REQUEST)
    
    return Response(result)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def clear_masks(request, session_id):
    """
    Clear masks for current or specific frame
    """
    session = interactive_sessions.get(session_id)
    if not session:
        return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
    
    frame_index = request.data.get('frame_index')  # Optional
    result = session.clear_masks(frame_index)
    
    return Response(result)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def save_masks(request, session_id):
    """
    Save current masks to database
    """
    session = interactive_sessions.get(session_id)
    if not session:
        return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
    
    result = session.save_masks_to_database()
    
    return Response(result)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def load_masks(request, session_id):
    """
    Load previously saved masks from database
    """
    session = interactive_sessions.get(session_id)
    if not session:
        return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
    
    result = session.load_masks_from_database()
    
    return Response(result)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_session_status(request, session_id):
    """
    Get current session status and information
    """
    session = interactive_sessions.get(session_id)
    if not session:
        return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
    
    result = session.get_session_status()
    return Response(result)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def close_interactive_session(request, session_id):
    """
    Close interactive session and cleanup resources
    """
    session = interactive_sessions.get(session_id)
    if not session:
        return Response({'error': 'Session not found'}, status=status.HTTP_404_NOT_FOUND)
    
    try:
        # Exit segmentation mode if active
        if session.mode.value == 'segmenting':
            session.exit_segmentation_mode(save_masks=False)
        
        # Remove from global sessions
        del interactive_sessions[session_id]
        
        return Response({
            'session_id': session_id,
            'status': 'closed',
            'message': 'Interactive session closed successfully'
        })
        
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class AnalysisSessionViewSet(viewsets.ModelViewSet):
    """ViewSet for managing analysis sessions"""
    serializer_class = AnalysisSessionSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter by current user and optionally by building project"""
        queryset = AnalysisSession.objects.filter(user=self.request.user)
        
        # Optional filtering by building project
        building_project = self.request.query_params.get('building_project')
        if building_project:
            queryset = queryset.filter(building_project_id=building_project)
            
        # Optional filtering by status
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
            
        return queryset

    def perform_create(self, serializer):
        """Automatically set the user and trigger SAM 2 initialization"""
        analysis_session = serializer.save(user=self.request.user)
        
        # TODO: Start Celery task for SAM 2 initialization
        # task = initialize_video_analysis.delay(
        #     analysis_session_id=str(analysis_session.id),
        #     video_file_path=analysis_session.video_file.file_path,
        #     sam2_model=analysis_session.sam2_model
        # )


class SegmentationResultViewSet(viewsets.ModelViewSet):
    """ViewSet for managing segmentation results"""
    serializer_class = SegmentationResultSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter by results belonging to sessions owned by current user"""
        queryset = SegmentationResult.objects.filter(
            analysis_session__user=self.request.user
        )
        
        # Optional filtering by analysis session
        analysis_session = self.request.query_params.get('analysis_session')
        if analysis_session:
            queryset = queryset.filter(analysis_session_id=analysis_session)
            
        # Optional filtering by frame
        frame_index = self.request.query_params.get('frame_index')
        if frame_index:
            queryset = queryset.filter(frame_index=frame_index)
            
        return queryset


class ClickPromptViewSet(viewsets.ModelViewSet):
    """ViewSet for managing click prompts"""
    serializer_class = ClickPromptSerializer
    permission_classes = [IsAuthenticated]
    
    def get_queryset(self):
        """Filter by prompts belonging to sessions owned by current user"""
        return ClickPrompt.objects.filter(
            analysis_session__user=self.request.user
        )

    def perform_create(self, serializer):
        """Set the user when creating a prompt"""
        serializer.save(user=self.request.user)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def start_analysis_session(request):
    """Start a new SAM 2 analysis session"""
    serializer = AnalysisSessionSerializer(data=request.data)
    
    if serializer.is_valid():
        # Set the user
        analysis_session = serializer.save(user=request.user)
        
        # Get recommendations for the response
        model_info = analysis_session.get_model_info()
        recommended_model = analysis_session.get_recommended_model()
        memory_estimate = analysis_session.get_memory_estimate()
        
        # Prepare response with recommendations
        response_data = AnalysisSessionSerializer(analysis_session).data
        
        # Add recommendations if different from selected
        if recommended_model != analysis_session.sam2_model:
            response_data['recommendation'] = {
                'suggested_model': recommended_model,
                'reason': f"Recommended for your video size and resolution",
                'current_memory_estimate': f"{memory_estimate:.1f}GB"
            }
        
        # TODO: Start Celery task with user-selected model
        # task = initialize_video_analysis.delay(
        #     analysis_session_id=str(analysis_session.id),
        #     video_file_path=analysis_session.video_file.file_path,
        #     sam2_model=analysis_session.sam2_model
        # )
        # response_data['task_id'] = task.id
        
        return Response(response_data, status=status.HTTP_201_CREATED)
    
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def add_segmentation_points(request, session_id):
    """Add segmentation points to a specific frame"""
    # Get the analysis session
    analysis_session = get_object_or_404(
        AnalysisSession.objects.filter(user=request.user), 
        id=session_id
    )
    
    # Validate required fields
    required_fields = ['frame_index', 'object_id', 'points', 'labels']
    for field in required_fields:
        if field not in request.data:
            return Response(
                {'error': f'Missing required field: {field}'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
    
    # Create click prompts for tracking
    points = request.data['points']
    labels = request.data['labels']
    frame_index = request.data['frame_index']
    object_id = request.data['object_id']
    
    for point, label in zip(points, labels):
        ClickPrompt.objects.create(
            analysis_session=analysis_session,
            frame_index=frame_index,
            object_id=object_id,
            prompt_type='positive' if label == 1 else 'negative',
            coordinates=point,
            user=request.user
        )
    
    # TODO: Call SAM 2 service to add points
    # task = add_segmentation_points.delay(
    #     sam2_session_id=analysis_session.sam2_session_id,
    #     frame_index=frame_index,
    #     object_id=object_id,
    #     points=points,
    #     labels=labels,
    #     clear_old_points=request.data.get('clear_old_points', False)
    # )
    
    return Response({
        'analysis_session_id': analysis_session.id,
        'frame_index': frame_index,
        'object_id': object_id,
        'points_added': len(points),
        'status': 'processing'
        # 'task_id': task.id
    }, status=status.HTTP_201_CREATED)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def propagate_segmentation(request, session_id):
    """Propagate segmentation across the entire video"""
    # Get the analysis session
    analysis_session = get_object_or_404(
        AnalysisSession.objects.filter(user=request.user), 
        id=session_id
    )
    
    # Validate session is ready for propagation
    if not analysis_session.sam2_session_id:
        return Response(
            {'error': 'Analysis session not initialized with SAM 2'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    if analysis_session.status != 'active':
        return Response(
            {'error': f'Session must be active to propagate. Current status: {analysis_session.status}'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Update status to processing
    analysis_session.status = 'processing'
    analysis_session.save()
    
    # Get start frame (default to 0)
    start_frame = request.data.get('start_frame_index', 0)
    
    # TODO: Start propagation task
    # task = propagate_segmentation.delay(
    #     sam2_session_id=analysis_session.sam2_session_id,
    #     start_frame_index=start_frame
    # )
    
    return Response({
        'analysis_session_id': analysis_session.id,
        'start_frame_index': start_frame,
        'status': 'processing',
        'message': 'Segmentation propagation started'
        # 'task_id': task.id
    }, status=status.HTTP_202_ACCEPTED)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_model_recommendations(request, session_id):
    """Get model recommendations for a specific analysis session"""
    analysis_session = get_object_or_404(
        AnalysisSession.objects.filter(user=request.user), 
        id=session_id
    )
    
    model_info = analysis_session.get_model_info()
    recommended_model = analysis_session.get_recommended_model()
    memory_estimate = analysis_session.get_memory_estimate()
    
    # Get info for all available models
    all_models = {}
    for model_choice in AnalysisSession.SAM2_MODEL_CHOICES:
        model_key = model_choice[0]
        temp_session = AnalysisSession(
            video_file=analysis_session.video_file, 
            sam2_model=model_key
        )
        all_models[model_key] = {
            'name': model_choice[1],
            'info': temp_session.get_model_info(),
            'memory_estimate': temp_session.get_memory_estimate(),
            'is_recommended': model_key == recommended_model
        }
    
    return Response({
        'current_model': analysis_session.sam2_model,
        'current_model_info': model_info,
        'current_memory_estimate': memory_estimate,
        'recommended_model': recommended_model,
        'all_models': all_models,
        'video_characteristics': {
            'is_4k': getattr(analysis_session.video_file, 'width', 0) >= 3840,
            'frame_count': getattr(analysis_session.video_file, 'frame_count', 0),
            'duration': getattr(analysis_session.video_file, 'duration', 0)
        }
    })


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_model_selection(request, session_id):
    """Update the SAM 2 model selection for an analysis session"""
    analysis_session = get_object_or_404(
        AnalysisSession.objects.filter(user=request.user), 
        id=session_id
    )
    
    # Only allow model changes if session is not active
    if analysis_session.status in ['processing', 'active']:
        return Response(
            {'error': 'Cannot change model while session is active or processing'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    new_model = request.data.get('sam2_model')
    if not new_model:
        return Response(
            {'error': 'sam2_model is required'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Validate model choice
    valid_models = [choice[0] for choice in AnalysisSession.SAM2_MODEL_CHOICES]
    if new_model not in valid_models:
        return Response(
            {'error': f'Invalid model. Choose from: {valid_models}'}, 
            status=status.HTTP_400_BAD_REQUEST
        )
    
    # Update the model
    old_model = analysis_session.sam2_model
    analysis_session.sam2_model = new_model
    analysis_session.status = 'initializing'  # Reset status
    analysis_session.sam2_session_id = ''  # Clear old session ID
    analysis_session.save()
    
    # Get new model info
    model_info = analysis_session.get_model_info()
    memory_estimate = analysis_session.get_memory_estimate()
    
    return Response({
        'analysis_session_id': analysis_session.id,
        'old_model': old_model,
        'new_model': new_model,
        'model_info': model_info,
        'memory_estimate': memory_estimate,
        'status': 'initializing',
        'message': f'Model updated from {old_model} to {new_model}'
    })


class StartAnalysisView(APIView):
    """Legacy view - use start_analysis_session function instead"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        return start_analysis_session(request)


class AddPointsView(APIView):
    """Legacy view - use add_segmentation_points function instead"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        session_id = request.data.get('session_id')
        if not session_id:
            return Response(
                {'error': 'session_id is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        return add_segmentation_points(request, session_id)


class PropagateView(APIView):
    """Legacy view - use propagate_segmentation function instead"""
    permission_classes = [IsAuthenticated]
    
    def post(self, request):
        session_id = request.data.get('session_id')
        if not session_id:
            return Response(
                {'error': 'session_id is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        return propagate_segmentation(request, session_id)