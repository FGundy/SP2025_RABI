import logging
import os
import tempfile
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

logger = logging.getLogger(__name__)

class DynamicVideoChunker:
    """
    Manages dynamic video chunking for memory-efficient SAM 2 processing
    Creates temporal chunks on-demand based on user interaction
    """
    
    def __init__(self, chunk_size: int = 200, overlap_frames: int = 10):
        """
        Initialize dynamic chunker
        
        Args:
            chunk_size: Number of frames per chunk (default 200)
            overlap_frames: Overlap between chunks for smooth transitions
        """
        self.chunk_size = chunk_size
        self.overlap_frames = overlap_frames
        self.chunk_cache = {}
        self.temp_dir = None
        
    def initialize_video(self, video_path: str, session_id: str) -> Dict[str, Any]:
        """Initialize video for chunked processing"""
        video_info = self._get_video_info(video_path)
        if not video_info:
            raise ValueError("Could not analyze video file")
        
        total_frames = video_info['frame_count']
        num_chunks = max(1, (total_frames + self.chunk_size - 1) // self.chunk_size)
        
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix=f"sam2_chunks_{session_id}_")
        
        strategy = {
            'session_id': session_id,
            'original_video': video_path,
            'total_frames': total_frames,
            'duration': video_info['duration'],
            'chunk_size': self.chunk_size,
            'overlap_frames': self.overlap_frames,
            'num_chunks': num_chunks,
            'chunks_directory': self.temp_dir,
            'video_info': video_info,
            'chunk_frame_ranges': self._calculate_chunk_ranges(total_frames)
        }
        
        logger.info(f"🎬 Initialized chunking: {num_chunks} chunks of {self.chunk_size} frames")
        return strategy
    
    def get_chunk_for_frame(self, frame_index: int, strategy: Dict[str, Any], sam2_api=None) -> Tuple[str, int, str]:
        """Get or create chunk containing the specified frame"""
        chunk_info = self._find_chunk_for_frame(frame_index, strategy)
        if not chunk_info:
            raise ValueError(f"Frame {frame_index} is outside video bounds")
        
        chunk_id, start_frame, end_frame = chunk_info
        chunk_key = f"{strategy['session_id']}_chunk_{chunk_id}"
        
        if chunk_key in self.chunk_cache:
            chunk_path = self.chunk_cache[chunk_key]['path']
            if os.path.exists(chunk_path):
                local_frame_index = frame_index - start_frame
                sam2_session_id = self.chunk_cache[chunk_key].get('sam2_session_id')
                logger.info(f"♻️  Using cached chunk {chunk_id} for frame {frame_index}")
                return chunk_path, local_frame_index, sam2_session_id
        
        logger.info(f"🔧 Creating chunk {chunk_id} for frame {frame_index} (frames {start_frame}-{end_frame})")
        
        chunk_path = self._create_chunk(
            strategy['original_video'],
            start_frame,
            end_frame,
            chunk_id,
            strategy['chunks_directory'],
            strategy['video_info']
        )
        
        sam2_session_id = None
        self.chunk_cache[chunk_key] = {
            'path': chunk_path,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'sam2_session_id': sam2_session_id
        }
        
        local_frame_index = frame_index - start_frame
        return chunk_path, local_frame_index, sam2_session_id
    
    def _calculate_chunk_ranges(self, total_frames: int) -> List[Tuple[int, int]]:
        """Calculate frame ranges for each chunk"""
        ranges = []
        for i in range(0, total_frames, self.chunk_size):
            start_frame = max(0, i - self.overlap_frames if i > 0 else 0)
            end_frame = min(total_frames - 1, i + self.chunk_size - 1)
            ranges.append((start_frame, end_frame))
        return ranges
    
    def _find_chunk_for_frame(self, frame_index: int, strategy: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
        """Find which chunk contains the given frame"""
        chunk_ranges = strategy['chunk_frame_ranges']
        for chunk_id, (start_frame, end_frame) in enumerate(chunk_ranges):
            if start_frame <= frame_index <= end_frame:
                return (chunk_id, start_frame, end_frame)
        return None
    
    def _create_chunk(self, video_path: str, start_frame: int, end_frame: int, 
                     chunk_id: int, output_dir: str, video_info: Dict[str, Any]) -> str:
        """Create a video chunk file"""
        fps = video_info.get('fps', 30)
        start_time = start_frame / fps
        duration = (end_frame - start_frame + 1) / fps
        
        output_path = os.path.join(output_dir, f"chunk_{chunk_id:03d}_{start_frame}_{end_frame}.mp4")
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            '-y',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create chunk {chunk_id}: {result.stderr}")
        
        return output_path
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information using ffprobe"""
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
                
                if video_stream:
                    fps_str = video_stream.get('r_frame_rate', '30/1')
                    fps = eval(fps_str) if '/' in fps_str else float(fps_str)
                    
                    return {
                        'width': int(video_stream.get('width', 0)),
                        'height': int(video_stream.get('height', 0)),
                        'fps': fps,
                        'duration': float(data['format']['duration']),
                        'frame_count': int(float(data['format']['duration']) * fps),
                        'codec': video_stream.get('codec_name'),
                        'is_4k': int(video_stream.get('width', 0)) >= 3840,
                    }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
        
        return {}
    
    def cleanup_session(self, session_id: str):
        """Clean up all chunks for a session"""
        chunks_to_remove = [k for k in self.chunk_cache.keys() if session_id in k]
        
        for chunk_key in chunks_to_remove:
            chunk_info = self.chunk_cache[chunk_key]
            
            try:
                chunk_path = chunk_info['path']
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            except Exception as e:
                logger.warning(f"Failed to remove chunk file: {e}")
            
            del self.chunk_cache[chunk_key]
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
                self.temp_dir = None
            except OSError:
                pass
        
        logger.info(f"🧹 Cleaned up {len(chunks_to_remove)} chunks for session {session_id}")
