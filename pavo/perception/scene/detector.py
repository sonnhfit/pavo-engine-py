"""
Scene detection module using PySceneDetect.

This module provides SceneDetector class for detecting scene changes in videos
using PySceneDetect library. It can detect scene boundaries and split videos
into individual scenes based on content changes.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import json


class SceneDetector:
    """
    A class for detecting scene changes in videos using PySceneDetect.
    
    This class handles scene detection with support for various detection algorithms
    and provides options for threshold adjustment, output format, and video splitting.
    
    Attributes:
        detector_type (str): The scene detection algorithm to use ('adaptive', 'content', 'threshold')
        threshold (float): Threshold value for scene detection (algorithm-dependent)
        min_scene_len (int): Minimum length of a scene in frames
        output_dir (Optional[str]): Directory to save split scenes
        stats_file (Optional[str]): Path to save detection statistics
    
    Example:
        >>> detector = SceneDetector(detector_type='adaptive', threshold=30.0)
        >>> scenes = detector.detect('video.mp4')
        >>> print(f"Found {len(scenes)} scenes")
        >>> for scene in scenes:
        >>>     print(f"Scene: {scene['start']} to {scene['end']}")
    """
    
    def __init__(
        self,
        detector_type: str = 'adaptive',
        threshold: float = 30.0,
        min_scene_len: int = 15,
        output_dir: Optional[str] = None,
        stats_file: Optional[str] = None
    ):
        """
        Initialize the SceneDetector.
        
        Args:
            detector_type (str): Scene detection algorithm. Options:
                                - 'adaptive': AdaptiveDetector (default)
                                - 'content': ContentDetector  
                                - 'threshold': ThresholdDetector
                                Default is 'adaptive'.
            threshold (float): Threshold value for scene detection.
                              For AdaptiveDetector: 30.0 (default)
                              For ContentDetector: 27.0-30.0
                              For ThresholdDetector: 12-20
            min_scene_len (int): Minimum scene length in frames. Default is 15.
            output_dir (Optional[str]): Directory to save split scenes. If None, scenes won't be split.
            stats_file (Optional[str]): Path to save detection statistics CSV file.
        
        Raises:
            ImportError: If scenedetect package is not installed.
            ValueError: If detector_type is invalid.
        """
        try:
            from scenedetect import detect, AdaptiveDetector, ContentDetector, ThresholdDetector
            from scenedetect import split_video_ffmpeg
            self.scenedetect = {
                'detect': detect,
                'AdaptiveDetector': AdaptiveDetector,
                'ContentDetector': ContentDetector,
                'ThresholdDetector': ThresholdDetector,
                'split_video_ffmpeg': split_video_ffmpeg
            }
        except ImportError:
            raise ImportError(
                "PySceneDetect is not installed. Please install it using:\n"
                "pip install scenedetect[opencv]"
            )
        
        # Validate detector type
        valid_detectors = ['adaptive', 'content', 'threshold']
        if detector_type not in valid_detectors:
            raise ValueError(
                f"Invalid detector_type: {detector_type}. "
                f"Must be one of: {', '.join(valid_detectors)}"
            )
        
        self.detector_type = detector_type
        self.threshold = threshold
        self.min_scene_len = min_scene_len
        self.output_dir = output_dir
        self.stats_file = stats_file
        
        # Create output directory if specified
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    def _get_detector(self):
        """Get the configured scene detector instance."""
        if self.detector_type == 'adaptive':
            return self.scenedetect['AdaptiveDetector'](
                adaptive_threshold=self.threshold,
                min_scene_len=self.min_scene_len
            )
        elif self.detector_type == 'content':
            return self.scenedetect['ContentDetector'](
                threshold=self.threshold,
                min_scene_len=self.min_scene_len
            )
        elif self.detector_type == 'threshold':
            return self.scenedetect['ThresholdDetector'](
                threshold=self.threshold,
                min_scene_len=self.min_scene_len
            )
    
    def detect(
        self,
        video_path: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        show_progress: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect scene changes in a video file.
        
        Args:
            video_path (str): Path to the video file.
            start_time (Optional[float]): Start time in seconds. If None, start from beginning.
            end_time (Optional[float]): End time in seconds. If None, process entire video.
            show_progress (bool): If True, show progress bar during detection.
        
        Returns:
            List[Dict[str, Any]]: List of scene dictionaries, each containing:
                - 'scene_number' (int): Scene index (1-based)
                - 'start' (float): Start time in seconds
                - 'end' (float): End time in seconds
                - 'duration' (float): Scene duration in seconds
                - 'start_frame' (int): Start frame number
                - 'end_frame' (int): End frame number
                - 'frame_count' (int): Number of frames in scene
        
        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If video format is not supported.
            RuntimeError: If scene detection fails.
        
        Example:
            >>> detector = SceneDetector()
            >>> scenes = detector.detect('video.mp4')
            >>> for scene in scenes:
            >>>     print(f"Scene {scene['scene_number']}: {scene['start']:.2f}s to {scene['end']:.2f}s")
        """
        # Validate video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Validate supported video format
        supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
        file_ext = Path(video_path).suffix.lower()
        if file_ext not in supported_formats:
            raise ValueError(
                f"Unsupported video format: {file_ext}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
        
        try:
            # Get detector instance
            detector = self._get_detector()
            
            # Detect scenes
            scene_list = self.scenedetect['detect'](
                video_path,
                detector,
                start_time=start_time,
                end_time=end_time,
                show_progress=show_progress
            )
            
            # Convert scene list to dictionary format
            scenes = []
            for i, scene in enumerate(scene_list):
                start_time_sec = scene[0].get_seconds()
                end_time_sec = scene[1].get_seconds()
                duration = end_time_sec - start_time_sec
                
                # Get frame numbers if available
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames()
                frame_count = end_frame - start_frame + 1 if start_frame and end_frame else 0
                
                scene_dict = {
                    'scene_number': i + 1,
                    'start': start_time_sec,
                    'end': end_time_sec,
                    'duration': duration,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'frame_count': frame_count,
                    'start_timecode': str(scene[0]),
                    'end_timecode': str(scene[1])
                }
                scenes.append(scene_dict)
            
            return scenes
        
        except Exception as e:
            raise RuntimeError(f"Scene detection failed: {str(e)}") from e
    
    def detect_and_split(
        self,
        video_path: str,
        output_pattern: Optional[str] = None,
        show_progress: bool = False
    ) -> Dict[str, Any]:
        """
        Detect scenes and split video into individual scene files.
        
        Args:
            video_path (str): Path to the video file.
            output_pattern (Optional[str]): Pattern for output filenames.
                                          Use {scene_number} for scene number.
                                          Default: 'scene_{scene_number}.mp4'
            show_progress (bool): If True, show progress bar during detection and splitting.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'scenes' (list): List of scene dictionaries (same as detect() output)
                - 'output_files' (list): List of paths to split scene files
                - 'output_dir' (str): Directory where scenes were saved
                - 'video_path' (str): Original video path
                - 'num_scenes' (int): Total number of scenes detected
        
        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If video format is not supported or output_dir not set.
            RuntimeError: If scene detection or splitting fails.
        
        Example:
            >>> detector = SceneDetector(output_dir='scenes')
            >>> result = detector.detect_and_split('video.mp4')
            >>> print(f"Split {result['num_scenes']} scenes to {result['output_dir']}")
        """
        if not self.output_dir:
            raise ValueError(
                "output_dir must be set in constructor to use detect_and_split(). "
                "Initialize SceneDetector with output_dir parameter."
            )
        
        # Detect scenes
        scenes = self.detect(video_path, show_progress=show_progress)
        
        # Set default output pattern
        if output_pattern is None:
            output_pattern = 'scene_{scene_number}.mp4'
        
        try:
            # Get detector instance
            detector = self._get_detector()
            
            # Detect scenes again for splitting (scenedetect requires fresh detection)
            scene_list = self.scenedetect['detect'](
                video_path,
                detector,
                show_progress=show_progress
            )
            
            # Split video into scenes
            output_files = self.scenedetect['split_video_ffmpeg'](
                video_path,
                scene_list,
                output_dir=self.output_dir,
                output_file_template=output_pattern,
                show_progress=show_progress
            )
            
            # Map output files to scenes
            for i, scene in enumerate(scenes):
                if i < len(output_files):
                    scene['output_file'] = output_files[i]
            
            return {
                'scenes': scenes,
                'output_files': output_files,
                'output_dir': self.output_dir,
                'video_path': video_path,
                'num_scenes': len(scenes)
            }
        
        except Exception as e:
            raise RuntimeError(f"Video splitting failed: {str(e)}") from e
    
    def analyze_scene_statistics(
        self,
        scenes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze scene statistics from detection results.
        
        Args:
            scenes (List[Dict[str, Any]]): List of scene dictionaries from detect().
        
        Returns:
            Dict[str, Any]: Dictionary containing scene statistics:
                - 'num_scenes' (int): Total number of scenes
                - 'total_duration' (float): Total duration of all scenes in seconds
                - 'avg_duration' (float): Average scene duration
                - 'min_duration' (float): Minimum scene duration
                - 'max_duration' (float): Maximum scene duration
                - 'duration_std' (float): Standard deviation of scene durations
                - 'short_scenes' (list): Scenes shorter than 2 seconds
                - 'long_scenes' (list): Scenes longer than 10 seconds
                - 'scene_distribution' (dict): Distribution of scene lengths by category
        
        Example:
            >>> scenes = detector.detect('video.mp4')
            >>> stats = detector.analyze_scene_statistics(scenes)
            >>> print(f"Average scene length: {stats['avg_duration']:.2f}s")
        """
        if not scenes:
            return {
                'num_scenes': 0,
                'total_duration': 0.0,
                'avg_duration': 0.0,
                'min_duration': 0.0,
                'max_duration': 0.0,
                'duration_std': 0.0,
                'short_scenes': [],
                'long_scenes': [],
                'scene_distribution': {}
            }
        
        # Extract durations
        durations = [scene['duration'] for scene in scenes]
        total_duration = sum(durations)
        avg_duration = total_duration / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        # Calculate standard deviation
        import math
        variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)
        duration_std = math.sqrt(variance)
        
        # Categorize scenes
        short_scenes = [s for s in scenes if s['duration'] < 2.0]
        long_scenes = [s for s in scenes if s['duration'] > 10.0]
        
        # Scene distribution by duration categories
        categories = {
            'very_short': (0, 2),
            'short': (2, 5),
            'medium': (5, 10),
            'long': (10, 20),
            'very_long': (20, float('inf'))
        }
        
        scene_distribution = {}
        for cat_name, (low, high) in categories.items():
            if high == float('inf'):
                scene_distribution[cat_name] = [
                    s for s in scenes if s['duration'] >= low
                ]
            else:
                scene_distribution[cat_name] = [
                    s for s in scenes if low <= s['duration'] < high
                ]
        
        return {
            'num_scenes': len(scenes),
            'total_duration': total_duration,
            'avg_duration': avg_duration,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'duration_std': duration_std,
            'short_scenes': short_scenes,
            'long_scenes': long_scenes,
            'scene_distribution': {
                cat: len(scenes) for cat, scenes in scene_distribution.items()
            }
        }
    
    def save_scenes_to_json(
        self,
        scenes: List[Dict[str, Any]],
        output_path: str
    ):
        """
        Save scene detection results to a JSON file.
        
        Args:
            scenes (List[Dict[str, Any]]): Scene detection results from detect().
            output_path (str): Path to save JSON file.
        
        Raises:
            ValueError: If output_path is not a .json file.
            RuntimeError: If saving fails.
        """
        if not output_path.lower().endswith('.json'):
            raise ValueError("Output path must be a .json file")
        
        try:
            # Convert to serializable format
            serializable_scenes = []
            for scene in scenes:
                serializable_scene = scene.copy()
                # Ensure all values are JSON serializable
                for key, value in serializable_scene.items():
                    if isinstance(value, (int, float)):
                        serializable_scene[key] = float(value)
                serializable_scenes.append(serializable_scene)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_scenes, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            raise RuntimeError(f"Failed to save scenes to JSON: {str(e)}") from e
    
    def save_statistics_to_csv(
        self,
        scenes: List[Dict[str, Any]],
        output_path: Optional[str] = None
    ):
        """
        Save scene statistics to a CSV file.
        
        Args:
            scenes (List[Dict[str, Any]]): Scene detection results from detect().
            output_path (Optional[str]): Path to save CSV file. If None, uses stats_file from constructor.
        
        Raises:
            ValueError: If no output path is provided.
            RuntimeError: If saving fails.
        """
        if output_path is None:
            output_path = self.stats_file
        
        if not output_path:
            raise ValueError("No output path provided for CSV file")
        
        try:
            import csv
            
            # Prepare CSV data
            headers = [
                'scene_number', 'start_time', 'end_time', 'duration',
                'start_frame', 'end_frame', 'frame_count',
                'start_timecode', 'end_timecode'
            ]
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                
                for scene in scenes:
                    row = {
                        'scene_number': scene['scene_number'],
                        'start_time': f"{scene['start']:.3f}",
                        'end_time': f"{scene['end']:.3f}",
                        'duration': f"{scene['duration']:.3f}",
                        'start_frame': scene['start_frame'],
                        'end_frame': scene['end_frame'],
                        'frame_count': scene['frame_count'],
                        'start_timecode': scene['start_timecode'],
                        'end_timecode': scene['end_timecode']
                    }
                    writer.writerow(row)
        
        except Exception as e:
            raise RuntimeError(f"Failed to save statistics to CSV: {str(e)}") from e
    
    def generate_scene_report(
        self,
        video_path: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive scene analysis report.
        
        Args:
            video_path (str): Path to the video file.
            output_dir (Optional[str]): Directory to save report files. If None, uses output_dir from constructor.
        
        Returns:
            Dict[str, Any]: Dictionary containing report information:
                - 'scenes' (list): Scene detection results
                - 'statistics' (dict): Scene statistics
                - 'json_path' (str): Path to saved JSON file
                - 'csv_path' (str): Path to saved CSV file
                - 'output_dir' (str): Directory where report was saved
        
        Raises:
            FileNotFoundError: If video file doesn't exist.
            RuntimeError: If report generation fails.
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        if not output_dir:
            raise ValueError(
                "output_dir must be set to generate scene report. "
                "Either pass output_dir parameter or set it in constructor."
            )
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Detect scenes
        scenes = self.detect(video_path, show_progress=True)
        
        # Analyze statistics
        statistics = self.analyze_scene_statistics(scenes)
        
        # Generate base filename from video name
        video_name = Path(video_path).stem
        base_path = os.path.join(output_dir, f"{video_name}_scenes")
        
        # Save JSON file
        json_path = f"{base_path}.json"
        self.save_scenes_to_json(scenes, json_path)
        
        # Save CSV file
        csv_path = f"{base_path}.csv"
        self.save_statistics_to_csv(scenes, csv_path)
        
        return {
            'scenes': scenes,
            'statistics': statistics,
            'json_path': json_path,
            'csv_path': csv_path,
            'output_dir': output_dir
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False
        
