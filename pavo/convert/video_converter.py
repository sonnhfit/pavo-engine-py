"""
Video converter module for converting various video formats to MP4.

This module provides the VideoConverter class that can convert video files
from common video formats to MP4 format using FFmpeg.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import ffmpeg
from tqdm import tqdm


class VideoConverter:
    """
    A class for converting video files to MP4 format.
    
    This class handles video format conversion from common video formats
    (AVI, MOV, MKV, FLV, WMV, etc.) to MP4 format using FFmpeg.
    
    Attributes:
        codec (str): Video codec to use (default: 'libx264')
        audio_codec (str): Audio codec to use (default: 'aac')
        preset (str): FFmpeg preset for encoding speed vs quality
                     Options: 'ultrafast', 'superfast', 'veryfast', 'faster', 
                             'fast', 'medium', 'slow', 'slower', 'veryslow'
                     Default: 'medium'
    
    Example:
        >>> converter = VideoConverter()
        >>> converter.convert('input.avi', 'output.mp4')
        >>> # Or with custom codec settings
        >>> converter = VideoConverter(codec='libx264', preset='fast')
        >>> converter.convert('input.mov', 'output.mp4', bitrate='8000k')
    """
    
    # Supported input video formats
    SUPPORTED_FORMATS = {
        '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v',
        '.mpg', '.mpeg', '.3gp', '.3g2', '.mxf', '.ogv', '.ts',
        '.m2ts', '.mts', '.vob', '.f4v', '.asf', '.rm', '.rmvb',
        '.mp4'  # MP4 can also be converted/re-encoded
    }
    
    def __init__(
        self,
        codec: str = 'libx264',
        audio_codec: str = 'aac',
        preset: str = 'medium',
        crf: int = 23
    ):
        """
        Initialize the VideoConverter.
        
        Args:
            codec (str): Video codec to use. Default is 'libx264' (H.264).
                        Other options: 'libx265' (H.265/HEVC), 'mpeg4', 'libvpx-vp9'
            audio_codec (str): Audio codec to use. Default is 'aac'.
                              Other options: 'libmp3lame' (MP3), 'libvorbis', 'flac'
            preset (str): FFmpeg preset for encoding speed vs quality.
                         Default is 'medium'.
                         Fast options: 'ultrafast', 'superfast', 'veryfast'
                         Balanced: 'faster', 'fast', 'medium'
                         Quality: 'slow', 'slower', 'veryslow'
            crf (int): Constant Rate Factor (0-51). Lower = better quality, higher = smaller file.
                      Default is 23 (medium quality).
        
        Raises:
            ImportError: If ffmpeg-python or ffmpeg is not installed.
        """
        try:
            # Check if FFmpeg is installed
            ffmpeg.input('/dev/null').output('/dev/null').compile()
        except Exception:
            raise ImportError(
                "FFmpeg is not installed or not accessible. "
                "Please install FFmpeg:\n"
                "  macOS: brew install ffmpeg\n"
                "  Ubuntu: sudo apt-get install ffmpeg\n"
                "  Windows: choco install ffmpeg"
            )
        
        self.codec = codec
        self.audio_codec = audio_codec
        self.preset = preset
        self.crf = crf
    
    def convert(
        self,
        input_path: str,
        output_path: str,
        bitrate: Optional[str] = None,
        audio_bitrate: Optional[str] = None,
        fps: Optional[int] = None,
        scale: Optional[str] = None,
        verbose: bool = True
    ) -> bool:
        """
        Convert a video file to MP4 format.
        
        Args:
            input_path (str): Path to the input video file.
            output_path (str): Path to the output MP4 file.
            bitrate (Optional[str]): Video bitrate (e.g., '8000k', '5M').
                                    If None, uses CRF for quality control.
            audio_bitrate (Optional[str]): Audio bitrate (e.g., '128k', '192k').
                                          Default is auto-detected.
            fps (Optional[int]): Target frames per second. If None, keeps original.
            scale (Optional[str]): Video scale (e.g., '1280:720', '1920:1080').
                                  If None, keeps original resolution.
            verbose (bool): If True, print progress information. Default is True.
        
        Returns:
            bool: True if conversion was successful, False otherwise.
        
        Raises:
            FileNotFoundError: If the input file doesn't exist.
            ValueError: If the input file format is not supported.
            RuntimeError: If conversion fails.
        
        Example:
            >>> converter = VideoConverter()
            >>> success = converter.convert('input.avi', 'output.mp4', fps=30, scale='1280:720')
            >>> if success:
            ...     print("Conversion completed successfully")
        """
        # Validate input file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
        
        # Check file format
        input_ext = Path(input_path).suffix.lower()
        if input_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported video format: {input_ext}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
        
        # Ensure output path is MP4
        output_ext = Path(output_path).suffix.lower()
        if output_ext != '.mp4':
            output_path = str(Path(output_path).with_suffix('.mp4'))
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            if verbose:
                print(f"Converting video: {input_path} -> {output_path}")
            
            # Load input stream
            stream = ffmpeg.input(input_path)
            
            # Build output stream
            v = stream.video
            a = stream.audio
            
            # Apply video filters
            if scale:
                v = v.filter('scale', scale)
            if fps:
                v = v.filter('fps', fps)
            
            # Prepare output kwargs
            output_kwargs = {
                'c:v': self.codec,
                'preset': self.preset,
                'c:a': self.audio_codec,
            }
            
            if bitrate:
                output_kwargs['b:v'] = bitrate
            else:
                output_kwargs['crf'] = str(self.crf)
            
            if audio_bitrate:
                output_kwargs['b:a'] = audio_bitrate
            
            # Create output
            out = ffmpeg.output(v, a, output_path, **output_kwargs)
            
            # Run conversion
            ffmpeg.run(out, quiet=not verbose, overwrite_output=True)
            
            if verbose:
                print(f"✓ Successfully converted to: {output_path}")
            
            return True
        
        except ffmpeg.Error as e:
            error_msg = f"FFmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}"
            if verbose:
                print(f"✗ {error_msg}")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Conversion failed: {str(e)}"
            if verbose:
                print(f"✗ {error_msg}")
            raise RuntimeError(error_msg) from e
    
    def get_video_info(self, input_path: str) -> Dict[str, Any]:
        """
        Get information about a video file.
        
        Args:
            input_path (str): Path to the video file.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'duration' (float): Duration in seconds
                - 'width' (int): Video width in pixels
                - 'height' (int): Video height in pixels
                - 'fps' (float): Frames per second
                - 'codec' (str): Video codec name
                - 'bitrate' (int): Bitrate in bits per second
        
        Raises:
            FileNotFoundError: If the input file doesn't exist.
            RuntimeError: If unable to read video information.
        
        Example:
            >>> converter = VideoConverter()
            >>> info = converter.get_video_info('input.avi')
            >>> print(f"Resolution: {info['width']}x{info['height']}")
            >>> print(f"Duration: {info['duration']:.2f}s")
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Video file not found: {input_path}")
        
        try:
            probe = ffmpeg.probe(input_path)
            video_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise RuntimeError("No video stream found in the file")
            
            # Extract information
            duration = float(probe.get('format', {}).get('duration', 0))
            width = video_stream.get('width', 0)
            height = video_stream.get('height', 0)
            r_frame_rate = video_stream.get('r_frame_rate', '0/1')
            
            # Parse frame rate (e.g., "30000/1001")
            if '/' in str(r_frame_rate):
                num, den = map(float, str(r_frame_rate).split('/'))
                fps = num / den if den != 0 else 0
            else:
                fps = float(r_frame_rate)
            
            codec = video_stream.get('codec_name', 'unknown')
            bitrate = int(video_stream.get('bit_rate', probe.get('format', {}).get('bit_rate', 0)))
            
            return {
                'duration': duration,
                'width': width,
                'height': height,
                'fps': fps,
                'codec': codec,
                'bitrate': bitrate,
            }
        
        except Exception as e:
            raise RuntimeError(f"Failed to get video information: {str(e)}") from e
    
    def batch_convert(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Convert all video files in a directory to MP4.
        
        Args:
            input_dir (str): Directory containing input video files.
            output_dir (str): Directory to save converted MP4 files.
            recursive (bool): If True, search subdirectories recursively.
            verbose (bool): If True, print progress information.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'total' (int): Total number of files processed
                - 'successful' (int): Number of successful conversions
                - 'failed' (int): Number of failed conversions
                - 'errors' (list): List of (filename, error_message) tuples
        
        Example:
            >>> converter = VideoConverter()
            >>> result = converter.batch_convert('/path/to/videos', '/path/to/output')
            >>> print(f"Converted: {result['successful']}/{result['total']}")
        """
        if not os.path.isdir(input_dir):
            raise ValueError(f"Input directory not found: {input_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all video files
        video_files = []
        if recursive:
            for ext in self.SUPPORTED_FORMATS:
                pattern = f"**/*{ext}"
                video_files.extend(Path(input_dir).glob(pattern))
        else:
            for ext in self.SUPPORTED_FORMATS:
                pattern = f"*{ext}"
                video_files.extend(Path(input_dir).glob(pattern))
        
        results = {
            'total': len(video_files),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        if not video_files:
            if verbose:
                print(f"No video files found in {input_dir}")
            return results
        
        if verbose:
            print(f"Found {len(video_files)} video file(s) to convert")
        
        # Process each file
        for input_file in tqdm(video_files, disable=not verbose, desc="Converting videos"):
            try:
                # Create output path preserving directory structure if recursive
                if recursive:
                    rel_path = input_file.relative_to(input_dir)
                    rel_dir = rel_path.parent
                    output_subdir = os.path.join(output_dir, rel_dir)
                else:
                    output_subdir = output_dir
                
                os.makedirs(output_subdir, exist_ok=True)
                
                # Create output filename
                output_file = os.path.join(
                    output_subdir,
                    input_file.stem + '.mp4'
                )
                
                # Convert
                self.convert(str(input_file), output_file, verbose=False)
                results['successful'] += 1
            
            except Exception as e:
                results['failed'] += 1
                results['errors'].append((input_file.name, str(e)))
                if verbose:
                    print(f"  ✗ Failed to convert {input_file.name}: {str(e)}")
        
        return results
