"""
Audio converter module for converting various audio formats to MP3 and WAV.

This module provides the AudioConverter class that can convert audio files
from common audio formats to MP3 or WAV format using FFmpeg.
"""

import os
from pathlib import Path
from typing import Optional, Literal, Dict, Any
import ffmpeg
from tqdm import tqdm


class AudioConverter:
    """
    A class for converting audio files to MP3 or WAV format.
    
    This class handles audio format conversion from common audio formats
    (MP3, WAV, FLAC, OGG, M4A, etc.) to MP3 or WAV format using FFmpeg.
    
    Attributes:
        output_format (str): Output format - 'mp3' or 'wav'. Default: 'mp3'
        bitrate (str): Audio bitrate (e.g., '128k', '192k', '320k' for MP3)
                      Not used for WAV format.
    
    Example:
        >>> converter = AudioConverter(output_format='mp3', bitrate='192k')
        >>> converter.convert('input.flac', 'output.mp3')
        >>> # Or convert to WAV
        >>> converter_wav = AudioConverter(output_format='wav')
        >>> converter_wav.convert('input.aac', 'output.wav')
    """
    
    # Supported input audio formats
    SUPPORTED_FORMATS = {
        '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.ape',
        '.alac', '.opus', '.webm', '.aiff', '.au', '.raw', '.vorbis',
        '.amr', '.awb', '.gsm', '.ac3', '.eac3'
    }
    
    def __init__(
        self,
        output_format: Literal['mp3', 'wav'] = 'mp3',
        bitrate: str = '192k'
    ):
        """
        Initialize the AudioConverter.
        
        Args:
            output_format (Literal['mp3', 'wav']): Output format.
                                                  Default is 'mp3'.
                                                  Options: 'mp3', 'wav'
            bitrate (str): Audio bitrate for MP3 format.
                          Default is '192k'.
                          Common options: '128k', '192k', '256k', '320k'
                          Not used for WAV format.
        
        Raises:
            ImportError: If ffmpeg-python or ffmpeg is not installed.
            ValueError: If output_format is not supported.
        """
        if output_format.lower() not in ['mp3', 'wav']:
            raise ValueError(
                f"Unsupported output format: {output_format}. "
                "Supported formats: 'mp3', 'wav'"
            )
        
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
        
        self.output_format = output_format.lower()
        self.bitrate = bitrate
    
    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        verbose: bool = True
    ) -> str:
        """
        Convert an audio file to MP3 or WAV format.
        
        Args:
            input_path (str): Path to the input audio file.
            output_path (Optional[str]): Path to the output audio file.
                                        If None, creates output in same directory
                                        as input with changed extension.
            sample_rate (Optional[int]): Target sample rate in Hz (e.g., 44100, 48000).
                                        If None, keeps original sample rate.
            channels (Optional[int]): Number of audio channels (1 for mono, 2 for stereo).
                                     If None, keeps original channels.
            verbose (bool): If True, print progress information. Default is True.
        
        Returns:
            str: Path to the converted audio file.
        
        Raises:
            FileNotFoundError: If the input file doesn't exist.
            ValueError: If the input file format is not supported.
            RuntimeError: If conversion fails.
        
        Example:
            >>> converter = AudioConverter(output_format='mp3', bitrate='192k')
            >>> result = converter.convert('input.flac', 'output.mp3')
            >>> print(f"Converted to: {result}")
        """
        # Validate input file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input audio file not found: {input_path}")
        
        # Check file format
        input_ext = Path(input_path).suffix.lower()
        if input_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {input_ext}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
            )
        
        # Determine output path
        if output_path is None:
            input_file = Path(input_path)
            output_path = str(input_file.with_suffix(f'.{self.output_format}'))
        
        # Ensure output extension matches format
        output_ext = Path(output_path).suffix.lower()
        if output_ext != f'.{self.output_format}':
            output_path = str(Path(output_path).with_suffix(f'.{self.output_format}'))
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            if verbose:
                print(f"Converting audio: {input_path} -> {output_path}")
            
            # Load input
            stream = ffmpeg.input(input_path)
            
            # Prepare output options
            output_kwargs = {}
            
            if self.output_format == 'mp3':
                output_kwargs['c:a'] = 'libmp3lame'
                output_kwargs['b:a'] = self.bitrate
                output_kwargs['q:a'] = '4'  # VBR quality
            elif self.output_format == 'wav':
                output_kwargs['c:a'] = 'pcm_s16le'
                output_kwargs['acodec'] = 'pcm_s16le'
            
            # Apply sample rate conversion if specified
            if sample_rate:
                stream = stream.audio.filter('aresample', sample_rate)
            else:
                stream = stream.audio
            
            # Apply channel conversion if specified
            if channels:
                if channels == 1:
                    stream = stream.filter('mono')
                elif channels == 2:
                    stream = stream.filter('stereo')
                else:
                    stream = stream.filter('aformat', channel_layouts='stereo')
            
            # Output
            output_kwargs['loglevel'] = 'error'
            out = stream.output(output_path, **output_kwargs)
            
            # Run conversion
            ffmpeg.run(out, quiet=not verbose, overwrite_output=True)
            
            if verbose:
                print(f"✓ Successfully converted to: {output_path}")
            
            return output_path
        
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
    
    def get_audio_info(self, input_path: str) -> Dict[str, Any]:
        """
        Get information about an audio file.
        
        Args:
            input_path (str): Path to the audio file.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'duration' (float): Duration in seconds
                - 'sample_rate' (int): Sample rate in Hz
                - 'channels' (int): Number of audio channels
                - 'codec' (str): Audio codec name
                - 'bitrate' (int): Bitrate in bits per second
        
        Raises:
            FileNotFoundError: If the input file doesn't exist.
            RuntimeError: If unable to read audio information.
        
        Example:
            >>> converter = AudioConverter()
            >>> info = converter.get_audio_info('input.flac')
            >>> print(f"Duration: {info['duration']:.2f}s")
            >>> print(f"Sample Rate: {info['sample_rate']} Hz")
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Audio file not found: {input_path}")
        
        try:
            probe = ffmpeg.probe(input_path)
            audio_stream = next(
                (s for s in probe['streams'] if s['codec_type'] == 'audio'),
                None
            )
            
            if not audio_stream:
                raise RuntimeError("No audio stream found in the file")
            
            # Extract information
            duration = float(probe.get('format', {}).get('duration', 0))
            sample_rate = int(audio_stream.get('sample_rate', 0))
            channels = int(audio_stream.get('channels', 0))
            codec = audio_stream.get('codec_name', 'unknown')
            bitrate = int(audio_stream.get('bit_rate', probe.get('format', {}).get('bit_rate', 0)))
            
            return {
                'duration': duration,
                'sample_rate': sample_rate,
                'channels': channels,
                'codec': codec,
                'bitrate': bitrate,
            }
        
        except Exception as e:
            raise RuntimeError(f"Failed to get audio information: {str(e)}") from e
    
    def batch_convert(
        self,
        input_dir: str,
        output_dir: str,
        recursive: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Convert all audio files in a directory to MP3 or WAV.
        
        Args:
            input_dir (str): Directory containing input audio files.
            output_dir (str): Directory to save converted audio files.
            recursive (bool): If True, search subdirectories recursively.
            verbose (bool): If True, print progress information.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'total' (int): Total number of files processed
                - 'successful' (int): Number of successful conversions
                - 'failed' (int): Number of failed conversions
                - 'errors' (list): List of (filename, error_message) tuples
        
        Example:
            >>> converter = AudioConverter(output_format='mp3')
            >>> result = converter.batch_convert('/path/to/audio', '/path/to/output')
            >>> print(f"Converted: {result['successful']}/{result['total']}")
        """
        if not os.path.isdir(input_dir):
            raise ValueError(f"Input directory not found: {input_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        if recursive:
            for ext in self.SUPPORTED_FORMATS:
                pattern = f"**/*{ext}"
                audio_files.extend(Path(input_dir).glob(pattern))
        else:
            for ext in self.SUPPORTED_FORMATS:
                pattern = f"*{ext}"
                audio_files.extend(Path(input_dir).glob(pattern))
        
        results = {
            'total': len(audio_files),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        if not audio_files:
            if verbose:
                print(f"No audio files found in {input_dir}")
            return results
        
        if verbose:
            print(f"Found {len(audio_files)} audio file(s) to convert")
        
        # Process each file
        for input_file in tqdm(audio_files, disable=not verbose, desc=f"Converting to {self.output_format.upper()}"):
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
                    input_file.stem + f'.{self.output_format}'
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
    
    def set_output_format(self, output_format: Literal['mp3', 'wav']):
        """
        Change the output format for subsequent conversions.
        
        Args:
            output_format (Literal['mp3', 'wav']): The new output format.
        
        Raises:
            ValueError: If output_format is not supported.
        
        Example:
            >>> converter = AudioConverter(output_format='mp3')
            >>> converter.set_output_format('wav')
            >>> converter.convert('input.aac', 'output.wav')
        """
        if output_format.lower() not in ['mp3', 'wav']:
            raise ValueError(
                f"Unsupported output format: {output_format}. "
                "Supported formats: 'mp3', 'wav'"
            )
        self.output_format = output_format.lower()
    
    def set_bitrate(self, bitrate: str):
        """
        Change the audio bitrate for MP3 conversions.
        
        Args:
            bitrate (str): Audio bitrate (e.g., '128k', '192k', '320k').
        
        Example:
            >>> converter = AudioConverter(bitrate='128k')
            >>> converter.set_bitrate('320k')
            >>> converter.convert('input.flac', 'output.mp3')
        """
        self.bitrate = bitrate
