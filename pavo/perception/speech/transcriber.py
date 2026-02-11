"""
Speech transcription module using OpenAI Whisper model.

This module provides a SpeechTranscriber class that can transcribe audio files
to text using the OpenAI Whisper speech recognition model.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any


class SpeechTranscriber:
    """
    A class for transcribing audio files to text using OpenAI's Whisper model.
    
    This class handles audio file transcription with support for various audio formats
    and provides options for language specification and model selection.
    
    Attributes:
        model_name (str): The Whisper model to use (tiny, base, small, medium, large)
        language (Optional[str]): The language code (e.g., 'en', 'vi'). If None, auto-detect.
        task (str): The task type - 'transcribe' or 'translate'
    
    Example:
        >>> transcriber = SpeechTranscriber(model_name='base', language='en')
        >>> result = transcriber.transcribe('audio.wav')
        >>> print(result['text'])
    """
    
    def __init__(
        self,
        model_name: str = 'base',
        language: Optional[str] = None,
        task: str = 'transcribe',
        device: str = 'cpu'
    ):
        """
        Initialize the SpeechTranscriber.
        
        Args:
            model_name (str): Whisper model size. Options: 'tiny', 'tiny.en', 'base', 
                            'base.en', 'small', 'small.en', 'medium', 'medium.en',
                            'large', 'large-v1', 'large-v2', 'large-v3', 'large-v3-turbo'.
                            Default is 'base'.
            language (Optional[str]): Language code (e.g., 'en', 'vi'). If None, language 
                                     will be auto-detected. Default is None.
            task (str): Task type - 'transcribe' for speech-to-text or 'translate' for
                       speech translation to English. Default is 'transcribe'.
            device (str): Device to use for inference - 'cpu' or 'cuda'. Default is 'cpu'.
        
        Raises:
            ImportError: If whisper package is not installed.
            ValueError: If model_name or task is invalid.
        """
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "Whisper is not installed. Please install it using:\n"
                "pip install openai-whisper"
            )
        
        self.whisper = whisper
        self.model_name = model_name
        self.language = language
        self.task = task
        self.device = device
        self._model = None
    
    def _load_model(self):
        """Load the Whisper model (lazy loading)."""
        if self._model is None:
            self._model = self.whisper.load_model(
                self.model_name,
                device=self.device
            )
        return self._model
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path (str): Path to the audio file to transcribe.
            language (Optional[str]): Override the instance language setting for this call.
                                     If None, uses the instance's language setting.
            task (Optional[str]): Override the instance task setting for this call.
                                 If None, uses the instance's task setting.
            verbose (bool): If True, print progress information. Default is False.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'text' (str): The transcribed text
                - 'language' (str): The detected language code
                - 'segments' (list): List of segment dictionaries with timing and text
                  Each segment contains: 'id', 'seek', 'start', 'end', 'text', 'tokens', 'temperature', 'avg_logprob', 'compression_ratio', 'no_speech_prob'
        
        Raises:
            FileNotFoundError: If the audio file doesn't exist.
            ValueError: If the audio file format is not supported.
            RuntimeError: If transcription fails.
        
        Example:
            >>> transcriber = SpeechTranscriber(model_name='base')
            >>> result = transcriber.transcribe('speech.wav')
            >>> print(result['text'])
            >>> print(f"Language: {result['language']}")
        """
        # Validate audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Validate supported audio format
        supported_formats = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus', '.webm'}
        file_ext = Path(audio_path).suffix.lower()
        if file_ext not in supported_formats:
            raise ValueError(
                f"Unsupported audio format: {file_ext}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
        
        # Use provided parameters or fall back to instance defaults
        language = language or self.language
        task = task or self.task
        
        try:
            # Load model
            model = self._load_model()
            
            # Transcribe audio
            result = model.transcribe(
                audio_path,
                language=language,
                task=task,
                verbose=verbose
            )
            
            return result
        
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}") from e
    
    def transcribe_file(self, audio_path: str) -> str:
        """
        Convenience method to transcribe an audio file and return only the text.
        
        Args:
            audio_path (str): Path to the audio file.
        
        Returns:
            str: The transcribed text.
        
        Raises:
            FileNotFoundError: If the audio file doesn't exist.
            ValueError: If the audio file format is not supported.
            RuntimeError: If transcription fails.
        
        Example:
            >>> transcriber = SpeechTranscriber()
            >>> text = transcriber.transcribe_file('speech.wav')
            >>> print(text)
        """
        result = self.transcribe(audio_path)
        return result['text']
    
    def get_language(self, audio_path: str) -> str:
        """
        Detect the language of an audio file.
        
        Args:
            audio_path (str): Path to the audio file.
        
        Returns:
            str: The detected language code (e.g., 'en', 'vi').
        
        Raises:
            FileNotFoundError: If the audio file doesn't exist.
            ValueError: If the audio file format is not supported.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        result = self.transcribe(audio_path)
        return result['language']
    
    def unload_model(self):
        """Unload the model to free up memory."""
        self._model = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model to free resources."""
        self.unload_model()
        return False
