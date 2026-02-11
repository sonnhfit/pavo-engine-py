"""
Speaker Diarization module using pyannote.audio.

This module provides comprehensive speaker diarization capabilities including:
- Speaker identification and classification
- Speech activity detection
- Speaker change detection
- Overlapping speech detection
- Speaker embeddings (voice fingerprints)

It supports both community and premium pipelines from pyannote.audio.
"""

import os
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import warnings

import torch


class SpeakerDiarizationResult:
    """
    Container for speaker diarization results.
    
    Attributes:
        speaker_turns (List[Tuple[float, float, str]]): List of (start, end, speaker) tuples
        num_speakers (int): Detected number of speakers
        timeline (Any): Pyannote timeline object
        metadata (Dict[str, Any]): Additional metadata
    """
    
    def __init__(
        self,
        speaker_turns: List[Tuple[float, float, str]],
        num_speakers: int,
        timeline: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize speaker diarization result.
        
        Args:
            speaker_turns: List of (start_time, end_time, speaker_id) tuples
            num_speakers: Number of unique speakers detected
            timeline: Raw timeline object from pyannote
            metadata: Additional metadata dictionary
        """
        self.speaker_turns = speaker_turns
        self.num_speakers = num_speakers
        self.timeline = timeline
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"SpeakerDiarizationResult(speakers={self.num_speakers}, turns={len(self.speaker_turns)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'speaker_turns': self.speaker_turns,
            'num_speakers': self.num_speakers,
            'metadata': self.metadata
        }
    
    def print_summary(self):
        """Print a human-readable summary of the diarization results."""
        print(f"Speaker Diarization Summary:")
        print(f"  Total speakers: {self.num_speakers}")
        print(f"  Total turns: {len(self.speaker_turns)}")
        if self.speaker_turns:
            duration = self.speaker_turns[-1][1]
            print(f"  Duration: {duration:.1f}s")
            print("\n  Speaker turns:")
            for start, end, speaker in self.speaker_turns:
                print(f"    {start:.1f}s - {end:.1f}s: {speaker}")


class SpeechActivityDetection:
    """
    Speech Activity Detection (VAD) - detects segments where speech is present.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize Speech Activity Detection.
        
        Args:
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = device
        self._model = None
    
    def _load_model(self):
        """Load the VAD model (lazy loading)."""
        if self._model is None:
            try:
                from pyannote.audio import Model
                self._model = Model.from_pretrained(
                    "pyannote/voice-activity-detection",
                    use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
                )
                self._model = self._model.to(torch.device(self.device))
            except ImportError:
                raise ImportError(
                    "pyannote.audio is not installed. Please install it using:\n"
                    "pip install pyannote.audio"
                )
        return self._model
    
    def detect(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        Detect speech activity in audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of (start, end) tuples representing speech segments
        """
        from pyannote.audio.core.io import AudioFile
        
        model = self._load_model()
        
        # Load audio
        waveform, sample_rate = AudioFile(audio_path)[:]
        
        # Get predictions
        predictions = model({"waveform": waveform, "sample_rate": sample_rate})
        
        # Extract speech segments
        speech_segments = []
        for segment, label in predictions.items():
            if label > 0.5:  # Threshold for speech activity
                speech_segments.append((float(segment.start), float(segment.end)))
        
        return speech_segments
    
    def unload_model(self):
        """Unload the model to free memory."""
        self._model = None


class SpeakerChangeDetection:
    """
    Speaker Change Detection - detects moments when speaker changes.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize Speaker Change Detection.
        
        Args:
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = device
        self._model = None
    
    def _load_model(self):
        """Load the speaker change detection model."""
        if self._model is None:
            try:
                from pyannote.audio import Model
                self._model = Model.from_pretrained(
                    "pyannote/speaker-segmentation",
                    use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
                )
                self._model = self._model.to(torch.device(self.device))
            except ImportError:
                raise ImportError(
                    "pyannote.audio is not installed. Please install it using:\n"
                    "pip install pyannote.audio"
                )
        return self._model
    
    def detect_changes(self, audio_path: str, threshold: float = 0.5) -> List[float]:
        """
        Detect speaker change points in audio file.
        
        Args:
            audio_path: Path to audio file
            threshold: Detection threshold (0.0 to 1.0)
            
        Returns:
            List of timestamps (in seconds) where speaker changes occur
        """
        from pyannote.audio.core.io import AudioFile
        
        model = self._load_model()
        
        # Load audio
        waveform, sample_rate = AudioFile(audio_path)[:]
        
        # Get predictions
        predictions = model({"waveform": waveform, "sample_rate": sample_rate})
        
        # Extract change points
        change_points = []
        prev_label = 0.0
        
        for segment, score in predictions.items():
            if score > threshold and prev_label <= threshold:
                change_points.append(float(segment.start))
            prev_label = score
        
        return change_points
    
    def unload_model(self):
        """Unload the model to free memory."""
        self._model = None


class OverlappingSpeechDetection:
    """
    Overlapping Speech Detection - detects when multiple speakers speak simultaneously.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize Overlapping Speech Detection.
        
        Args:
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = device
        self._model = None
    
    def _load_model(self):
        """Load the overlapping speech detection model."""
        if self._model is None:
            try:
                from pyannote.audio import Model
                self._model = Model.from_pretrained(
                    "pyannote/overlapped-speech-detection",
                    use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
                )
                self._model = self._model.to(torch.device(self.device))
            except ImportError:
                raise ImportError(
                    "pyannote.audio is not installed. Please install it using:\n"
                    "pip install pyannote.audio"
                )
        return self._model
    
    def detect(self, audio_path: str, threshold: float = 0.5) -> List[Tuple[float, float]]:
        """
        Detect overlapping speech segments in audio file.
        
        Args:
            audio_path: Path to audio file
            threshold: Detection threshold (0.0 to 1.0)
            
        Returns:
            List of (start, end) tuples where overlapping speech occurs
        """
        from pyannote.audio.core.io import AudioFile
        
        model = self._load_model()
        
        # Load audio
        waveform, sample_rate = AudioFile(audio_path)[:]
        
        # Get predictions
        predictions = model({"waveform": waveform, "sample_rate": sample_rate})
        
        # Extract overlapping segments
        overlapping_segments = []
        for segment, label in predictions.items():
            if label > threshold:
                overlapping_segments.append((float(segment.start), float(segment.end)))
        
        return overlapping_segments
    
    def unload_model(self):
        """Unload the model to free memory."""
        self._model = None


class SpeakerEmbedding:
    """
    Speaker Embedding - generates speaker embeddings (voice fingerprints) for speaker identification.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize Speaker Embedding.
        
        Args:
            device: Device to use ('cpu' or 'cuda')
        """
        self.device = device
        self._model = None
    
    def _load_model(self):
        """Load the speaker embedding model."""
        if self._model is None:
            try:
                from pyannote.audio import Model
                self._model = Model.from_pretrained(
                    "pyannote/speaker-embedding",
                    use_auth_token=os.getenv('HUGGINGFACE_TOKEN')
                )
                self._model = self._model.to(torch.device(self.device))
            except ImportError:
                raise ImportError(
                    "pyannote.audio is not installed. Please install it using:\n"
                    "pip install pyannote.audio"
                )
        return self._model
    
    def get_embeddings(
        self,
        audio_path: str,
        speaker_turns: Optional[List[Tuple[float, float, str]]] = None
    ) -> Dict[str, Any]:
        """
        Extract speaker embeddings from audio file.
        
        Args:
            audio_path: Path to audio file
            speaker_turns: Optional list of speaker turns for specific segments
            
        Returns:
            Dictionary mapping speaker IDs to their embeddings
        """
        from pyannote.audio.core.io import AudioFile
        import numpy as np
        
        model = self._load_model()
        
        # Load audio
        waveform, sample_rate = AudioFile(audio_path)[:]
        
        embeddings_dict = {}
        
        if speaker_turns:
            # Extract embeddings for specific speaker segments
            from pyannote.core import Segment
            
            for start, end, speaker_id in speaker_turns:
                segment = Segment(start, end)
                embedding = model.crop(
                    {"waveform": waveform, "sample_rate": sample_rate},
                    segment
                )
                
                if speaker_id not in embeddings_dict:
                    embeddings_dict[speaker_id] = []
                
                embeddings_dict[speaker_id].append(embedding.numpy())
            
            # Average embeddings for each speaker
            for speaker_id in embeddings_dict:
                embeddings_dict[speaker_id] = np.mean(
                    embeddings_dict[speaker_id],
                    axis=0
                )
        else:
            # Extract embeddings for entire audio
            embedding = model({"waveform": waveform, "sample_rate": sample_rate})
            embeddings_dict['full_audio'] = embedding.numpy()
        
        return embeddings_dict
    
    def compare_embeddings(self, embedding1: Any, embedding2: Any) -> float:
        """
        Compare two speaker embeddings using cosine similarity.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Reshape for sklearn
        emb1 = embedding1.reshape(1, -1) if embedding1.ndim == 1 else embedding1
        emb2 = embedding2.reshape(1, -1) if embedding2.ndim == 1 else embedding2
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        return float(similarity)
    
    def unload_model(self):
        """Unload the model to free memory."""
        self._model = None


class SpeakerDiarization:
    """
    Main Speaker Diarization class - orchestrates speaker identification and classification.
    
    This class provides high-level interface for speaker diarization tasks including:
    - Identifying and tracking speakers over time
    - Detecting speech activity
    - Detecting speaker changes
    - Detecting overlapping speech
    - Extracting speaker embeddings
    
    Supports both community (open-source) and premium (pyannoteAI) pipelines.
    
    Example:
        >>> diarizer = SpeakerDiarization(
        ...     pipeline_type='community',
        ...     token='HUGGINGFACE_TOKEN'
        ... )
        >>> result = diarizer.diarize('audio.wav')
        >>> result.print_summary()
    """
    
    def __init__(
        self,
        pipeline_type: str = 'community',
        token: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 10
    ):
        """
        Initialize Speaker Diarization.
        
        Args:
            pipeline_type: Type of pipeline ('community' or 'premium')
                - 'community': Open-source community-1 pipeline (free)
                - 'premium': Premium precision-2 pipeline (requires pyannoteAI API key)
                Default: 'community'
            token: Huggingface or pyannoteAI token
                - For 'community': HUGGINGFACE_TOKEN
                - For 'premium': PYANNOTEAI_API_KEY
                If None, will look for environment variables:
                - HUGGINGFACE_TOKEN
                - PYANNOTEAI_API_KEY
            device: Device to use ('cpu' or 'cuda')
            num_speakers: Exact number of speakers. If None, will be auto-detected.
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
            
        Raises:
            ImportError: If pyannote.audio is not installed
            ValueError: If invalid pipeline_type
        """
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError(
                "pyannote.audio is not installed. Please install it using:\n"
                "pip install pyannote.audio\n\n"
                "Note: You also need ffmpeg installed on your system. "
                "On macOS: brew install ffmpeg"
            )
        
        if pipeline_type not in ['community', 'premium']:
            raise ValueError(f"Invalid pipeline_type: {pipeline_type}. Must be 'community' or 'premium'")
        
        self.Pipeline = Pipeline
        self.pipeline_type = pipeline_type
        self.device = device
        self.num_speakers = num_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline = None
        
        # Set token from parameter or environment variable
        if token is None:
            if pipeline_type == 'community':
                token = os.getenv('HUGGINGFACE_TOKEN')
            else:
                token = os.getenv('PYANNOTEAI_API_KEY')
        
        self.token = token
        
        # Initialize sub-components
        self.vad = SpeechActivityDetection(device=device)
        self.speaker_change = SpeakerChangeDetection(device=device)
        self.overlapping = OverlappingSpeechDetection(device=device)
        self.speaker_embedding = SpeakerEmbedding(device=device)
    
    def _load_pipeline(self):
        """Load the diarization pipeline (lazy loading)."""
        if self._pipeline is None:
            if self.pipeline_type == 'community':
                model_id = "pyannote/speaker-diarization-community-1"
            else:
                model_id = "pyannote/speaker-diarization-precision-2"
            
            if self.token is None:
                raise ValueError(
                    f"Token is required for {self.pipeline_type} pipeline. "
                    f"Please provide token parameter or set environment variable:\n"
                    f"For community: export HUGGINGFACE_TOKEN='your_token'\n"
                    f"For premium: export PYANNOTEAI_API_KEY='your_token'"
                )
            
            self._pipeline = self.Pipeline.from_pretrained(
                model_id,
                use_auth_token=self.token
            )
            
            # Send to device
            self._pipeline.to(torch.device(self.device))
        
        return self._pipeline
    
    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        verbose: bool = False
    ) -> SpeakerDiarizationResult:
        """
        Perform speaker diarization on an audio file.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Override instance num_speakers setting
            min_speakers: Override instance min_speakers setting
            max_speakers: Override instance max_speakers setting
            verbose: Print progress information
            
        Returns:
            SpeakerDiarizationResult containing speaker turns and metadata
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If diarization fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            if verbose:
                print(f"Loading diarization pipeline ({self.pipeline_type})...")
            
            pipeline = self._load_pipeline()
            
            # Use provided parameters or fall back to instance defaults
            num_speakers = num_speakers or self.num_speakers
            min_speakers = min_speakers or self.min_speakers
            max_speakers = max_speakers or self.max_speakers
            
            # Configure pipeline parameters
            if num_speakers is not None:
                pipeline.instantiate({"num_speakers": num_speakers})
            else:
                pipeline.instantiate({
                    "min_speakers": min_speakers,
                    "max_speakers": max_speakers
                })
            
            if verbose:
                print(f"Processing audio: {audio_path}")
            
            # Run diarization
            if verbose:
                from pyannote.audio.pipelines.utils.hook import ProgressHook
                with ProgressHook() as hook:
                    diarization = pipeline(audio_path, hook=hook)
            else:
                diarization = pipeline(audio_path)
            
            # Extract speaker turns
            speaker_turns = []
            unique_speakers = set()
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_turns.append((float(turn.start), float(turn.end), speaker))
                unique_speakers.add(speaker)
            
            # Sort by start time
            speaker_turns.sort(key=lambda x: x[0])
            
            if verbose:
                print(f"Detected {len(unique_speakers)} speakers")
                print(f"Total speech turns: {len(speaker_turns)}")
            
            return SpeakerDiarizationResult(
                speaker_turns=speaker_turns,
                num_speakers=len(unique_speakers),
                timeline=diarization,
                metadata={
                    'audio_path': audio_path,
                    'pipeline_type': self.pipeline_type,
                    'min_speakers': min_speakers,
                    'max_speakers': max_speakers
                }
            )
        
        except Exception as e:
            raise RuntimeError(f"Diarization failed: {str(e)}") from e
    
    def detect_speech_activity(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        Detect speech activity segments in audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of (start, end) tuples for speech segments
        """
        return self.vad.detect(audio_path)
    
    def detect_speaker_changes(self, audio_path: str) -> List[float]:
        """
        Detect speaker change points in audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of timestamps where speaker changes occur
        """
        return self.speaker_change.detect_changes(audio_path)
    
    def detect_overlapping_speech(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        Detect overlapping speech segments in audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of (start, end) tuples where multiple speakers overlap
        """
        return self.overlapping.detect(audio_path)
    
    def get_speaker_embeddings(
        self,
        audio_path: str,
        speaker_turns: Optional[List[Tuple[float, float, str]]] = None
    ) -> Dict[str, Any]:
        """
        Extract speaker embeddings from audio file.
        
        Args:
            audio_path: Path to audio file
            speaker_turns: Optional speaker turns to extract embeddings for
            
        Returns:
            Dictionary mapping speaker IDs to embeddings
        """
        return self.speaker_embedding.get_embeddings(audio_path, speaker_turns)
    
    def unload_models(self):
        """Unload all models to free memory."""
        if self._pipeline is not None:
            self._pipeline = None
        self.vad.unload_model()
        self.speaker_change.unload_model()
        self.overlapping.unload_model()
        self.speaker_embedding.unload_model()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload models."""
        self.unload_models()
        return False
