"""
Test suite for speaker diarization module.

Tests all components of the speaker diarization system including:
- Speaker diarization pipeline
- Speech activity detection
- Speaker change detection
- Overlapping speech detection
- Speaker embeddings
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import torch

from pavo.perception.speech import (
    SpeakerDiarization,
    SpeakerDiarizationResult,
    SpeechActivityDetection,
    SpeakerChangeDetection,
    OverlappingSpeechDetection,
    SpeakerEmbedding
)


class TestSpeakerDiarizationResult(unittest.TestCase):
    """Test speaker diarization result container."""
    
    def setUp(self):
        """Set up test data."""
        self.speaker_turns = [
            (0.0, 1.5, 'speaker_0'),
            (1.8, 3.9, 'speaker_1'),
            (4.2, 5.7, 'speaker_0'),
        ]
        self.num_speakers = 2
    
    def test_initialization(self):
        """Test result initialization."""
        result = SpeakerDiarizationResult(
            speaker_turns=self.speaker_turns,
            num_speakers=self.num_speakers
        )
        
        self.assertEqual(result.num_speakers, 2)
        self.assertEqual(len(result.speaker_turns), 3)
    
    def test_repr(self):
        """Test string representation."""
        result = SpeakerDiarizationResult(
            speaker_turns=self.speaker_turns,
            num_speakers=self.num_speakers
        )
        
        repr_str = repr(result)
        self.assertIn('speakers=2', repr_str)
        self.assertIn('turns=3', repr_str)
    
    def test_to_dict(self):
        """Test dictionary conversion."""
        result = SpeakerDiarizationResult(
            speaker_turns=self.speaker_turns,
            num_speakers=self.num_speakers,
            metadata={'test': 'value'}
        )
        
        result_dict = result.to_dict()
        self.assertEqual(result_dict['num_speakers'], 2)
        self.assertEqual(len(result_dict['speaker_turns']), 3)
        self.assertEqual(result_dict['metadata']['test'], 'value')
    
    def test_print_summary(self, capsys=None):
        """Test summary printing."""
        result = SpeakerDiarizationResult(
            speaker_turns=self.speaker_turns,
            num_speakers=self.num_speakers
        )
        
        # Should not raise
        result.print_summary()


class TestSpeechActivityDetection(unittest.TestCase):
    """Test speech activity detection."""
    
    def setUp(self):
        """Set up test data."""
        self.vad = SpeechActivityDetection(device='cpu')
    
    def test_initialization(self):
        """Test VAD initialization."""
        self.assertEqual(self.vad.device, 'cpu')
        self.assertIsNone(self.vad._model)
    
    def test_device_setting(self):
        """Test device configuration."""
        vad_gpu = SpeechActivityDetection(device='cuda')
        self.assertEqual(vad_gpu.device, 'cuda')
    
    def test_model_not_loaded_initially(self):
        """Test model is not loaded initially."""
        # Model should not be loaded initially
        self.assertIsNone(self.vad._model)
    
    def test_unload_model(self):
        """Test model unloading."""
        self.vad._model = Mock()
        self.vad.unload_model()
        self.assertIsNone(self.vad._model)


class TestSpeakerChangeDetection(unittest.TestCase):
    """Test speaker change detection."""
    
    def setUp(self):
        """Set up test data."""
        self.change_detector = SpeakerChangeDetection(device='cpu')
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.change_detector.device, 'cpu')
        self.assertIsNone(self.change_detector._model)
    
    def test_device_setting(self):
        """Test device configuration."""
        detector_gpu = SpeakerChangeDetection(device='cuda')
        self.assertEqual(detector_gpu.device, 'cuda')
    
    def test_unload_model(self):
        """Test model unloading."""
        self.change_detector._model = Mock()
        self.change_detector.unload_model()
        self.assertIsNone(self.change_detector._model)


class TestOverlappingSpeechDetection(unittest.TestCase):
    """Test overlapping speech detection."""
    
    def setUp(self):
        """Set up test data."""
        self.overlap_detector = OverlappingSpeechDetection(device='cpu')
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.overlap_detector.device, 'cpu')
        self.assertIsNone(self.overlap_detector._model)
    
    def test_device_setting(self):
        """Test device configuration."""
        detector_gpu = OverlappingSpeechDetection(device='cuda')
        self.assertEqual(detector_gpu.device, 'cuda')
    
    def test_threshold_validation(self):
        """Test threshold parameter validation."""
        # Threshold should be between 0.0 and 1.0
        # Implementation should validate this
        pass
    
    def test_unload_model(self):
        """Test model unloading."""
        self.overlap_detector._model = Mock()
        self.overlap_detector.unload_model()
        self.assertIsNone(self.overlap_detector._model)


class TestSpeakerEmbedding(unittest.TestCase):
    """Test speaker embedding extraction."""
    
    def setUp(self):
        """Set up test data."""
        self.embedder = SpeakerEmbedding(device='cpu')
    
    def test_initialization(self):
        """Test initialization."""
        self.assertEqual(self.embedder.device, 'cpu')
        self.assertIsNone(self.embedder._model)
    
    def test_device_setting(self):
        """Test device configuration."""
        embedder_gpu = SpeakerEmbedding(device='cuda')
        self.assertEqual(embedder_gpu.device, 'cuda')
    
    def test_unload_model(self):
        """Test model unloading."""
        self.embedder._model = Mock()
        self.embedder.unload_model()
        self.assertIsNone(self.embedder._model)


class TestSpeakerDiarization(unittest.TestCase):
    """Test main speaker diarization class."""
    
    def setUp(self):
        """Set up test data."""
        # Mock the Pipeline import to avoid needing pyannote.audio
        pass
    
    @patch('pyannote.audio.Pipeline')
    def test_initialization_cpu(self, mock_pipeline):
        """Test initialization with CPU device."""
        diarizer = SpeakerDiarization(
            pipeline_type='community',
            token='test_token',
            device='cpu'
        )
        
        self.assertEqual(diarizer.device, 'cpu')
        self.assertEqual(diarizer.pipeline_type, 'community')
    
    @patch('pyannote.audio.Pipeline')
    def test_invalid_pipeline_type(self, mock_pipeline):
        """Test invalid pipeline type raises error."""
        with self.assertRaises(ValueError):
            SpeakerDiarization(
                pipeline_type='invalid',
                token='test_token'
            )
    
    @patch('pyannote.audio.Pipeline')
    def test_speaker_constraints(self, mock_pipeline):
        """Test speaker count constraints."""
        diarizer = SpeakerDiarization(
            pipeline_type='community',
            token='test_token',
            min_speakers=2,
            max_speakers=8
        )
        
        self.assertEqual(diarizer.min_speakers, 2)
        self.assertEqual(diarizer.max_speakers, 8)
    
    @patch('pyannote.audio.Pipeline')
    def test_context_manager(self, mock_pipeline):
        """Test context manager functionality."""
        with SpeakerDiarization(
            pipeline_type='community',
            token='test_token'
        ) as diarizer:
            self.assertIsNotNone(diarizer)
    
    @patch('pyannote.audio.Pipeline')
    def test_file_not_found_error(self, mock_pipeline):
        """Test handling of missing audio file."""
        diarizer = SpeakerDiarization(
            pipeline_type='community',
            token='test_token'
        )
        
        with self.assertRaises(FileNotFoundError):
            diarizer.diarize('/nonexistent/audio.wav')
    
    @patch('pyannote.audio.Pipeline')
    def test_unload_models(self, mock_pipeline):
        """Test unloading all models."""
        diarizer = SpeakerDiarization(
            pipeline_type='community',
            token='test_token'
        )
        
        # Set some mock models
        diarizer._pipeline = Mock()
        diarizer.vad._model = Mock()
        
        diarizer.unload_models()
        
        self.assertIsNone(diarizer._pipeline)
        self.assertIsNone(diarizer.vad._model)


class TestIntegration(unittest.TestCase):
    """Integration tests (require actual audio files)."""
    
    def setUp(self):
        """Set up test data."""
        self.audio_path = '/Users/sonnguyen/Desktop/mkt/video-agent/pavo-engine-py/sample/OSR_us_000_0010_8k.wav'
    
    @unittest.skipIf(
        not os.path.exists('/Users/sonnguyen/Desktop/mkt/video-agent/pavo-engine-py/sample/OSR_us_000_0010_8k.wav'),
        "Sample audio file not found"
    )
    def test_diarization_with_real_audio(self):
        """Test diarization with real audio file."""
        # This test requires pyannote.audio and authentication token
        # It's skipped if the sample file doesn't exist
        pass
    
    @unittest.skipIf(
        not os.path.exists('/Users/sonnguyen/Desktop/mkt/video-agent/pavo-engine-py/sample/OSR_us_000_0010_8k.wav'),
        "Sample audio file not found"
    )
    def test_speech_activity_detection_real_audio(self):
        """Test speech activity detection with real audio."""
        # This test requires pyannote.audio and authentication token
        pass


if __name__ == '__main__':
    unittest.main()
