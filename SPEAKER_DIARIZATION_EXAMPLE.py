"""
Speaker Diarization Examples

This file demonstrates how to use the speaker diarization module to identify
and track different speakers in audio files, detect speech activity, speaker
changes, and overlapping speech.

Features demonstrated:
1. Basic speaker diarization
2. Speech activity detection
3. Speaker change detection
4. Overlapping speech detection
5. Speaker embeddings and comparison
6. Batch processing multiple audio files
"""

import os
import torch
from pavo.perception.speech import (
    SpeakerDiarization,
    SpeechActivityDetection,
    SpeakerChangeDetection,
    OverlappingSpeechDetection,
    SpeakerEmbedding
)


# ============================================================================
# EXAMPLE 1: Basic Speaker Diarization
# ============================================================================

def example_basic_diarization():
    """
    Perform basic speaker diarization on an audio file.
    
    This is the simplest way to identify and track speakers in an audio file.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Speaker Diarization")
    print("="*70)
    
    # Initialize diarizer with community pipeline
    diarizer = SpeakerDiarization(
        pipeline_type='community',
        token='your_huggingface_token',  # Set via environment variable
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    try:
        # Perform diarization
        result = diarizer.diarize(
            audio_path='audio.wav',
            verbose=True  # Show progress
        )
        
        # Print results
        result.print_summary()
        
        # Access speaker turns programmatically
        print("\nDetailed speaker turns:")
        for start, end, speaker in result.speaker_turns:
            duration = end - start
            print(f"  {start:.1f}s - {end:.1f}s ({duration:.1f}s): {speaker}")
        
        # Convert to dictionary
        result_dict = result.to_dict()
        print(f"\nResult as dictionary: {result_dict}")
        
    finally:
        # Clean up
        diarizer.unload_models()


# ============================================================================
# EXAMPLE 2: Using Premium Pipeline
# ============================================================================

def example_premium_diarization():
    """
    Use the premium precision-2 pipeline for higher accuracy.
    
    This requires a pyannoteAI API key with credentials.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Premium Speaker Diarization (Precision-2)")
    print("="*70)
    
    # Initialize diarizer with premium pipeline
    diarizer = SpeakerDiarization(
        pipeline_type='premium',
        token='your_pyannoteai_api_key',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    try:
        # Perform diarization with known number of speakers
        result = diarizer.diarize(
            audio_path='audio.wav',
            num_speakers=2  # Specify exact number of speakers
        )
        
        result.print_summary()
        
    finally:
        diarizer.unload_models()


# ============================================================================
# EXAMPLE 3: Specifying Speaker Count Constraints
# ============================================================================

def example_with_constraints():
    """
    Perform diarization with constraints on speaker count.
    
    Useful when you know the approximate number of speakers.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Diarization with Speaker Count Constraints")
    print("="*70)
    
    diarizer = SpeakerDiarization(
        pipeline_type='community',
        token='your_huggingface_token',
        min_speakers=1,    # Minimum 1 speaker
        max_speakers=5     # Maximum 5 speakers
    )
    
    try:
        # Diarize with auto-detection within constraints
        result = diarizer.diarize(
            audio_path='meeting.wav',
            verbose=True
        )
        
        print(f"Detected speakers: {result.num_speakers}")
        
        # Or override at diarization time
        result2 = diarizer.diarize(
            audio_path='meeting.wav',
            num_speakers=3  # Force 3 speakers
        )
        
        print(f"Using forced speaker count: {result2.num_speakers}")
        
    finally:
        diarizer.unload_models()


# ============================================================================
# EXAMPLE 4: Context Manager (Recommended)
# ============================================================================

def example_context_manager():
    """
    Use context manager for automatic resource cleanup.
    
    This is the recommended way to use the diarizer.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Using Context Manager")
    print("="*70)
    
    # Models are automatically unloaded after exiting the context
    with SpeakerDiarization(
        pipeline_type='community',
        token='your_huggingface_token'
    ) as diarizer:
        result = diarizer.diarize('audio.wav', verbose=True)
        result.print_summary()
        # Models automatically cleaned up here


# ============================================================================
# EXAMPLE 5: Speech Activity Detection
# ============================================================================

def example_speech_activity_detection():
    """
    Detect where speech occurs in an audio file.
    
    Useful for identifying silent periods or background noise.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Speech Activity Detection (VAD)")
    print("="*70)
    
    with SpeakerDiarization(
        pipeline_type='community',
        token='your_huggingface_token'
    ) as diarizer:
        # Detect speech segments
        speech_segments = diarizer.detect_speech_activity('audio.wav')
        
        print(f"Found {len(speech_segments)} speech segments:")
        total_speech_time = 0
        for start, end in speech_segments:
            duration = end - start
            total_speech_time += duration
            print(f"  {start:.1f}s - {end:.1f}s ({duration:.1f}s)")
        
        print(f"\nTotal speech time: {total_speech_time:.1f}s")


# ============================================================================
# EXAMPLE 6: Speaker Change Detection
# ============================================================================

def example_speaker_change_detection():
    """
    Detect moments when the speaker changes.
    
    Useful for identifying conversation transitions.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Speaker Change Detection")
    print("="*70)
    
    with SpeakerDiarization(
        pipeline_type='community',
        token='your_huggingface_token'
    ) as diarizer:
        # Detect speaker changes
        change_points = diarizer.detect_speaker_changes('audio.wav')
        
        print(f"Found {len(change_points)} speaker changes at:")
        for timestamp in change_points:
            print(f"  {timestamp:.1f}s")


# ============================================================================
# EXAMPLE 7: Overlapping Speech Detection
# ============================================================================

def example_overlapping_speech_detection():
    """
    Detect segments where multiple speakers speak simultaneously.
    
    Useful for identifying discussion or debate segments.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Overlapping Speech Detection")
    print("="*70)
    
    with SpeakerDiarization(
        pipeline_type='community',
        token='your_huggingface_token'
    ) as diarizer:
        # Detect overlapping speech
        overlapping_segments = diarizer.detect_overlapping_speech('audio.wav')
        
        print(f"Found {len(overlapping_segments)} overlapping segments:")
        total_overlap_time = 0
        for start, end in overlapping_segments:
            duration = end - start
            total_overlap_time += duration
            print(f"  {start:.1f}s - {end:.1f}s ({duration:.1f}s)")
        
        print(f"\nTotal overlapping time: {total_overlap_time:.1f}s")


# ============================================================================
# EXAMPLE 8: Speaker Embeddings
# ============================================================================

def example_speaker_embeddings():
    """
    Extract speaker embeddings for speaker identification and comparison.
    
    Embeddings can be used for speaker verification and clustering.
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Speaker Embeddings")
    print("="*70)
    
    with SpeakerDiarization(
        pipeline_type='community',
        token='your_huggingface_token'
    ) as diarizer:
        # First, perform diarization
        result = diarizer.diarize('audio.wav')
        
        # Extract embeddings for the detected speakers
        embeddings = diarizer.get_speaker_embeddings(
            'audio.wav',
            speaker_turns=result.speaker_turns
        )
        
        print(f"Extracted embeddings for {len(embeddings)} speakers:")
        for speaker_id, embedding in embeddings.items():
            print(f"  {speaker_id}: shape {embedding.shape}")


# ============================================================================
# EXAMPLE 9: Comparing Speaker Embeddings
# ============================================================================

def example_embedding_comparison():
    """
    Compare speaker embeddings from different audio files.
    
    Useful for speaker verification across multiple sessions.
    """
    print("\n" + "="*70)
    print("EXAMPLE 9: Comparing Speaker Embeddings")
    print("="*70)
    
    with SpeakerDiarization(
        pipeline_type='community',
        token='your_huggingface_token'
    ) as diarizer:
        # Extract embeddings from two audio files
        embeddings1 = diarizer.get_speaker_embeddings('audio1.wav')
        embeddings2 = diarizer.get_speaker_embeddings('audio2.wav')
        
        speaker_embedder = diarizer.speaker_embedding
        
        # Compare embeddings
        if 'full_audio' in embeddings1 and 'full_audio' in embeddings2:
            similarity = speaker_embedder.compare_embeddings(
                embeddings1['full_audio'],
                embeddings2['full_audio']
            )
            
            print(f"Speaker similarity: {similarity:.3f}")
            if similarity > 0.8:
                print("  → Likely the same speaker")
            elif similarity > 0.5:
                print("  → Possibly the same speaker")
            else:
                print("  → Different speakers")


# ============================================================================
# EXAMPLE 10: Batch Processing Multiple Files
# ============================================================================

def example_batch_processing():
    """
    Process multiple audio files with a single diarizer instance.
    
    More efficient than creating a new diarizer for each file.
    """
    print("\n" + "="*70)
    print("EXAMPLE 10: Batch Processing Multiple Audio Files")
    print("="*70)
    
    audio_files = [
        'interview_1.wav',
        'interview_2.wav',
        'meeting.wav',
        'podcast.wav'
    ]
    
    results = {}
    
    # Create diarizer once and reuse
    with SpeakerDiarization(
        pipeline_type='community',
        token='your_huggingface_token'
    ) as diarizer:
        for audio_file in audio_files:
            if os.path.exists(audio_file):
                print(f"\nProcessing: {audio_file}")
                try:
                    result = diarizer.diarize(audio_file)
                    results[audio_file] = result
                    print(f"  Speakers: {result.num_speakers}")
                    print(f"  Turns: {len(result.speaker_turns)}")
                except FileNotFoundError:
                    print(f"  File not found: {audio_file}")
    
    # Summarize results
    print("\n" + "-"*70)
    print("Summary of all files:")
    for audio_file, result in results.items():
        print(f"  {audio_file}: {result.num_speakers} speakers, {len(result.speaker_turns)} turns")


# ============================================================================
# EXAMPLE 11: Advanced Configuration
# ============================================================================

def example_advanced_configuration():
    """
    Advanced configuration with custom parameters.
    """
    print("\n" + "="*70)
    print("EXAMPLE 11: Advanced Configuration")
    print("="*70)
    
    # Detect GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    diarizer = SpeakerDiarization(
        pipeline_type='community',
        token='your_huggingface_token',
        device=device,
        num_speakers=None,  # Auto-detect
        min_speakers=1,
        max_speakers=10
    )
    
    try:
        # Diarize with all components
        result = diarizer.diarize('audio.wav', verbose=True)
        
        # Get additional insights
        vad_segments = diarizer.detect_speech_activity('audio.wav')
        changes = diarizer.detect_speaker_changes('audio.wav')
        overlaps = diarizer.detect_overlapping_speech('audio.wav')
        
        print(f"\nComprehensive Analysis:")
        print(f"  Speakers: {result.num_speakers}")
        print(f"  Speech segments: {len(vad_segments)}")
        print(f"  Speaker changes: {len(changes)}")
        print(f"  Overlapping segments: {len(overlaps)}")
        
    finally:
        diarizer.unload_models()


# ============================================================================
# EXAMPLE 12: Using Individual Components
# ============================================================================

def example_individual_components():
    """
    Use individual components without the main SpeakerDiarization class.
    
    Useful when you only need specific functionality.
    """
    print("\n" + "="*70)
    print("EXAMPLE 12: Using Individual Components")
    print("="*70)
    
    # Use only speech activity detection
    vad = SpeechActivityDetection(device='cpu')
    try:
        segments = vad.detect('audio.wav')
        print(f"Speech activity detected {len(segments)} segments")
    finally:
        vad.unload_model()
    
    # Use only speaker change detection
    scd = SpeakerChangeDetection(device='cpu')
    try:
        changes = scd.detect_changes('audio.wav')
        print(f"Found {len(changes)} speaker changes")
    finally:
        scd.unload_model()
    
    # Use only overlapping speech detection
    osd = OverlappingSpeechDetection(device='cpu')
    try:
        overlaps = osd.detect('audio.wav')
        print(f"Found {len(overlaps)} overlapping segments")
    finally:
        osd.unload_model()


# ============================================================================
# EXAMPLE 13: Error Handling
# ============================================================================

def example_error_handling():
    """
    Proper error handling for common issues.
    """
    print("\n" + "="*70)
    print("EXAMPLE 13: Error Handling")
    print("="*70)
    
    with SpeakerDiarization(
        pipeline_type='community',
        token='your_huggingface_token'
    ) as diarizer:
        
        # Handle missing files
        try:
            result = diarizer.diarize('nonexistent.wav')
        except FileNotFoundError as e:
            print(f"Error: {e}")
        
        # Handle missing tokens
        try:
            bad_diarizer = SpeakerDiarization(
                pipeline_type='community',
                token=None  # No token provided
            )
            bad_diarizer.diarize('audio.wav')
        except ValueError as e:
            print(f"Error: {e}")


# ============================================================================
# EXAMPLE 14: Output Formats
# ============================================================================

def example_output_formats():
    """
    Convert diarization results to different formats.
    """
    print("\n" + "="*70)
    print("EXAMPLE 14: Output Formats")
    print("="*70)
    
    with SpeakerDiarization(
        pipeline_type='community',
        token='your_huggingface_token'
    ) as diarizer:
        result = diarizer.diarize('audio.wav')
        
        # As dictionary
        print("As dictionary:")
        print(result.to_dict())
        
        # As print summary
        print("\nAs formatted summary:")
        result.print_summary()
        
        # Raw access
        print("\nRaw speaker turns:")
        print(result.speaker_turns)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("SPEAKER DIARIZATION MODULE - EXAMPLES")
    print("="*70)
    print("\nBefore running these examples, you need to:")
    print("1. Install pyannote.audio: pip install pyannote.audio")
    print("2. Install ffmpeg: brew install ffmpeg (on macOS)")
    print("3. Create a Hugging Face account and accept user conditions for:")
    print("   - pyannote/speaker-diarization-community-1 (free)")
    print("4. Generate a Hugging Face access token at hf.co/settings/tokens")
    print("5. Set environment variable: export HUGGINGFACE_TOKEN='your_token'")
    print("\n" + "="*70)
    
    # Uncomment the examples you want to run:
    # Make sure to replace 'audio.wav' with actual audio file paths
    
    # example_basic_diarization()
    # example_premium_diarization()
    # example_with_constraints()
    # example_context_manager()
    # example_speech_activity_detection()
    # example_speaker_change_detection()
    # example_overlapping_speech_detection()
    # example_speaker_embeddings()
    # example_embedding_comparison()
    # example_batch_processing()
    # example_advanced_configuration()
    # example_individual_components()
    # example_error_handling()
    # example_output_formats()
    
    print("\nTo run an example, uncomment it in the main() section.")
    print("Don't forget to update audio file paths and your token!")
