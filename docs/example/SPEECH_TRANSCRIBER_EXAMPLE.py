#!/usr/bin/env python3
"""
Example usage of the SpeechTranscriber class from pavo.perception.speech module.

This script demonstrates how to use the speech transcription functionality
to convert audio files to text using OpenAI's Whisper model.
"""

from pavo.perception.speech import SpeechTranscriber


def example_1_basic_usage():
    """Example 1: Basic transcription with default settings."""
    print("\n" + "="*60)
    print("Example 1: Basic Transcription")
    print("="*60)
    
    # Create a transcriber with base model
    transcriber = SpeechTranscriber(model_name='base', language='en')
    
    # Transcribe an audio file
    audio_file = '/Users/sonnguyen/Desktop/mkt/video-agent/pavo-engine-py/sample/OSR_us_000_0010_8k.wav'
    result = transcriber.transcribe(audio_file)
    
    print(f"Transcribed text: {result['text']}")
    print(f"Language: {result['language']}")
    if result['segments']:
        duration = result['segments'][-1]['end']
        print(f"Duration: {duration:.2f} seconds")


def example_2_quick_transcription():
    """Example 2: Quick transcription - get only the text."""
    print("\n" + "="*60)
    print("Example 2: Quick Text Extraction")
    print("="*60)
    
    transcriber = SpeechTranscriber()
    
    # Get only the transcribed text
    text = transcriber.transcribe_file('/Users/sonnguyen/Desktop/mkt/video-agent/pavo-engine-py/sample/OSR_us_000_0010_8k.wav')
    print(f"Text: {text}")


def example_3_language_detection():
    """Example 3: Auto-detect language."""
    print("\n" + "="*60)
    print("Example 3: Language Detection")
    print("="*60)
    
    # Don't specify language - it will auto-detect
    transcriber = SpeechTranscriber(model_name='base')
    
    result = transcriber.transcribe('/Users/sonnguyen/Desktop/mkt/video-agent/pavo-engine-py/sample/OSR_us_000_0010_8k.wav')
    
    print(f"Detected language: {result['language']}")
    print(f"Transcribed text: {result['text']}")


def example_4_context_manager():
    """Example 4: Using context manager for automatic cleanup."""
    print("\n" + "="*60)
    print("Example 4: Context Manager (Recommended)")
    print("="*60)
    
    # Model will be automatically unloaded when exiting the with block
    with SpeechTranscriber(model_name='base') as transcriber:
        text = transcriber.transcribe_file('path/to/audio.wav')
        print(f"Transcribed: {text}")
    
    print("Model automatically unloaded and memory freed!")


def example_5_batch_processing():
    """Example 5: Batch process multiple audio files."""
    print("\n" + "="*60)
    print("Example 5: Batch Processing Multiple Files")
    print("="*60)
    
    import os
    
    # Create transcriber once to reuse model
    transcriber = SpeechTranscriber(model_name='base')
    
    # Process multiple files
    audio_folder = 'path/to/audio/folder'
    
    results = {}
    for filename in os.listdir(audio_folder):
        if filename.endswith(('.wav', '.mp3', '.m4a')):
            filepath = os.path.join(audio_folder, filename)
            try:
                text = transcriber.transcribe_file(filepath)
                results[filename] = text
                print(f"✓ {filename}: {text[:50]}...")
            except Exception as e:
                print(f"✗ {filename}: {e}")
    
    # Save results
    import json
    with open('transcriptions.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Unload model to free memory
    transcriber.unload_model()


def example_6_with_segments():
    """Example 6: Get detailed transcription with segments and timestamps."""
    print("\n" + "="*60)
    print("Example 6: Detailed Transcription with Segments")
    print("="*60)
    
    transcriber = SpeechTranscriber(model_name='base')
    result = transcriber.transcribe('/Users/sonnguyen/Desktop/mkt/video-agent/pavo-engine-py/sample/OSR_us_000_0010_8k.wav')
    
    print("Full text:")
    print(result['text'])
    
    print("\nSegments with timestamps:")
    for segment in result['segments']:
        start = segment['start']
        end = segment['end']
        text = segment['text']
        print(f"[{start:.2f}s - {end:.2f}s] {text}")


def example_7_error_handling():
    """Example 7: Proper error handling."""
    print("\n" + "="*60)
    print("Example 7: Error Handling")
    print("="*60)
    
    transcriber = SpeechTranscriber()
    
    try:
        result = transcriber.transcribe('nonexistent_file.wav')
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
    except ValueError as e:
        print(f"❌ Format error: {e}")
    except RuntimeError as e:
        print(f"❌ Transcription error: {e}")


def example_8_gpu_acceleration():
    """Example 8: Use GPU for faster processing (NVIDIA)."""
    print("\n" + "="*60)
    print("Example 8: GPU Acceleration (NVIDIA)")
    print("="*60)
    
    # Use larger model with GPU for better results and speed
    transcriber = SpeechTranscriber(
        model_name='small',
        device='cuda'  # Requires NVIDIA GPU with CUDA
    )
    
    result = transcriber.transcribe('path/to/long_audio.wav')
    print(f"Transcribed (via GPU): {result['text']}")


def example_9_translation():
    """Example 9: Translate speech to English."""
    print("\n" + "="*60)
    print("Example 9: Speech Translation")
    print("="*60)
    
    # Set task to 'translate' for translation to English
    transcriber = SpeechTranscriber(
        model_name='base',
        task='translate'
    )
    
    result = transcriber.transcribe('path/to/spanish_audio.wav')
    print(f"English translation: {result['text']}")


def main():
    """Display all examples."""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  SpeechTranscriber Usage Examples".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    
    print("\nAvailable Examples:")
    print("1. Basic Transcription")
    print("2. Quick Text Extraction")
    print("3. Language Detection")
    print("4. Context Manager (Recommended)")
    print("5. Batch Processing")
    print("6. Detailed Transcription with Segments")
    print("7. Error Handling")
    print("8. GPU Acceleration")
    print("9. Speech Translation")
    
    print("\n" + "="*60)
    print("To use these examples:")
    print("="*60)
    print("\n1. Replace 'path/to/audio.wav' with your actual audio file path")
    print("2. Choose the model size:")
    print("   - 'tiny': Fast, low accuracy (75 MB)")
    print("   - 'base': Balanced (142 MB) - Recommended")
    print("   - 'small': Good accuracy (466 MB)")
    print("   - 'medium': Very good (1.5 GB)")
    print("   - 'large': Best accuracy (2.9 GB)")
    print("\n3. Uncomment the example function you want to run")
    print("4. Call the function in the __main__ block below")
    
    print("\n" + "="*60)
    print("Quick Start:")
    print("="*60)
    print("""
from pavo.perception.speech import SpeechTranscriber

# Create transcriber
transcriber = SpeechTranscriber(model_name='base')

# Transcribe audio
result = transcriber.transcribe('audio.wav')
print(result['text'])  # Print transcribed text
""")


if __name__ == '__main__':
    main()
    
    # Uncomment the example you want to run:
    # example_1_basic_usage()
    # example_2_quick_transcription()
    # example_3_language_detection()
    # example_4_context_manager()
    # example_5_batch_processing()
    example_6_with_segments()
    # example_7_error_handling()
    # example_8_gpu_acceleration()
    # example_9_translation()
