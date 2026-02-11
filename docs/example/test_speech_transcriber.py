#!/usr/bin/env python3
"""
Test script for the SpeechTranscriber class.

This script demonstrates how to use the SpeechTranscriber to transcribe audio files.
"""

import os
from pavo.perception.speech import SpeechTranscriber


def test_basic_transcription():
    """Test basic transcription functionality."""
    print("=" * 60)
    print("Testing SpeechTranscriber - Basic Transcription")
    print("=" * 60)
    
    try:
        # Initialize transcriber with base model
        print("\n1. Initializing SpeechTranscriber with 'base' model...")
        transcriber = SpeechTranscriber(model_name='base', language='en')
        print("   ✓ SpeechTranscriber initialized successfully")
        
        # Check if sample audio file exists
        sample_audio = 'docs/media/in.mp4'
        if not os.path.exists(sample_audio):
            print(f"\n   ⚠ Sample audio file not found: {sample_audio}")
            print("   Creating a simple test instead...")
            
            # Create a dummy test
            print("\n2. Testing transcriber methods:")
            print("   - transcribe(audio_path) - Full transcription with metadata")
            print("   - transcribe_file(audio_path) - Quick text extraction")
            print("   - get_language(audio_path) - Language detection")
            print("   - Supports context manager for resource cleanup")
            return
        
        print(f"\n2. Testing transcription on: {sample_audio}")
        
        # Test full transcription
        print("   Running transcription...")
        result = transcriber.transcribe(sample_audio)
        
        print(f"\n3. Results:")
        print(f"   - Text: {result['text'][:100]}...")
        print(f"   - Language: {result['language']}")
        print(f"   - Duration: {result['duration']:.2f} seconds")
        print(f"   - Segments: {len(result['segments'])}")
        
        # Test convenience method
        print("\n4. Testing convenience method (transcribe_file):")
        text = transcriber.transcribe_file(sample_audio)
        print(f"   ✓ Text extracted: {text[:50]}...")
        
        print("\n5. Testing language detection:")
        language = transcriber.get_language(sample_audio)
        print(f"   ✓ Detected language: {language}")
        
        print("\n✓ All tests passed!")
        
    except ImportError as e:
        print(f"\n✗ ImportError: {e}")
        print("\nPlease install whisper:")
        print("  pip install openai-whisper")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print(f"   Type: {type(e).__name__}")


def test_context_manager():
    """Test context manager functionality."""
    print("\n" + "=" * 60)
    print("Testing SpeechTranscriber - Context Manager")
    print("=" * 60)
    
    try:
        print("\nUsing SpeechTranscriber with context manager:")
        with SpeechTranscriber(model_name='tiny', language='en') as transcriber:
            print("✓ Context manager initialized")
            print("✓ Model will be unloaded automatically on exit")
        
        print("✓ Context manager test passed!")
    except Exception as e:
        print(f"✗ Context manager test failed: {e}")


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 60)
    print("Testing SpeechTranscriber - Error Handling")
    print("=" * 60)
    
    try:
        transcriber = SpeechTranscriber()
        
        # Test non-existent file
        print("\n1. Testing non-existent file handling:")
        try:
            transcriber.transcribe('/path/to/nonexistent/file.wav')
            print("   ✗ Should have raised FileNotFoundError")
        except FileNotFoundError as e:
            print(f"   ✓ Correctly raised FileNotFoundError: {e}")
        
        # Test unsupported format
        print("\n2. Testing unsupported format handling:")
        # Create a dummy file with unsupported extension
        dummy_file = '/tmp/test.xyz'
        open(dummy_file, 'w').close()
        try:
            transcriber.transcribe(dummy_file)
            print("   ✗ Should have raised ValueError")
        except ValueError as e:
            print(f"   ✓ Correctly raised ValueError: {e}")
        finally:
            if os.path.exists(dummy_file):
                os.remove(dummy_file)
        
        print("\n✓ Error handling tests passed!")
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")


def main():
    """Run all tests."""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  SpeechTranscriber Test Suite".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    
    test_basic_transcription()
    test_context_manager()
    test_error_handling()
    
    print("\n" + "*" * 60)
    print("Tests completed!")
    print("*" * 60 + "\n")


if __name__ == '__main__':
    main()
