"""
Audio Converter Example

This example demonstrates how to use the AudioConverter class to convert
audio files from various formats to MP3 or WAV.
"""

from pavo.convert import AudioConverter
import os


def example_basic_mp3_conversion():
    """Basic audio conversion to MP3."""
    print("=" * 50)
    print("Example 1: Basic Audio to MP3 Conversion")
    print("=" * 50)
    
    converter = AudioConverter(output_format='mp3', bitrate='192k')
    
    input_file = 'input_audio.flac'
    output_file = 'output_audio.mp3'
    
    if os.path.exists(input_file):
        try:
            result = converter.convert(input_file, output_file)
            print(f"✓ Successfully converted to: {result}")
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
    else:
        print(f"! Input file not found: {input_file}")
        print("  To try this example, create an input_audio.flac file")
    
    print()


def example_basic_wav_conversion():
    """Basic audio conversion to WAV."""
    print("=" * 50)
    print("Example 2: Basic Audio to WAV Conversion")
    print("=" * 50)
    
    converter = AudioConverter(output_format='wav')
    
    input_file = 'input_audio.aac'
    output_file = 'output_audio.wav'
    
    if os.path.exists(input_file):
        try:
            result = converter.convert(input_file, output_file)
            print(f"✓ Successfully converted to: {result}")
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
    else:
        print(f"! Input file not found: {input_file}")
        print("  To try this example, create an input_audio.aac file")
    
    print()


def example_with_sample_rate_conversion():
    """Convert audio with sample rate and channel conversion."""
    print("=" * 50)
    print("Example 3: Audio Conversion with Sample Rate & Channels")
    print("=" * 50)
    
    converter = AudioConverter(output_format='mp3', bitrate='320k')
    
    input_file = 'input_audio.flac'
    output_file = 'output_audio_converted.mp3'
    
    if os.path.exists(input_file):
        try:
            result = converter.convert(
                input_file,
                output_file,
                sample_rate=44100,  # Convert to 44.1 kHz
                channels=2  # Stereo
            )
            print(f"✓ Converted with sample rate 44.1kHz, 2 channels")
            print(f"  Output: {result}")
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
    else:
        print(f"! Input file not found: {input_file}")
        print("  To try this example, create an input_audio.flac file")
    
    print()


def example_get_audio_info():
    """Get information about an audio file."""
    print("=" * 50)
    print("Example 4: Get Audio Information")
    print("=" * 50)
    
    converter = AudioConverter()
    
    # Use the sample audio file
    input_file = 'sample/OSR_us_000_0010_8k.wav'
    
    if os.path.exists(input_file):
        try:
            info = converter.get_audio_info(input_file)
            
            print(f"\nAudio Information for: {input_file}")
            print(f"  Duration: {info['duration']:.2f} seconds")
            print(f"  Sample Rate: {info['sample_rate']} Hz ({info['sample_rate']/1000:.1f} kHz)")
            print(f"  Channels: {info['channels']} ({'Mono' if info['channels'] == 1 else 'Stereo' if info['channels'] == 2 else 'Multi-channel'})")
            print(f"  Codec: {info['codec']}")
            print(f"  Bitrate: {info['bitrate']:,} bps ({info['bitrate']/1000:.1f} kbps)")
            
        except Exception as e:
            print(f"✗ Failed to get audio info: {e}")
    else:
        print(f"! Audio file not found: {input_file}")
    
    print()


def example_batch_conversion():
    """Convert multiple audio files in a directory."""
    print("=" * 50)
    print("Example 5: Batch Audio Conversion")
    print("=" * 50)
    
    converter = AudioConverter(output_format='mp3', bitrate='192k')
    
    input_dir = 'audio_to_convert'
    output_dir = 'converted_audio'
    
    # Create input directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)
    
    # Check if there are any audio files
    audio_files = []
    for ext in converter.SUPPORTED_FORMATS:
        audio_files.extend(
            f for f in os.listdir(input_dir) if f.endswith(ext)
        ) if os.path.exists(input_dir) else []
    
    if audio_files:
        try:
            result = converter.batch_convert(
                input_dir,
                output_dir,
                recursive=False,
                verbose=True
            )
            
            print(f"\nBatch Conversion Results:")
            print(f"  Total files: {result['total']}")
            print(f"  Successful: {result['successful']}")
            print(f"  Failed: {result['failed']}")
            
            if result['errors']:
                print(f"\n  Errors:")
                for filename, error in result['errors']:
                    print(f"    - {filename}: {error}")
            
        except Exception as e:
            print(f"✗ Batch conversion failed: {e}")
    else:
        print(f"! No audio files found in: {input_dir}")
        print(f"  To try this example:")
        print(f"    1. Create an '{input_dir}' directory")
        print(f"    2. Add audio files in supported formats")
        print(f"    3. Run this example again")
    
    print()


def example_change_output_format():
    """Demonstrate changing output format dynamically."""
    print("=" * 50)
    print("Example 6: Dynamic Output Format Change")
    print("=" * 50)
    
    converter = AudioConverter(output_format='mp3', bitrate='192k')
    
    input_file = 'input_audio.flac'
    
    if os.path.exists(input_file):
        try:
            # Convert to MP3
            print("\nConverting to MP3...")
            result_mp3 = converter.convert(input_file, 'output_audio.mp3')
            print(f"  ✓ Created: {result_mp3}")
            
            # Change to WAV format
            print("\nChanging format to WAV...")
            converter.set_output_format('wav')
            result_wav = converter.convert(input_file, 'output_audio.wav')
            print(f"  ✓ Created: {result_wav}")
            
            # Change MP3 bitrate and convert again
            print("\nChanging MP3 bitrate to 320k...")
            converter.set_output_format('mp3')
            converter.set_bitrate('320k')
            result_mp3_hq = converter.convert(input_file, 'output_audio_hq.mp3')
            print(f"  ✓ Created: {result_mp3_hq}")
            
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
    else:
        print(f"! Input file not found: {input_file}")
        print("  To try this example, create an input_audio.flac file")
    
    print()


def example_mp3_bitrate_options():
    """Demonstrate different MP3 bitrate options."""
    print("=" * 50)
    print("Example 7: MP3 Bitrate Options")
    print("=" * 50)
    
    input_file = 'input_audio.flac'
    
    if os.path.exists(input_file):
        bitrates = ['128k', '192k', '256k', '320k']
        
        for bitrate in bitrates:
            print(f"\nConverting to MP3 with bitrate {bitrate}...")
            converter = AudioConverter(output_format='mp3', bitrate=bitrate)
            
            try:
                output_file = f'output_audio_{bitrate.replace("k", "kbps")}.mp3'
                result = converter.convert(input_file, output_file)
                print(f"  ✓ Created: {result}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    else:
        print(f"! Input file not found: {input_file}")
        print("  To try this example, create an input_audio.flac file")
    
    print()


def example_mono_conversion():
    """Convert stereo audio to mono."""
    print("=" * 50)
    print("Example 8: Stereo to Mono Conversion")
    print("=" * 50)
    
    converter = AudioConverter(output_format='mp3', bitrate='128k')
    
    input_file = 'input_audio_stereo.flac'
    output_file = 'output_audio_mono.mp3'
    
    if os.path.exists(input_file):
        try:
            result = converter.convert(
                input_file,
                output_file,
                channels=1  # Mono
            )
            print(f"✓ Converted stereo to mono: {result}")
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
    else:
        print(f"! Input file not found: {input_file}")
        print("  To try this example, create an input_audio_stereo.flac file")
    
    print()


def example_error_handling():
    """Demonstrate error handling."""
    print("=" * 50)
    print("Example 9: Error Handling")
    print("=" * 50)
    
    converter = AudioConverter()
    
    # Test 1: Non-existent file
    print("\nTest 1: Non-existent file")
    try:
        converter.convert('nonexistent.flac', 'output.mp3')
    except FileNotFoundError as e:
        print(f"  ✓ Caught FileNotFoundError: {e}")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
    
    # Test 2: Unsupported format
    print("\nTest 2: Unsupported audio format")
    test_file = 'test_audio.xyz'
    with open(test_file, 'w') as f:
        f.write('dummy content')
    
    try:
        converter.convert(test_file, 'output.mp3')
    except ValueError as e:
        print(f"  ✓ Caught ValueError: {e}")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
    
    # Test 3: Invalid output format
    print("\nTest 3: Invalid output format")
    try:
        AudioConverter(output_format='invalid')
    except ValueError as e:
        print(f"  ✓ Caught ValueError: {e}")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
    
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 48 + "╗")
    print("║" + " " * 12 + "AUDIO CONVERTER EXAMPLES" + " " * 12 + "║")
    print("╚" + "=" * 48 + "╝")
    print()
    
    example_basic_mp3_conversion()
    example_basic_wav_conversion()
    example_with_sample_rate_conversion()
    example_get_audio_info()
    example_batch_conversion()
    example_change_output_format()
    example_mp3_bitrate_options()
    example_mono_conversion()
    example_error_handling()
    
    print("=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
