"""
Video Converter Example

This example demonstrates how to use the VideoConverter class to convert
video files from various formats to MP4.
"""

from pavo.convert import VideoConverter
import os


def example_basic_conversion():
    """Basic video conversion example."""
    print("=" * 50)
    print("Example 1: Basic Video Conversion")
    print("=" * 50)
    
    converter = VideoConverter()
    
    # Example conversion (you would replace with your actual files)
    input_file = 'input_video.avi'
    output_file = 'output_video.mp4'
    
    if os.path.exists(input_file):
        try:
            converter.convert(input_file, output_file)
            print(f"✓ Successfully converted {input_file} to {output_file}")
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
    else:
        print(f"! Input file not found: {input_file}")
        print("  To try this example, create an input_video.avi file")
    
    print()


def example_with_quality_settings():
    """Convert video with custom quality settings."""
    print("=" * 50)
    print("Example 2: Video Conversion with Quality Settings")
    print("=" * 50)
    
    # Create converter with custom settings
    converter = VideoConverter(
        codec='libx264',  # H.264 codec
        preset='medium',  # medium speed/quality trade-off
        crf=23  # medium quality (0-51, lower = better)
    )
    
    input_file = 'input_video.mov'
    output_file = 'output_quality.mp4'
    
    if os.path.exists(input_file):
        try:
            converter.convert(
                input_file,
                output_file,
                bitrate='5000k',  # 5 Mbps
                audio_bitrate='192k'
            )
            print(f"✓ Converted with quality settings")
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
    else:
        print(f"! Input file not found: {input_file}")
        print("  To try this example, create an input_video.mov file")
    
    print()


def example_with_resolution_change():
    """Convert video with resolution scaling."""
    print("=" * 50)
    print("Example 3: Video Conversion with Resolution Change")
    print("=" * 50)
    
    converter = VideoConverter(preset='fast')  # faster processing
    
    input_file = 'input_video.mkv'
    output_file = 'output_720p.mp4'
    
    if os.path.exists(input_file):
        try:
            converter.convert(
                input_file,
                output_file,
                scale='1280:720',  # Scale to 720p
                fps=30  # 30 frames per second
            )
            print(f"✓ Converted to 720p 30fps")
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
    else:
        print(f"! Input file not found: {input_file}")
        print("  To try this example, create an input_video.mkv file")
    
    print()


def example_get_video_info():
    """Get information about a video file."""
    print("=" * 50)
    print("Example 4: Get Video Information")
    print("=" * 50)
    
    converter = VideoConverter()
    
    # Use the example video in docs/media
    input_file = 'docs/media/in.mp4'
    
    if os.path.exists(input_file):
        try:
            info = converter.get_video_info(input_file)
            
            print(f"\nVideo Information for: {input_file}")
            print(f"  Resolution: {info['width']}x{info['height']} pixels")
            print(f"  FPS: {info['fps']:.2f} frames/second")
            print(f"  Duration: {info['duration']:.2f} seconds")
            print(f"  Codec: {info['codec']}")
            print(f"  Bitrate: {info['bitrate']:,} bps ({info['bitrate']/1000/1000:.2f} Mbps)")
            
        except Exception as e:
            print(f"✗ Failed to get video info: {e}")
    else:
        print(f"! Video file not found: {input_file}")
    
    print()


def example_batch_conversion():
    """Convert multiple video files in a directory."""
    print("=" * 50)
    print("Example 5: Batch Video Conversion")
    print("=" * 50)
    
    converter = VideoConverter(preset='medium')
    
    input_dir = 'videos_to_convert'
    output_dir = 'converted_videos'
    
    # Create input directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)
    
    # Check if there are any video files
    video_files = []
    for ext in converter.SUPPORTED_FORMATS:
        video_files.extend(
            f for f in os.listdir(input_dir) if f.endswith(ext)
        ) if os.path.exists(input_dir) else []
    
    if video_files:
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
        print(f"! No video files found in: {input_dir}")
        print(f"  To try this example:")
        print(f"    1. Create a '{input_dir}' directory")
        print(f"    2. Add video files in supported formats")
        print(f"    3. Run this example again")
    
    print()


def example_different_codecs():
    """Convert video using different codecs."""
    print("=" * 50)
    print("Example 6: Using Different Video Codecs")
    print("=" * 50)
    
    input_file = 'input_video.avi'
    
    if os.path.exists(input_file):
        # H.264 (libx264) - Most compatible
        print("\nConverting with H.264 (libx264)...")
        converter_h264 = VideoConverter(codec='libx264', preset='medium')
        try:
            converter_h264.convert(input_file, 'output_h264.mp4')
            print("  ✓ H.264 conversion successful")
        except Exception as e:
            print(f"  ✗ H.264 conversion failed: {e}")
        
        # H.265/HEVC (libx265) - Better compression
        print("\nConverting with H.265/HEVC (libx265)...")
        converter_h265 = VideoConverter(codec='libx265', preset='medium')
        try:
            converter_h265.convert(input_file, 'output_h265.mp4')
            print("  ✓ H.265 conversion successful")
        except Exception as e:
            print(f"  ✗ H.265 conversion failed: {e}")
    else:
        print(f"! Input file not found: {input_file}")
        print("  To try this example, create an input_video.avi file")
    
    print()


def example_error_handling():
    """Demonstrate error handling."""
    print("=" * 50)
    print("Example 7: Error Handling")
    print("=" * 50)
    
    converter = VideoConverter()
    
    # Test 1: Non-existent file
    print("\nTest 1: Non-existent file")
    try:
        converter.convert('nonexistent.avi', 'output.mp4')
    except FileNotFoundError as e:
        print(f"  ✓ Caught FileNotFoundError: {e}")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
    
    # Test 2: Unsupported format
    print("\nTest 2: Unsupported format")
    # Create a dummy unsupported file for testing
    test_file = 'test_file.xyz'
    with open(test_file, 'w') as f:
        f.write('dummy content')
    
    try:
        converter.convert(test_file, 'output.mp4')
    except ValueError as e:
        print(f"  ✓ Caught ValueError: {e}")
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)
    
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 48 + "╗")
    print("║" + " " * 12 + "VIDEO CONVERTER EXAMPLES" + " " * 13 + "║")
    print("╚" + "=" * 48 + "╝")
    print()
    
    example_basic_conversion()
    example_with_quality_settings()
    example_with_resolution_change()
    example_get_video_info()
    example_batch_conversion()
    example_different_codecs()
    example_error_handling()
    
    print("=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
