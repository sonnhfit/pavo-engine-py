# Convert Module

The `convert` module provides high-level classes for converting video and audio files between common formats using FFmpeg.

## Features

### VideoConverter
Convert video files from common formats to MP4:
- **Supported Input Formats**: AVI, MOV, MKV, FLV, WMV, WebM, M4V, MPG, MPEG, 3GP, MXF, OGV, TS, VB, F4V, ASF, RM, RMVB, MP4
- **Output Format**: MP4 (H.264 or H.265)
- **Features**:
  - Configurable video codec (libx264, libx265, etc.)
  - Configurable audio codec (AAC, MP3, etc.)
  - Quality control via bitrate or CRF (Constant Rate Factor)
  - FPS adjustment
  - Video scaling/resolution conversion
  - Batch conversion support
  - Get video information (resolution, FPS, duration, codec, bitrate)

### AudioConverter
Convert audio files to MP3 or WAV format:
- **Supported Input Formats**: MP3, WAV, FLAC, OGG, M4A, AAC, WMA, APE, ALAC, Opus, WebM, AIFF, AU, Raw, Vorbis, AMR, GSM, AC3, EAC3
- **Output Formats**: MP3, WAV
- **Features**:
  - Configurable output format (MP3 or WAV)
  - Bitrate control for MP3 (128k, 192k, 256k, 320k)
  - Sample rate conversion
  - Channel conversion (mono, stereo)
  - Batch conversion support
  - Get audio information (duration, sample rate, channels, codec, bitrate)

## Installation

Make sure FFmpeg is installed on your system:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (with Chocolatey)
choco install ffmpeg
```

## Usage

### Video Conversion

#### Basic Conversion
```python
from pavo.convert import VideoConverter

# Create converter
converter = VideoConverter()

# Convert a single video file
converter.convert('input.avi', 'output.mp4')
```

#### Advanced Options
```python
from pavo.convert import VideoConverter

converter = VideoConverter(preset='fast')  # faster encoding
converter.convert(
    'input.mov',
    'output.mp4',
    bitrate='5000k',  # 5 Mbps
    audio_bitrate='192k',
    fps=30,
    scale='1280:720'
)
```

#### Get Video Information
```python
from pavo.convert import VideoConverter

converter = VideoConverter()
info = converter.get_video_info('input.avi')

print(f"Resolution: {info['width']}x{info['height']}")
print(f"FPS: {info['fps']}")
print(f"Duration: {info['duration']} seconds")
print(f"Codec: {info['codec']}")
print(f"Bitrate: {info['bitrate']} bps")
```

#### Batch Conversion
```python
from pavo.convert import VideoConverter

converter = VideoConverter(preset='medium')

result = converter.batch_convert(
    input_dir='/path/to/videos',
    output_dir='/path/to/output',
    recursive=True
)

print(f"Converted: {result['successful']}/{result['total']}")
if result['failed'] > 0:
    for filename, error in result['errors']:
        print(f"Failed: {filename} - {error}")
```

### Audio Conversion

#### Basic Conversion
```python
from pavo.convert import AudioConverter

# Convert to MP3 (default)
converter = AudioConverter(output_format='mp3', bitrate='192k')
converter.convert('input.flac', 'output.mp3')

# Or convert to WAV
converter_wav = AudioConverter(output_format='wav')
converter_wav.convert('input.aac', 'output.wav')
```

#### With Sample Rate and Channel Conversion
```python
from pavo.convert import AudioConverter

converter = AudioConverter(output_format='mp3', bitrate='320k')
result_path = converter.convert(
    'input.flac',
    'output.mp3',
    sample_rate=44100,  # 44.1 kHz
    channels=2  # Stereo
)
```

#### Get Audio Information
```python
from pavo.convert import AudioConverter

converter = AudioConverter()
info = converter.get_audio_info('input.flac')

print(f"Duration: {info['duration']} seconds")
print(f"Sample Rate: {info['sample_rate']} Hz")
print(f"Channels: {info['channels']}")
print(f"Codec: {info['codec']}")
print(f"Bitrate: {info['bitrate']} bps")
```

#### Batch Conversion
```python
from pavo.convert import AudioConverter

converter = AudioConverter(output_format='mp3', bitrate='192k')

result = converter.batch_convert(
    input_dir='/path/to/audio',
    output_dir='/path/to/output',
    recursive=False
)

print(f"Converted: {result['successful']}/{result['total']}")
if result['errors']:
    for filename, error in result['errors']:
        print(f"Error: {filename} - {error}")
```

#### Change Output Format
```python
from pavo.convert import AudioConverter

converter = AudioConverter(output_format='mp3', bitrate='192k')

# Convert to MP3
converter.convert('input.flac', 'output.mp3')

# Switch to WAV format
converter.set_output_format('wav')
converter.convert('input2.aac', 'output2.wav')

# Adjust MP3 bitrate
converter.set_output_format('mp3')
converter.set_bitrate('320k')
converter.convert('input3.flac', 'output3.mp3')
```

## API Reference

### VideoConverter

#### Constructor
```python
VideoConverter(
    codec='libx264',           # 'libx264', 'libx265', 'mpeg4', 'libvpx-vp9'
    audio_codec='aac',         # 'aac', 'libmp3lame', 'libvorbis', 'flac'
    preset='medium',           # 'ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'
    crf=23                     # 0-51, lower = better quality
)
```

#### Methods
- `convert(input_path, output_path, bitrate=None, audio_bitrate=None, fps=None, scale=None, verbose=True)` -> bool
- `get_video_info(input_path)` -> Dict[str, Any]
- `batch_convert(input_dir, output_dir, recursive=False, verbose=True)` -> Dict[str, Any]

### AudioConverter

#### Constructor
```python
AudioConverter(
    output_format='mp3',       # 'mp3' or 'wav'
    bitrate='192k'             # '128k', '192k', '256k', '320k' for MP3
)
```

#### Methods
- `convert(input_path, output_path=None, sample_rate=None, channels=None, verbose=True)` -> str
- `get_audio_info(input_path)` -> Dict[str, Any]
- `batch_convert(input_dir, output_dir, recursive=False, verbose=True)` -> Dict[str, Any]
- `set_output_format(output_format)` -> None
- `set_bitrate(bitrate)` -> None

## Error Handling

Both converters raise descriptive exceptions:
- `FileNotFoundError`: If input file doesn't exist
- `ValueError`: If unsupported format is used
- `RuntimeError`: If FFmpeg conversion fails
- `ImportError`: If FFmpeg is not installed

```python
from pavo.convert import VideoConverter

converter = VideoConverter()

try:
    converter.convert('input.avi', 'output.mp4')
except FileNotFoundError:
    print("Input file not found")
except ValueError as e:
    print(f"Unsupported format: {e}")
except RuntimeError as e:
    print(f"Conversion failed: {e}")
```

## Notes

- FFmpeg must be installed and accessible from command line
- Large files may take considerable time to convert
- CRF values (0-51): 0 = lossless, 23 = default (medium quality), 51 = worst quality
- Batch conversion processes files sequentially
- Progress bars are shown for batch operations with `verbose=True`
