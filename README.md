# 🎬 Pavo Engine - AI-Powered Video Editing Agent

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Required-orange.svg)](https://ffmpeg.org/)

**Pavo Engine** is an intelligent AI agent for automated video editing that transforms natural language prompts into professionally edited videos. Built with a modular 4-layer architecture, it combines computer vision, speech processing, and timeline automation to create dynamic video content.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sonnhfit/pavo-engine-py.git
cd pavo-engine-py

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required)
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (using Chocolatey)
choco install ffmpeg
```

### Basic Usage

```python
from pavo import render_video

# Render a video from JSON timeline
render_video(
    'docs/data.json',          # Timeline configuration
    'output/video.mp4'         # Output file
)
```

## 🏗️ Architecture Overview

Pavo Engine follows a sophisticated 4-layer architecture:

### Layer 1: Perception
- **Video Understanding**: Scene detection, object recognition, face detection
- **Audio Processing**: Speech-to-text, speaker diarization, emotion detection
- **Content Analysis**: Motion tracking, visual saliency, semantic segmentation

### Layer 2: Planner (The Brain)
- **LLM Director**: Converts natural language prompts into edit plans
- **Edit Planning**: Generates JSON-based timeline specifications
- **Layout Engine**: Automatically positions overlays using smart algorithms

### Layer 3: Timeline Engine
- **Multi-track Editing**: Supports parallel video, image, and text tracks
- **Transition Effects**: Fade, slide, zoom, and custom animations
- **Timeline Management**: Frame-accurate synchronization

### Layer 4: Execution Engine
- **FFmpeg Integration**: High-performance video rendering
- **Cloud Asset Management**: S3 integration for remote resources
- **Real-time Processing**: Optimized for speed and efficiency

## 📋 Core Features

### 🎥 Video Processing
- **Format Conversion**: Convert between 50+ video formats
- **Resolution Scaling**: Automatic upscaling/downscaling
- **Frame Rate Adjustment**: Smooth FPS conversion
- **Quality Optimization**: Bitrate and codec optimization

### 🔊 Audio Processing
- **Speech Transcription**: Whisper-based accurate transcription
- **Speaker Diarization**: Identify different speakers
- **Audio Conversion**: Format and quality conversion
- **Background Music**: Intelligent soundtrack integration

### 🖼️ Image & Text Overlays
- **Smart Positioning**: Automatic empty space detection
- **Dynamic Layout**: Responsive positioning algorithms
- **Text Rendering**: Custom fonts, colors, and animations
- **Image Effects**: Filters, transitions, and transformations

### ☁️ Cloud Integration
- **S3 Support**: Direct asset loading from cloud storage
- **CDN Ready**: Optimized for distributed content delivery
- **Batch Processing**: Parallel video rendering
- **Progress Tracking**: Real-time status updates

## 📊 Timeline JSON Format

Pavo Engine uses a structured JSON format to define video timelines:

```json
{
  "timeline": {
    "n_frames": 30,
    "soundtrack": {
      "src": "https://example.com/music.mp3",
      "effect": "fadeOut"
    },
    "background": "#000000",
    "tracks": [
      {
        "track_id": 0,
        "strips": [
          {
            "asset": {
              "type": "image",
              "src": "path/to/image.jpg"
            },
            "start": 0,
            "length": 5,
            "effect": "zoomIn",
            "transition": {
              "in": "fade",
              "out": "fade"
            }
          }
        ]
      }
    ]
  },
  "output": {
    "format": "mp4",
    "resolution": "sd",
    "fps": 30,
    "width": 640,
    "height": 1080
  }
}
```

## 🎯 Use Cases

### 1. **Social Media Content Creation**
- Automatically edit TikTok/Instagram/YouTube shorts
- Add captions, effects, and transitions
- Resize for different platform requirements

### 2. **Educational Videos**
- Add text overlays to lecture recordings
- Insert relevant images and diagrams
- Create highlight reels from long content

### 3. **Marketing & Advertising**
- Generate product demo videos
- Create promotional content from raw footage
- Add branding and call-to-action elements

### 4. **Personal Video Editing**
- Automatically edit vacation videos
- Create montages with music synchronization
- Add subtitles for accessibility

## 🔧 Advanced Usage

### Using S3 for Asset Management

```python
from pavo import json_render_with_s3_asset

S3_BUCKET_RESOURCE = 'your-bucket-name'
S3_ACCESS_KEY = 'your-access-key'
S3_SECRET_KEY = 'your-secret-key'

output = json_render_with_s3_asset(
    'timeline.json',
    'output/video.mp4',
    S3_BUCKET_RESOURCE,
    S3_ACCESS_KEY,
    S3_SECRET_KEY
)
```

### Custom Video Conversion

```python
from pavo.convert import VideoConverter

converter = VideoConverter(
    codec='libx264',
    preset='medium',
    crf=23
)

# Convert with custom settings
converter.convert(
    'input.mov',
    'output.mp4',
    scale='1920:1080',
    fps=30,
    bitrate='5000k'
)
```

### Speech Processing

```python
from pavo.perception.speech import Transcriber

transcriber = Transcriber()
transcription = transcriber.transcribe('audio.wav')

from pavo.perception.speech import Diarization
diarization = Diarization()
speakers = diarization.process('audio.wav')
```

## 🧠 Smart Positioning Algorithms

Pavo Engine includes advanced algorithms for automatic overlay positioning:

### 1. **Spatiotemporal Heatmap + Sliding Window** (Recommended for MVP)
- Combines multiple frames into temporal occupancy heatmaps
- Uses sliding window to find optimal empty spaces
- Considers motion patterns and semantic zones
- Optimized with integral images for O(1) mean calculation

### 2. **Maximal Empty Rectangle (MER)**
- Classic computational geometry approach
- Finds largest empty rectangle in binary masks
- Efficient O(n³) algorithms available

### 3. **Rectangle Packing with Forbidden Regions**
- Treats positioning as rectangle packing problem
- Uses simulated annealing and genetic algorithms
- Maximizes score while avoiding occupied regions

### 4. **Dynamic Programming Over Time**
- Considers temporal consistency
- Uses Viterbi algorithm for smooth transitions
- Minimizes position changes between frames

## 📁 Project Structure

```
pavo-engine-py/
├── pavo/                    # Core engine
│   ├── __init__.py         # Main exports
│   ├── pavo.py             # Primary rendering functions
│   ├── const.py            # Constants and configuration
│   ├── convert/            # Audio/video conversion
│   ├── perception/         # Video understanding
│   │   ├── object/         # Object detection
│   │   ├── scene/          # Scene detection
│   │   ├── speech/         # Speech processing
│   │   └── emotion/        # Emotion detection
│   ├── planner/            # LLM planning layer
│   ├── preparation/        # Asset preparation
│   └── sequancer/          # Timeline sequencing
├── docs/                   # Documentation and examples
│   ├── example/           # Usage examples
│   ├── experiments/       # Experimental code
│   └── media/            # Sample media files
├── notebook/              # Jupyter notebooks
├── sample/               # Sample audio files
├── tests/                # Test suite
├── requirements.txt      # Python dependencies
├── setup.py             # Package configuration
└── main.py              # Entry point examples
```

## 🛠️ Installation Details

### System Requirements
- **Python**: 3.9 or higher
- **FFmpeg**: Latest version
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 1GB free space

### Python Dependencies

```bash
# Core dependencies
ffmpeg-python==0.2.0      # FFmpeg integration
requests==2.32.3          # HTTP requests
retrying==1.3.4           # Retry logic
tqdm==4.66.4              # Progress bars
pillow==10.4.0            # Image processing

# Cloud integration
boto3==1.26.109           # AWS S3
botocore==1.29.109        # AWS core

# AI/ML capabilities
openai-whisper>=20230314  # Speech recognition
ultralytics               # Object detection
scenedetect[opencv]>=0.6.2 # Scene detection
```

### Development Installation

```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Build package
python setup.py sdist bdist_wheel
```

## 📚 Examples

Check the `docs/example/` directory for comprehensive examples:

- `VIDEO_CONVERTER_EXAMPLE.py` - Video format conversion
- `AUDIO_CONVERTER_EXAMPLE.py` - Audio processing
- `SPEECH_TRANSCRIBER_EXAMPLE.py` - Speech-to-text
- `SPEAKER_DIARIZATION_EXAMPLE.py` - Speaker identification
- `OBJECT_DETECTION_EXAMPLE.py` - Object recognition
- `SCENE_DETECTION_EXAMPLE.py` - Scene change detection

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_render.py

# Run with coverage
python -m pytest --cov=pavo tests/
```

## 🔍 Performance Optimization

### Rendering Optimization
- **Parallel Processing**: Multi-core FFmpeg encoding
- **Memory Management**: Efficient asset caching
- **Progressive Loading**: Stream large files
- **GPU Acceleration**: CUDA support for compatible hardware

### Algorithm Optimization
- **Integral Images**: O(1) mean calculation for heatmaps
- **Caching**: Reuse computed results
- **Lazy Evaluation**: Compute only when needed
- **Batch Processing**: Process multiple frames simultaneously

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Run tests**
   ```bash
   python -m pytest tests/
   ```
5. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
6. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
7. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation accordingly

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FFmpeg Team** for the incredible multimedia framework
- **OpenAI** for the Whisper speech recognition model
- **Ultralytics** for YOLO object detection
- **SceneDetect** for scene change detection

## 📞 Support & Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/sonnhfit/pavo-engine-py/issues)
- **Email**: sonnhfit@gmail.com
- **Documentation**: [Full documentation](docs/)

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=sonnhfit/pavo-engine-py&type=Date)](https://star-history.com/#sonnhfit/pavo-engine-py&Date)

---

**Pavo Engine** - Transforming how we create video content with AI. From simple edits to complex productions, automate your video workflow with intelligent algorithms and cloud-scale processing.

*"Edit videos with words, not timelines."*