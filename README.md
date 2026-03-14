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
    'docs/data.json',      # Timeline configuration (JSON path)
    'output/video.mp4'     # Output MP4 file
)
```

### Sample Timeline JSON

Save the following as `timeline.json` and pass its path to `render_video`:

```json
{
  "timeline": {
    "n_frames": 75,
    "background": "#1a1a2e",
    "soundtrack": {
      "src": "path/to/background_music.mp3",
      "effect": "fadeOut"
    },
    "tracks": [
      {
        "track_id": 0,
        "strips": [
          {
            "asset": {
              "type": "image",
              "src": "path/to/intro.jpg"
            },
            "start": 0,
            "video_start_frame": 0,
            "length": 25,
            "effect": "zoomIn",
            "transition": {"in": "fade", "out": "fade"}
          },
          {
            "asset": {
              "type": "video",
              "src": "path/to/clip.mp4"
            },
            "start": 25,
            "video_start_frame": 0,
            "length": 50,
            "effect": null,
            "transition": {"in": "fade", "out": "fade"}
          }
        ]
      }
    ]
  },
  "output": {
    "format": "mp4",
    "fps": 25,
    "width": 1280,
    "height": 720
  }
}
```

#### Audio Ducking

Enable the `audio_ducking` flag in the `output` section to automatically lower
the `soundtrack` volume whenever speech is detected in video clips:

```json
{
  "timeline": {
    "n_frames": 75,
    "background": "#1a1a2e",
    "soundtrack": {
      "src": "path/to/background_music.mp3"
    },
    "tracks": [
      {
        "track_id": 0,
        "strips": [
          {
            "asset": {
              "type": "video",
              "src": "path/to/narration.mp4"
            },
            "start": 0,
            "video_start_frame": 0,
            "length": 75,
            "effect": null,
            "transition": {}
          }
        ]
      }
    ]
  },
  "output": {
    "format": "mp4",
    "fps": 25,
    "width": 1280,
    "height": 720,
    "audio_ducking": true,
    "ducking_reduction_db": 10
  }
}
```

```python
from pavo import render_video

render_video('timeline.json', 'output/video.mp4')
```

| `output` field | Type | Default | Description |
|---|---|---|---|
| `audio_ducking` | `bool` | `false` | Enable automatic volume ducking of `soundtrack` during detected speech segments. |
| `ducking_reduction_db` | `float` | `10.0` | Volume reduction (in dB) applied to the soundtrack during speech. Requires `openai-whisper`. |

#### Audio Tracks

In addition to a global `soundtrack`, you can attach audio assets directly to individual strips
by setting `"type": "audio"` in the asset definition.  Each audio strip is mixed with the
global soundtrack (if any) using FFmpeg `adelay` + `amix` filters, so you can layer
voice‑overs, sound effects, or per‑segment music without pre‑mixing your files.

**Required field:** `src`  
**Optional fields:** `volume` (float, default `1.0`), `start` (frame offset within the strip)

```json
{
  "timeline": {
    "n_frames": 75,
    "background": "#1a1a2e",
    "soundtrack": {
      "src": "path/to/background_music.mp3",
      "effect": "fadeOut"
    },
    "tracks": [
      {
        "track_id": 0,
        "strips": [
          {
            "asset": {"type": "image", "src": "path/to/intro.jpg"},
            "start": 0,
            "video_start_frame": 0,
            "length": 25,
            "effect": "zoomIn",
            "transition": {"in": "fade", "out": "fade"}
          }
        ]
      },
      {
        "track_id": 1,
        "strips": [
          {
            "asset": {
              "type": "audio",
              "src": "path/to/voiceover.mp3",
              "volume": 0.9,
              "start": 5
            },
            "start": 0,
            "video_start_frame": 0,
            "length": 50,
            "effect": null,
            "transition": {}
          }
        ]
      }
    ]
  },
  "output": {
    "format": "mp4",
    "fps": 25,
    "width": 1280,
    "height": 720
  }
}
```

| Audio asset field | Type | Required | Default | Description |
|---|---|---|---|---|
| `src` | `string` | ✅ | — | Path to the audio file (MP3, WAV, AAC, etc.). |
| `volume` | `float` | ❌ | `1.0` | Playback volume (e.g. `0.5` for half volume, `1.5` for 50 % louder). |
| `start` | `number` | ❌ | `0` | Frame offset within the audio asset where playback begins. Added to the strip's timeline `start` position to determine the absolute video position in milliseconds. |

> **Note:** Audio strips are mixed with the global `soundtrack` (if present) using
> FFmpeg's `amix` filter with `normalize=0` so that individual volumes are
> preserved rather than being normalised to fit a single output level.
> Audio ducking (if enabled) applies only to the global `soundtrack`.

#### Error handling

`render_video` raises descriptive exceptions for common mistakes:

```python
from pavo import render_video

try:
    render_video('timeline.json', 'output/video.mp4')
except FileNotFoundError as exc:
    print(f"JSON file missing: {exc}")
except ValueError as exc:
    print(f"Invalid timeline JSON: {exc}")
```

## 🗂️ JSON Schema

Pavo Engine validates every timeline file against a strict [Pydantic](https://docs.pydantic.dev/) schema before rendering begins.  Any missing or invalid field raises a `ValueError` with a clear, human-readable message that lists every problem found.

### Validation rules at a glance

| Location | Field | Type | Required | Constraint |
|---|---|---|---|---|
| root | `timeline` | object | ✅ | — |
| root | `output` | object | ❌ | — |
| `timeline` | `n_frames` | int | ✅ | ≥ 0 |
| `timeline` | `background` | string | ❌ | must start with `#` |
| `timeline` | `tracks` | array | ❌ | defaults to `[]` |
| `timeline` | `soundtrack` | object | ❌ | — |
| `soundtrack` | `src` | string | ✅ | — |
| `soundtrack` | `effect` | string | ❌ | e.g. `"fadeOut"` |
| `tracks[*]` | `track_id` | int | ✅ | ≥ 0 |
| `tracks[*]` | `strips` | array | ✅ | ≥ 1 item |
| `strips[*]` | `asset` | object | ✅ | — |
| `strips[*]` | `start` | int | ✅ | ≥ 0 |
| `strips[*]` | `length` | int | ✅ | ≥ 1 |
| `strips[*]` | `video_start_frame` | int | ❌ | ≥ 0, default `0` |
| `strips[*]` | `transition` | object | ❌ | — |
| `asset` | `type` | string | ✅ | `"image"` \| `"video"` \| `"text"` |
| `asset` | `src` | string | ✅ for image/video | file path |
| `asset` | `content` | string | ✅ for text | non-empty |
| `asset` | `size` | int | ❌ | ≥ 1, default `24` |
| `asset` | `color` | string | ❌ | default `"white"` |
| `asset` | `position.x/y` | number \| `"center"` | ❌ | default `0` |
| `transition` | `in` | string | ❌ | `fade`\|`slide`\|`wipe`\|`dissolve` |
| `transition` | `out` | string | ❌ | `fade`\|`slide`\|`wipe`\|`dissolve` |
| `transition` | `duration` | int | ❌ | ≥ 1, default `5` |
| `output` | `fps` | float | ❌ | > 0, default `25` |
| `output` | `width` | int | ❌ | ≥ 1 |
| `output` | `height` | int | ❌ | ≥ 1 |
| `output` | `audio_ducking` | bool | ❌ | default `false` |
| `output` | `ducking_reduction_db` | float | ❌ | default `10.0` |

### Programmatic access

You can also validate a timeline dict directly without rendering:

```python
from pavo.schema import validate_timeline_json

data = {
    "timeline": {
        "n_frames": 25,
        "background": "#1a1a2e",
        "tracks": [
            {
                "track_id": 0,
                "strips": [
                    {
                        "asset": {"type": "image", "src": "intro.jpg"},
                        "start": 0,
                        "length": 25,
                    }
                ],
            }
        ],
    },
    "output": {"fps": 25, "width": 1280, "height": 720},
}

model = validate_timeline_json(data)   # raises ValueError on failure
print(model.timeline.n_frames)         # 25
```


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

### Transition Effects

Each strip can specify `"in"` and `"out"` transitions that are applied when the strip enters or exits the composition. The `"duration"` field controls how many frames the transition lasts (default: `5`).

| `transition.type` | Description | FFmpeg filter |
|---|---|---|
| `fade` | Gradually modulates the strip's alpha channel from 0 → 1 (in) or 1 → 0 (out). | `format=rgba`, `colorchannelmixer`, `overlay` |
| `slide` | Translates the strip in from the left edge (in) or out to the right edge (out). | `overlay` with computed `x` expression |
| `wipe` | Progressively reveals the strip left-to-right (in) or hides it (out) via a crop mask. | `crop`, `overlay` |
| `dissolve` | Cross-blends the strip with the base image using the `blend` filter. | `blend` |

```json
{
  "timeline": {
    "n_frames": 50,
    "background": "#000000",
    "tracks": [
      {
        "track_id": 0,
        "strips": [
          {
            "asset": {"type": "image", "src": "path/to/intro.jpg"},
            "start": 0,
            "video_start_frame": 0,
            "length": 25,
            "effect": "zoomIn",
            "transition": {"in": "fade", "out": "slide", "duration": 8}
          },
          {
            "asset": {"type": "image", "src": "path/to/clip.jpg"},
            "start": 20,
            "video_start_frame": 0,
            "length": 30,
            "effect": null,
            "transition": {"in": "wipe", "out": "dissolve", "duration": 10}
          }
        ]
      }
    ]
  },
  "output": {
    "format": "mp4",
    "fps": 25,
    "width": 1280,
    "height": 720
  }
}
```

> **Note:** The transition `"duration"` is automatically clamped to half the strip's length so that
> in- and out-transitions never overlap within the same strip.

### Text Overlays

Add text on top of your video or image content using `"type": "text"` in any strip's asset. Text is rendered via the FFmpeg `drawtext` filter.

**Required fields:** `content`  
**Optional fields:** `font`, `size`, `color`, `position`, `animation`

```json
{
  "timeline": {
    "n_frames": 75,
    "background": "#1a1a2e",
    "tracks": [
      {
        "track_id": 0,
        "strips": [
          {
            "asset": {
              "type": "video",
              "src": "path/to/clip.mp4"
            },
            "start": 0,
            "video_start_frame": 0,
            "length": 75,
            "effect": null,
            "transition": {"in": "fade", "out": "fade"}
          }
        ]
      },
      {
        "track_id": 1,
        "strips": [
          {
            "asset": {
              "type": "text",
              "content": "Hello World",
              "font": "/path/to/font.ttf",
              "size": 48,
              "color": "white",
              "position": {"x": 10, "y": 10},
              "animation": "fadeIn"
            },
            "start": 0,
            "video_start_frame": 0,
            "length": 25,
            "effect": null,
            "transition": {}
          }
        ]
      }
    ]
  },
  "output": {
    "format": "mp4",
    "fps": 25,
    "width": 1280,
    "height": 720
  }
}
```

| Text asset field | Type | Required | Default | Description |
|---|---|---|---|---|
| `content` | `string` | ✅ | — | The text string to display. |
| `font` | `string` | ❌ | system default | Path to a `.ttf` / `.otf` font file. |
| `size` | `number` | ❌ | `24` | Font size in pixels. |
| `color` | `string` | ❌ | `"white"` | Font color (FFmpeg color name or hex, e.g. `"red"`, `"#ff0000"`). |
| `position` | `object` | ❌ | `{"x": 0, "y": 0}` | Top-left anchor. Values can be numbers or `"center"` (e.g. `{"x": "center", "y": 50}`). |
| `animation` | `string` | ❌ | `null` | Optional animation tag (stored for future use, e.g. `"fadeIn"`). |

> **Tip:** Text strips in a higher `track_id` are drawn on top of strips with a lower `track_id`.

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