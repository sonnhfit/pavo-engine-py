# Speech Perception Module

## Overview

The Speech Perception module provides comprehensive audio processing capabilities including:

1. **Speech Transcription**: High-quality speech-to-text using OpenAI's Whisper model
2. **Speaker Diarization**: Identify and track different speakers in audio files
3. **Speech Activity Detection**: Detect segments where speech is present
4. **Speaker Change Detection**: Identify moments when speakers change
5. **Overlapping Speech Detection**: Detect when multiple speakers speak simultaneously
6. **Speaker Embeddings**: Extract speaker voice fingerprints for speaker identification

This module is built on state-of-the-art deep learning models from pyannote.audio and OpenAI Whisper.

## Features

- 🎤 **High-Quality Transcription**: Uses OpenAI's Whisper model for accurate speech-to-text conversion
- 🌍 **Multi-Language Support**: Automatically detects or explicitly specifies languages
- 📁 **Multiple Format Support**: Works with WAV, MP3, M4A, FLAC, OGG, Opus, and WebM formats
- 🚀 **Easy Integration**: Simple API for quick integration into your projects
- 💾 **Memory Efficient**: Supports lazy loading and model unloading
- 🔄 **Context Manager Support**: Automatic resource cleanup
- ⚡ **GPU Support**: Can leverage CUDA for faster processing

## Installation

1. Install the package with speech support:

```bash
pip install openai-whisper
```

2. Or install with the main pavo package:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from pavo.perception.speech import SpeechTranscriber

# Create a transcriber instance
transcriber = SpeechTranscriber(model_name='base', language='en')

# Transcribe an audio file
result = transcriber.transcribe('audio.wav')
print(result['text'])  # Print the transcribed text
```

### Using Context Manager (Recommended)

```python
from pavo.perception.speech import SpeechTranscriber

# Automatically unload model after use
with SpeechTranscriber(model_name='base') as transcriber:
    text = transcriber.transcribe_file('audio.wav')
    print(text)
# Model is automatically unloaded
```

### Quick Text Extraction

```python
transcriber = SpeechTranscriber()
text = transcriber.transcribe_file('speech.wav')
print(text)
```

### Language Detection

```python
transcriber = SpeechTranscriber()
language = transcriber.get_language('audio.wav')
print(f"Detected language: {language}")
```

## API Reference

### SpeechTranscriber Class

#### Constructor Parameters

```python
SpeechTranscriber(
    model_name: str = 'base',
    language: Optional[str] = None,
    task: str = 'transcribe',
    device: str = 'cpu'
)
```

**Parameters:**

- `model_name` (str): Whisper model size
  - Options: `'tiny'`, `'tiny.en'`, `'base'`, `'base.en'`, `'small'`, `'small.en'`, `'medium'`, `'medium.en'`, `'large'`, `'large-v1'`, `'large-v2'`, `'large-v3'`, `'large-v3-turbo'`
  - Default: `'base'`
  - Smaller models are faster but less accurate; larger models are slower but more accurate

- `language` (Optional[str]): Language code (e.g., `'en'`, `'vi'`, `'fr'`)
  - If `None`, language will be auto-detected
  - Default: `None`

- `task` (str): Type of task to perform
  - `'transcribe'`: Convert speech to text
  - `'translate'`: Translate speech to English
  - Default: `'transcribe'`

- `device` (str): Device for inference
  - `'cpu'`: Use CPU (default, works everywhere)
  - `'cuda'`: Use GPU (requires NVIDIA GPU and CUDA toolkit)
  - Default: `'cpu'`

#### Methods

##### `transcribe(audio_path, language=None, task=None, verbose=False) -> Dict[str, Any]`

Transcribe an audio file with full metadata.

**Parameters:**
- `audio_path` (str): Path to the audio file
- `language` (Optional[str]): Override instance language (optional)
- `task` (Optional[str]): Override instance task (optional)
- `verbose` (bool): Print progress information

**Returns:**
- Dictionary containing:
  - `'text'` (str): The transcribed text
  - `'language'` (str): Detected language code
  - `'segments'` (list): Segments with timing information
  - `'duration'` (float): Duration in seconds

**Raises:**
- `FileNotFoundError`: If audio file doesn't exist
- `ValueError`: If audio format is not supported
- `RuntimeError`: If transcription fails

**Example:**
```python
transcriber = SpeechTranscriber(model_name='base')
result = transcriber.transcribe('speech.wav')
print(result['text'])
print(f"Language: {result['language']}")
print(f"Duration: {result['duration']:.2f} seconds")
```

##### `transcribe_file(audio_path) -> str`

Convenience method to transcribe and return only the text.

**Parameters:**
- `audio_path` (str): Path to the audio file

**Returns:**
- str: The transcribed text

**Example:**
```python
transcriber = SpeechTranscriber()
text = transcriber.transcribe_file('speech.wav')
print(text)
```

##### `get_language(audio_path) -> str`

Detect the language of an audio file.

**Parameters:**
- `audio_path` (str): Path to the audio file

**Returns:**
- str: Language code (e.g., 'en', 'vi')

**Example:**
```python
language = transcriber.get_language('audio.wav')
print(f"Detected language: {language}")
```

##### `unload_model()`

Unload the model to free up memory.

**Example:**
```python
transcriber = SpeechTranscriber()
# ... use transcriber ...
transcriber.unload_model()  # Free memory
```

## Supported Audio Formats

The following audio formats are supported:

- `.wav` - WAV (Waveform Audio File Format)
- `.mp3` - MPEG-1 Audio Layer III
- `.m4a` - MPEG-4 Audio
- `.flac` - Free Lossless Audio Codec
- `.ogg` - Ogg Vorbis
- `.opus` - Opus Audio Codec
- `.webm` - WebM (WebM Audio)

## Model Selection Guide

| Model | Disk | Memory | Speed | Accuracy |
|-------|------|--------|-------|----------|
| tiny | 75 MB | ~273 MB | Very Fast | Low |
| base | 142 MB | ~388 MB | Fast | Medium |
| small | 466 MB | ~852 MB | Medium | Good |
| medium | 1.5 GB | ~2.1 GB | Slow | Very Good |
| large | 2.9 GB | ~3.9 GB | Very Slow | Excellent |

**Recommendations:**

- **Real-time applications**: Use `tiny` or `tiny.en`
- **Balanced use**: Use `base` or `base.en` (recommended for most cases)
- **High accuracy needed**: Use `small`, `medium`, or `large`
- **Known language**: Use `.en` variant (e.g., `base.en`) for faster processing

## Examples

### Example 1: Simple Transcription

```python
from pavo.perception.speech import SpeechTranscriber

# Create transcriber
transcriber = SpeechTranscriber(model_name='base', language='en')

# Transcribe file
result = transcriber.transcribe('interview.wav')

# Print results
print(f"Transcribed Text:\n{result['text']}")
print(f"\nLanguage: {result['language']}")
print(f"Duration: {result['duration']:.2f} seconds")
```

### Example 2: Batch Processing

```python
from pavo.perception.speech import SpeechTranscriber
import os

# Create transcriber once to reuse model
transcriber = SpeechTranscriber(model_name='base')

# Process multiple files
audio_folder = 'audio_files'
for filename in os.listdir(audio_folder):
    if filename.endswith('.wav'):
        filepath = os.path.join(audio_folder, filename)
        text = transcriber.transcribe_file(filepath)
        print(f"{filename}: {text[:50]}...")

# Unload model when done
transcriber.unload_model()
```

### Example 3: Multi-Language with Context Manager

```python
from pavo.perception.speech import SpeechTranscriber

# Process files without explicit language
with SpeechTranscriber(model_name='base') as transcriber:
    files = {
        'english.wav': 'en',
        'vietnamese.wav': 'vi',
        'french.wav': 'fr'
    }
    
    for filepath, expected_lang in files.items():
        result = transcriber.transcribe(filepath)
        print(f"{filepath}:")
        print(f"  Language: {result['language']}")
        print(f"  Text: {result['text'][:50]}...")
```

### Example 4: GPU Acceleration

```python
from pavo.perception.speech import SpeechTranscriber

# Use GPU for faster processing
transcriber = SpeechTranscriber(
    model_name='small',
    device='cuda'  # Requires NVIDIA GPU
)

result = transcriber.transcribe('long_audio.wav')
print(result['text'])
```

### Example 5: Speech Translation

```python
from pavo.perception.speech import SpeechTranscriber

# Translate any language to English
transcriber = SpeechTranscriber(
    model_name='base',
    task='translate'  # Set task to 'translate'
)

result = transcriber.transcribe('spanish_speech.wav')
print(result['text'])  # Will contain English translation
```

## Error Handling

```python
from pavo.perception.speech import SpeechTranscriber

transcriber = SpeechTranscriber()

try:
    result = transcriber.transcribe('audio.wav')
except FileNotFoundError:
    print("Audio file not found!")
except ValueError as e:
    print(f"Unsupported format: {e}")
except RuntimeError as e:
    print(f"Transcription failed: {e}")
```

## Performance Tips

1. **Reuse transcriber instance**: Create once, use multiple times
2. **Use smaller models for real-time**: Prefer `tiny` or `base` for speed
3. **Use GPU when available**: Specify `device='cuda'` for faster processing
4. **Unload model when done**: Call `unload_model()` to free memory
5. **Use context manager**: Automatic cleanup with `with` statement
6. **Specify language**: Faster processing when language is known

## Troubleshooting

### ImportError: No module named 'whisper'

Install openai-whisper:
```bash
pip install openai-whisper
```

### CUDA errors (GPU processing)

Make sure you have:
1. NVIDIA GPU installed
2. CUDA toolkit installed
3. PyTorch with CUDA support installed

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of memory errors

- Use smaller model (e.g., `tiny` instead of `large`)
- Use CPU instead of GPU
- Call `unload_model()` to free memory between transcriptions

### Audio format not supported

Convert your audio to a supported format using ffmpeg:
```bash
ffmpeg -i input.audio -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

---

# Speaker Diarization Module

## Overview

Speaker diarization is the task of answering "who spoke when?" in an audio file. The module provides comprehensive speaker identification and tracking capabilities.

**Key Features:**
- 🎤 Speaker identification and tracking
- 🔊 Speech activity detection (Voice Activity Detection - VAD)
- 🔄 Speaker change detection
- 📢 Overlapping speech detection
- 🎯 Speaker embeddings (voice fingerprints)
- 🚀 GPU acceleration support
- 🤗 Community and premium pipeline options

## Installation

### Prerequisites

1. **ffmpeg** is required for audio processing:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows (using chocolatey)
choco install ffmpeg
```

2. **pyannote.audio** library:

```bash
pip install pyannote.audio
```

### Setup Access Tokens

1. **Create Hugging Face Account**
   - Visit [hf.co](https://hf.co)
   - Create account and verify email

2. **Accept Model Conditions**
   - Go to [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   - Click "Access repository" and accept conditions

3. **Generate Access Token**
   - Go to [hf.co/settings/tokens](https://hf.co/settings/tokens)
   - Create new token with "read" permission
   - Copy token and set environment variable:

```bash
export HUGGINGFACE_TOKEN='your_token_here'
```

## Quick Start

### Basic Speaker Diarization

```python
from pavo.perception.speech import SpeakerDiarization

# Create diarizer
diarizer = SpeakerDiarization(
    pipeline_type='community',
    token='your_huggingface_token'
)

# Perform diarization
result = diarizer.diarize('audio.wav')

# Print results
result.print_summary()

# Or access data programmatically
for start, end, speaker in result.speaker_turns:
    print(f"{start:.1f}s - {end:.1f}s: {speaker}")
```

### Using Context Manager (Recommended)

```python
from pavo.perception.speech import SpeakerDiarization

# Automatic resource cleanup
with SpeakerDiarization(pipeline_type='community', token='your_token') as diarizer:
    result = diarizer.diarize('audio.wav', verbose=True)
    result.print_summary()
# Models automatically unloaded
```

## API Reference

### SpeakerDiarization Class

#### Constructor

```python
SpeakerDiarization(
    pipeline_type: str = 'community',
    token: Optional[str] = None,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    num_speakers: Optional[int] = None,
    min_speakers: int = 1,
    max_speakers: int = 10
)
```

**Parameters:**

- `pipeline_type` (str): Pipeline to use
  - `'community'`: Free open-source community-1 (recommended)
  - `'premium'`: Premium precision-2 (requires pyannoteAI API key)
  - Default: `'community'`

- `token` (Optional[str]): Authentication token
  - For community: Hugging Face token
  - For premium: pyannoteAI API key
  - If None, looks for environment variables
  - Default: `None`

- `device` (str): Device for computation
  - `'cpu'`: CPU processing (slower but always works)
  - `'cuda'`: GPU processing (requires NVIDIA GPU)
  - Default: Auto-detect

- `num_speakers` (Optional[int]): Exact number of speakers
  - If provided, forces detection of this many speakers
  - If None, auto-detects within min/max range
  - Default: `None`

- `min_speakers` (int): Minimum speakers to detect
  - Default: `1`

- `max_speakers` (int): Maximum speakers to detect
  - Default: `10`

#### Methods

##### `diarize(audio_path, num_speakers=None, min_speakers=None, max_speakers=None, verbose=False)`

Perform speaker diarization on an audio file.

**Parameters:**
- `audio_path` (str): Path to audio file
- `num_speakers` (Optional[int]): Override instance setting
- `min_speakers` (Optional[int]): Override instance setting
- `max_speakers` (Optional[int]): Override instance setting
- `verbose` (bool): Print progress information

**Returns:**
- `SpeakerDiarizationResult`: Result object with speaker turns

**Example:**
```python
result = diarizer.diarize('meeting.wav', num_speakers=3, verbose=True)
print(f"Detected {result.num_speakers} speakers")
for start, end, speaker in result.speaker_turns:
    print(f"{start:.1f}s - {end:.1f}s: {speaker}")
```

##### `detect_speech_activity(audio_path) -> List[Tuple[float, float]]`

Detect speech activity segments (Voice Activity Detection).

**Returns:**
- List of (start_time, end_time) tuples

**Example:**
```python
segments = diarizer.detect_speech_activity('audio.wav')
for start, end in segments:
    duration = end - start
    print(f"Speech: {start:.1f}s - {end:.1f}s ({duration:.1f}s)")
```

##### `detect_speaker_changes(audio_path) -> List[float]`

Detect moments when the speaker changes.

**Returns:**
- List of timestamps where speaker changes occur

**Example:**
```python
changes = diarizer.detect_speaker_changes('audio.wav')
for timestamp in changes:
    print(f"Speaker change at {timestamp:.1f}s")
```

##### `detect_overlapping_speech(audio_path) -> List[Tuple[float, float]]`

Detect segments where multiple speakers overlap.

**Returns:**
- List of (start_time, end_time) tuples

**Example:**
```python
overlaps = diarizer.detect_overlapping_speech('audio.wav')
print(f"Found {len(overlaps)} overlapping segments")
```

##### `get_speaker_embeddings(audio_path, speaker_turns=None) -> Dict[str, Any]`

Extract speaker embeddings for speaker identification.

**Parameters:**
- `audio_path` (str): Path to audio file
- `speaker_turns` (Optional[List]): Speaker turns to extract embeddings for

**Returns:**
- Dictionary mapping speaker IDs to embeddings

**Example:**
```python
result = diarizer.diarize('audio.wav')
embeddings = diarizer.get_speaker_embeddings('audio.wav', speaker_turns=result.speaker_turns)
for speaker_id, embedding in embeddings.items():
    print(f"{speaker_id}: embedding shape {embedding.shape}")
```

##### `unload_models()`

Unload all models to free GPU/CPU memory.

**Example:**
```python
diarizer.unload_models()
```

### SpeakerDiarizationResult Class

Container for diarization results.

**Attributes:**
- `speaker_turns` (List): List of (start, end, speaker) tuples
- `num_speakers` (int): Number of detected speakers
- `timeline` (Any): Raw timeline object
- `metadata` (Dict): Additional metadata

**Methods:**

##### `print_summary()`

Print formatted summary of results.

```python
result.print_summary()
# Output:
# Speaker Diarization Summary:
#   Total speakers: 2
#   Total turns: 5
#   Duration: 123.4s
#   Speaker turns:
#     0.0s - 10.5s: speaker_0
#     10.8s - 25.3s: speaker_1
#     ...
```

##### `to_dict() -> Dict[str, Any]`

Convert result to dictionary format.

```python
result_dict = result.to_dict()
```

### Individual Component Classes

You can use components independently:

#### SpeechActivityDetection

```python
from pavo.perception.speech import SpeechActivityDetection

vad = SpeechActivityDetection(device='cpu')
segments = vad.detect('audio.wav')
vad.unload_model()
```

#### SpeakerChangeDetection

```python
from pavo.perception.speech import SpeakerChangeDetection

scd = SpeakerChangeDetection(device='cpu')
changes = scd.detect_changes('audio.wav', threshold=0.5)
scd.unload_model()
```

#### OverlappingSpeechDetection

```python
from pavo.perception.speech import OverlappingSpeechDetection

osd = OverlappingSpeechDetection(device='cpu')
overlaps = osd.detect('audio.wav', threshold=0.5)
osd.unload_model()
```

#### SpeakerEmbedding

```python
from pavo.perception.speech import SpeakerEmbedding

se = SpeakerEmbedding(device='cpu')
embeddings = se.get_embeddings('audio.wav')
# Compare embeddings
similarity = se.compare_embeddings(emb1, emb2)
se.unload_model()
```

## Examples

See `SPEAKER_DIARIZATION_EXAMPLE.py` for 14 comprehensive examples including:

1. Basic speaker diarization
2. Premium pipeline usage
3. Speaker count constraints
4. Context manager pattern
5. Speech activity detection
6. Speaker change detection
7. Overlapping speech detection
8. Speaker embeddings
9. Embedding comparison
10. Batch processing
11. Advanced configuration
12. Individual components
13. Error handling
14. Output formats

## Pipeline Comparison

| Feature | Community-1 | Premium Precision-2 |
|---------|-------------|-------------------|
| Cost | Free | Paid (pyannoteAI) |
| Accuracy | Very Good | Excellent |
| Speed | Good | 2-3x faster |
| Processing | Local | Cloud |
| Token | Hugging Face | pyannoteAI |
| Setup | Easy | Simple |

## Benchmark Results

Community-1 pipeline diarization error rates (lower is better):

| Dataset | Error Rate |
|---------|-----------|
| AMI (IHM) | 17.0% |
| AliMeeting | 20.3% |
| DIHARD 3 | 20.2% |
| VoxConverse | 11.2% |

Processing speed on NVIDIA H100:
- Community-1: ~31s per hour of audio
- Premium: ~14s per hour of audio (2.2x faster)

## GPU Support

### Requirements for GPU acceleration

1. NVIDIA GPU (CUDA compute capability 3.5+)
2. CUDA Toolkit installed
3. cuDNN library
4. PyTorch with CUDA support

### Installation for GPU

```bash
# PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Enable GPU in code

```python
diarizer = SpeakerDiarization(
    pipeline_type='community',
    token='your_token',
    device='cuda'  # Use GPU
)
```

## Troubleshooting

### ImportError: No module named 'pyannote'

```bash
pip install pyannote.audio
```

### ffmpeg not found

Install ffmpeg for your OS:
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt-get install ffmpeg
```

### CUDA out of memory

- Use CPU: `device='cpu'`
- Use smaller pipeline or smaller audio files
- Increase GPU memory allocation

### Token errors

Ensure environment variable is set:
```bash
export HUGGINGFACE_TOKEN='your_token'
# Or pass token to constructor
diarizer = SpeakerDiarization(token='your_token')
```

## Dependencies

- `pyannote.audio`: Speaker diarization pipeline
- `torch`: Deep learning framework
- `numpy`: Numerical computing
- `scikit-learn`: For embedding similarity
- `ffmpeg`: Audio processing

## License

This module is part of the Pavo Engine project and follows the same license.

## References

- [pyannote.audio Documentation](https://github.com/pyannote/pyannote-audio)
- [pyannote.audio Benchmarks](https://github.com/pyannote/pyannote-audio#benchmark)
- [pyannoteAI (Premium Service)](https://pyannote.ai)
- [Hugging Face Models](https://huggingface.co/pyannote)

## Citations

If you use pyannote.audio in your research, please cite:

```bibtex
@inproceedings{Plaquet23,
  author={Alexis Plaquet and Hervé Bredin},
  title={{Powerset multi-class cross entropy loss for neural speaker diarization}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}

@inproceedings{Bredin23,
  author={Hervé Bredin},
  title={{pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
}
```

---

## Dependencies

- `openai-whisper`: Speech recognition model
- `pyannote.audio`: Speaker diarization and related tasks
- `numpy`: Numerical computations
- `torch`: Deep learning framework
- `scikit-learn`: Machine learning utilities

## License

This module is part of the Pavo Engine project and follows the same license.

## References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [Whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [Supported Languages](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)
