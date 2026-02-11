# Scene Detection Module

The Scene Detection module provides functionality for detecting scene changes in videos using PySceneDetect library. It can detect scene boundaries and split videos into individual scenes based on content changes.

## Features

- **Multiple Detection Algorithms**: Support for AdaptiveDetector, ContentDetector, and ThresholdDetector
- **Scene Splitting**: Automatically split videos into individual scene files
- **Comprehensive Analysis**: Generate detailed scene statistics and reports
- **Flexible Configuration**: Adjustable thresholds, minimum scene lengths, and time ranges
- **Multiple Output Formats**: Save results as JSON, CSV, or split video files

## Installation

First, install the required dependencies:

```bash
pip install scenedetect[opencv]
```

The module is included in the Pavo Engine package. If you're installing from source, ensure scenedetect is in your requirements.

## Quick Start

```python
from pavo.perception.scene import SceneDetector

# Initialize detector
detector = SceneDetector(
    detector_type='adaptive',  # 'adaptive', 'content', or 'threshold'
    threshold=30.0,
    min_scene_len=15
)

# Detect scenes in a video
scenes = detector.detect('video.mp4', show_progress=True)

# Print results
print(f"Found {len(scenes)} scenes")
for scene in scenes:
    print(f"Scene {scene['scene_number']}: {scene['start']:.2f}s to {scene['end']:.2f}s")
```

## Class Reference

### SceneDetector

Main class for scene detection operations.

#### Constructor

```python
SceneDetector(
    detector_type='adaptive',
    threshold=30.0,
    min_scene_len=15,
    output_dir=None,
    stats_file=None
)
```

**Parameters:**
- `detector_type` (str): Scene detection algorithm. Options: 'adaptive', 'content', 'threshold'
- `threshold` (float): Threshold value for scene detection (algorithm-dependent)
- `min_scene_len` (int): Minimum scene length in frames
- `output_dir` (Optional[str]): Directory to save split scenes
- `stats_file` (Optional[str]): Path to save detection statistics CSV file

#### Methods

##### `detect(video_path, start_time=None, end_time=None, show_progress=False)`
Detect scene changes in a video file.

**Returns:** List of scene dictionaries with timing information.

##### `detect_and_split(video_path, output_pattern=None, show_progress=False)`
Detect scenes and split video into individual scene files.

**Returns:** Dictionary containing scenes and output file information.

##### `analyze_scene_statistics(scenes)`
Analyze scene statistics from detection results.

**Returns:** Dictionary with scene statistics and distribution.

##### `save_scenes_to_json(scenes, output_path)`
Save scene detection results to a JSON file.

##### `save_statistics_to_csv(scenes, output_path=None)`
Save scene statistics to a CSV file.

##### `generate_scene_report(video_path, output_dir=None)`
Generate a comprehensive scene analysis report with JSON and CSV files.

## Examples

### Basic Scene Detection

```python
from pavo.perception.scene import SceneDetector

detector = SceneDetector()
scenes = detector.detect('my_video.mp4')

for scene in scenes:
    print(f"Scene {scene['scene_number']}:")
    print(f"  Start: {scene['start_timecode']}")
    print(f"  End: {scene['end_timecode']}")
    print(f"  Duration: {scene['duration']:.2f}s")
```

### Split Video into Scenes

```python
detector = SceneDetector(output_dir='scenes_output')
result = detector.detect_and_split(
    'my_video.mp4',
    output_pattern='scene_{scene_number:03d}.mp4'
)

print(f"Split {result['num_scenes']} scenes")
for file_path in result['output_files']:
    print(f"Created: {file_path}")
```

### Generate Comprehensive Report

```python
detector = SceneDetector(output_dir='scene_report')
report = detector.generate_scene_report('my_video.mp4')

print(f"Report saved to: {report['output_dir']}")
print(f"JSON: {report['json_path']}")
print(f"CSV: {report['csv_path']}")
```

### Compare Different Detectors

```python
detectors = ['adaptive', 'content', 'threshold']
results = {}

for detector_type in detectors:
    detector = SceneDetector(detector_type=detector_type)
    scenes = detector.detect('my_video.mp4')
    stats = detector.analyze_scene_statistics(scenes)
    results[detector_type] = {
        'scenes': len(scenes),
        'avg_duration': stats['avg_duration']
    }
```

## Detection Algorithms

### AdaptiveDetector (Default)
- **Best for**: General-purpose scene detection
- **Threshold range**: 20.0-40.0 (default: 30.0)
- **Characteristics**: Adapts to video content, good for videos with varying lighting

### ContentDetector
- **Best for**: Content-based scene changes
- **Threshold range**: 25.0-35.0 (default: 27.0-30.0)
- **Characteristics**: Detects changes in visual content, good for movies and TV shows

### ThresholdDetector
- **Best for**: Brightness/contrast changes
- **Threshold range**: 8.0-20.0 (default: 12.0)
- **Characteristics**: Detects changes in luminance, good for animations and cartoons

## Use Cases

1. **Video Editing**: Split home videos or source footage into individual scenes
2. **Commercial Removal**: Detect and remove commercials from recorded TV shows
3. **Surveillance Analysis**: Process surveillance footage to identify scene changes
4. **Academic Research**: Analyze film and video for shot length statistics
5. **Content Creation**: Find suitable loops for GIFs or cinemagraphs
6. **Video Compression**: Identify scene boundaries for optimized encoding

## Tips for Best Results

1. **Adjust Threshold**: Lower thresholds detect more scenes, higher thresholds detect fewer but more significant changes
2. **Set Minimum Scene Length**: Prevent detection of very short scenes that may be false positives
3. **Use Appropriate Detector**: Choose detector based on video content type
4. **Process Specific Time Ranges**: Use `start_time` and `end_time` parameters to analyze specific segments
5. **Enable Progress Display**: Use `show_progress=True` for long videos to monitor processing

## Error Handling

The module provides comprehensive error handling:

- `FileNotFoundError`: When video file doesn't exist
- `ValueError`: For invalid parameters or unsupported video formats
- `ImportError`: When scenedetect package is not installed
- `RuntimeError`: For scene detection or splitting failures

## Integration with Pavo Engine

The Scene Detection module integrates seamlessly with other Pavo Engine components:

- Use scene boundaries as input for the Timeline Engine
- Combine with Object Detection for scene content analysis
- Use with Speech Transcription for multimodal scene understanding
- Integrate with the Execution Engine for automated video editing workflows

## Performance Considerations

- Scene detection is CPU-intensive; consider processing time for long videos
- Video splitting requires sufficient disk space for output files
- Memory usage scales with video resolution and length
- For batch processing, consider implementing parallel processing

## License

Part of the Pavo Engine project. See main project license for details.