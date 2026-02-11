# Object Detection Module

This module provides object detection capabilities using Ultralytics YOLO models for the Pavo video editing engine. It can detect objects in images and videos, output bounding boxes and positions, and provide context for determining the center of frames.

## Features

- **Image Detection**: Detect objects in single images with bounding boxes and confidence scores
- **Video Detection**: Process video files frame-by-frame with configurable frame intervals
- **Batch Processing**: Detect objects in multiple images efficiently
- **Center Context Analysis**: Analyze object positions relative to frame center for composition feedback
- **Multiple Output Formats**: Return results as dictionaries, save annotated images/videos, or export to JSON
- **Flexible Model Selection**: Support for YOLO26, YOLO11, and custom trained models
- **Device Support**: CPU, CUDA (GPU), and MPS (Apple Silicon) acceleration

## Installation

The module requires the following dependencies:

```bash
pip install ultralytics opencv-python pillow numpy
```

Add to `requirements.txt`:
```
ultralytics>=8.0.0
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0
```

## Usage

### Basic Image Detection

```python
from pavo.perception.object import ObjectDetector

# Initialize detector with default model (YOLO26n)
detector = ObjectDetector(model_name='yolo26n.pt', device='cpu')

# Detect objects in an image
results = detector.detect_image('path/to/image.jpg', return_image=True)

# Access detection results
print(f"Number of detections: {results['num_detections']}")
for detection in results['detections']:
    print(f"Class: {detection['class_name']}")
    print(f"Confidence: {detection['confidence']:.2f}")
    print(f"Bounding box: {detection['bbox']}")
    print(f"Center: {detection['center']}")

# Save annotated image
if results['annotated_image'] is not None:
    import cv2
    cv2.imwrite('annotated_image.jpg', results['annotated_image'])
```

### Video Detection

```python
from pavo.perception.object import ObjectDetector

detector = ObjectDetector(model_name='yolo11n.pt')

# Detect objects in video with annotated output
results = detector.detect_video(
    video_path='path/to/video.mp4',
    output_path='annotated_video.mp4',
    frame_interval=5,  # Process every 5th frame
    verbose=True
)

# Access video results
print(f"Video FPS: {results['video_info']['fps']}")
print(f"Total frames processed: {results['video_info']['processed_frames']}")
print(f"Total detections: {results['summary']['total_detections']}")

# Analyze frame-by-frame detections
for frame_result in results['frame_detections'][:10]:  # First 10 frames
    print(f"Frame {frame_result['frame_number']}: {frame_result['num_detections']} objects")
```

### Center Context Analysis

```python
from pavo.perception.object import ObjectDetector

detector = ObjectDetector()

# Detect objects
results = detector.detect_image('path/to/image.jpg')

# Analyze frame center context
context = detector.get_frame_center_context(
    detections=results['detections'],
    image_width=results['image_info']['width'],
    image_height=results['image_info']['height']
)

print(f"Frame center: {context['frame_center']}")
print(f"Objects near center: {context['num_objects_near_center']}")
print(f"Recommendation: {context['recommendation']}")

if context['dominant_object']:
    print(f"Dominant object: {context['dominant_object']['class_name']}")
```

### Batch Processing

```python
from pavo.perception.object import ObjectDetector

detector = ObjectDetector()

# List of image paths
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']

# Process all images
batch_results = detector.detect_batch(image_paths, verbose=True)

# Save results to JSON
for i, result in enumerate(batch_results):
    if 'error' not in result:
        detector.save_detections_to_json(
            result,
            f'detections_image_{i+1}.json'
        )
```

## API Reference

### ObjectDetector Class

#### `__init__(model_name='yolo26n.pt', device='cpu', conf_threshold=0.25, iou_threshold=0.45, classes=None)`

Initialize the object detector.

- `model_name`: YOLO model name or path (default: 'yolo26n.pt')
- `device`: Inference device ('cpu', 'cuda', or 'mps')
- `conf_threshold`: Confidence threshold (0.0 to 1.0)
- `iou_threshold`: IoU threshold for NMS (0.0 to 1.0)
- `classes`: List of class IDs to detect (None for all classes)

#### `detect_image(image_path, conf=None, iou=None, classes=None, return_image=False, verbose=False)`

Detect objects in an image.

- `image_path`: Path to image, numpy array, or PIL Image
- `return_image`: If True, returns annotated image with bounding boxes
- Returns: Dictionary with detections, image info, and optional annotated image

#### `detect_video(video_path, conf=None, iou=None, classes=None, frame_interval=1, output_path=None, verbose=False)`

Detect objects in a video file.

- `video_path`: Path to video file
- `frame_interval`: Process every Nth frame
- `output_path`: Save annotated video to this path
- Returns: Dictionary with frame detections, video info, and summary

#### `detect_batch(image_paths, conf=None, iou=None, classes=None, verbose=False)`

Detect objects in multiple images.

- `image_paths`: List of image file paths
- Returns: List of detection results for each image

#### `get_frame_center_context(detections, image_width, image_height)`

Analyze object positions relative to frame center.

- `detections`: List of detection dictionaries
- `image_width`, `image_height`: Frame dimensions
- Returns: Context information including center analysis and recommendations

#### `save_detections_to_json(detections, output_path)`

Save detection results to JSON file.

- `detections`: Detection results dictionary
- `output_path`: Path to save JSON file

## Model Options

### Available YOLO Models

- **YOLO26 Series**: Latest models with NMS-free inference
  - `yolo26n.pt` (nano) - Fastest, lowest accuracy
  - `yolo26s.pt` (small)
  - `yolo26m.pt` (medium)
  - `yolo26l.pt` (large)
  - `yolo26x.pt` (extra large) - Slowest, highest accuracy

- **YOLO11 Series**: Stable production models
  - `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt`

### Custom Models

You can use custom trained YOLO models by providing the path:

```python
detector = ObjectDetector(model_name='path/to/custom_model.pt')
```

## Performance Tips

1. **Device Selection**: Use 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, 'cpu' for CPU-only
2. **Frame Interval**: For video processing, increase `frame_interval` to speed up processing
3. **Confidence Threshold**: Adjust `conf_threshold` based on your accuracy vs. false positive needs
4. **Model Size**: Choose smaller models (n, s) for speed, larger models (l, x) for accuracy
5. **Batch Processing**: Use `detect_batch()` for multiple images to avoid repeated model loading

## Integration with Pavo Engine

This module is designed to work with the Pavo video editing engine's perception layer. The detection results can be used as context for:

1. **Frame Composition Analysis**: Determine if objects are properly centered
2. **Object Tracking**: Track objects across frames for editing decisions
3. **Scene Understanding**: Identify key objects for automatic editing
4. **Timeline Generation**: Use object positions to inform edit timing

## Example Use Case: Center-Framing Assistant

```python
from pavo.perception.object import ObjectDetector

def analyze_video_composition(video_path):
    """Analyze video composition and provide framing feedback."""
    detector = ObjectDetector()
    results = detector.detect_video(video_path, frame_interval=30)
    
    framing_issues = []
    for frame_result in results['frame_detections']:
        if frame_result['num_detections'] > 0:
            # Get center context for this frame
            context = detector.get_frame_center_context(
                frame_result['detections'],
                results['video_info']['width'],
                results['video_info']['height']
            )
            
            if context['num_objects_near_center'] == 0:
                framing_issues.append({
                    'frame': frame_result['frame_number'],
                    'issue': 'No objects near center',
                    'recommendation': context['recommendation']
                })
    
    return {
        'total_frames': results['video_info']['processed_frames'],
        'framing_issues': framing_issues,
        'issues_percentage': len(framing_issues) / results['video_info']['processed_frames'] * 100
    }
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure ultralytics is installed and internet connection is available for model download
2. **CUDA out of memory**: Reduce batch size or use smaller model
3. **Video processing slow**: Increase `frame_interval` or use smaller model
4. **No detections**: Lower `conf_threshold` or check if objects are in model's training classes

### Error Handling

The module includes comprehensive error handling for:
- Missing files
- Unsupported formats
- Model loading failures
- Inference errors
- Resource cleanup

## License

This module is part of the Pavo Engine project. See the main project LICENSE for details.