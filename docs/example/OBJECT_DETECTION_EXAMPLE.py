"""
Object Detection Example for Pavo Engine

This example demonstrates how to use the ObjectDetector module for detecting
objects in images and videos, and analyzing frame composition.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import pavo modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pavo.perception.object import ObjectDetector


def example_image_detection():
    """Example: Detect objects in an image."""
    print("=" * 60)
    print("IMAGE DETECTION EXAMPLE")
    print("=" * 60)
    
    # Initialize detector with YOLO26 nano model (small and fast)
    detector = ObjectDetector(
        model_name='yolo26n.pt',
        device='cpu',  # Change to 'cuda' if you have GPU
        conf_threshold=0.25
    )
    
    # Use a sample image from the media directory
    image_path = 'docs/media/im1.jpg'
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Sample image not found: {image_path}")
        print("Please ensure you have sample images in docs/media/")
        return
    
    print(f"Detecting objects in: {image_path}")
    
    # Detect objects with annotated image output
    results = detector.detect_image(
        image_path,
        return_image=True,
        verbose=True
    )
    
    # Print results
    print(f"\nDetection Results:")
    print(f"  Image size: {results['image_info']['width']}x{results['image_info']['height']}")
    print(f"  Number of detections: {results['num_detections']}")
    
    if results['num_detections'] > 0:
        print(f"\nDetected objects:")
        for i, detection in enumerate(results['detections']):
            print(f"  {i+1}. {detection['class_name']} (confidence: {detection['confidence']:.2f})")
            print(f"     Bounding box: {[int(x) for x in detection['bbox']]}")
            print(f"     Center: ({detection['center'][0]:.1f}, {detection['center'][1]:.1f})")
    
    # Analyze frame center context
    context = detector.get_frame_center_context(
        results['detections'],
        results['image_info']['width'],
        results['image_info']['height']
    )
    
    print(f"\nFrame Center Analysis:")
    print(f"  Frame center: ({context['frame_center'][0]:.1f}, {context['frame_center'][1]:.1f})")
    print(f"  Objects near center: {context['num_objects_near_center']}/{context['total_objects']}")
    print(f"  Recommendation: {context['recommendation']}")
    
    if context['dominant_object']:
        print(f"  Dominant object: {context['dominant_object']['class_name']}")
    
    # Save annotated image
    if results['annotated_image'] is not None:
        output_path = 'docs/example/annotated_image.jpg'
        import cv2
        cv2.imwrite(output_path, results['annotated_image'])
        print(f"\nAnnotated image saved to: {output_path}")
    
    # Save detections to JSON
    json_path = 'docs/example/detections.json'
    detector.save_detections_to_json(results, json_path)
    print(f"Detection results saved to: {json_path}")
    
    return results


def example_video_detection():
    """Example: Detect objects in a video."""
    print("\n" + "=" * 60)
    print("VIDEO DETECTION EXAMPLE")
    print("=" * 60)
    
    # Initialize detector with YOLO11 small model (good balance)
    detector = ObjectDetector(
        model_name='yolo11s.pt',
        device='cpu',
        conf_threshold=0.3
    )
    
    # Use a sample video from the media directory
    video_path = 'docs/media/in.mp4'
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Sample video not found: {video_path}")
        print("Please ensure you have sample videos in docs/media/")
        return
    
    print(f"Detecting objects in video: {video_path}")
    print("This may take a while depending on video length and hardware...")
    
    # Detect objects in video (process every 10th frame for speed)
    results = detector.detect_video(
        video_path,
        frame_interval=10,  # Process every 10th frame
        output_path='docs/example/annotated_video.mp4',
        verbose=True
    )
    
    # Print video results summary
    print(f"\nVideo Detection Summary:")
    print(f"  Video info: {results['video_info']['width']}x{results['video_info']['height']} @ {results['video_info']['fps']}fps")
    print(f"  Total frames: {results['video_info']['total_frames']}")
    print(f"  Processed frames: {results['video_info']['processed_frames']}")
    print(f"  Total detections: {results['summary']['total_detections']}")
    print(f"  Avg detections per frame: {results['summary']['avg_detections_per_frame']:.2f}")
    print(f"  Frames with detections: {results['summary']['frames_with_detections']}")
    
    # Show sample frame detections
    print(f"\nSample frame detections (first 5 processed frames):")
    for frame_result in results['frame_detections'][:5]:
        if frame_result['num_detections'] > 0:
            classes = [d['class_name'] for d in frame_result['detections']]
            print(f"  Frame {frame_result['frame_number']}: {frame_result['num_detections']} objects - {', '.join(classes)}")
        else:
            print(f"  Frame {frame_result['frame_number']}: No objects detected")
    
    # Analyze composition for frames with detections
    print(f"\nComposition Analysis:")
    frames_with_center_objects = 0
    for frame_result in results['frame_detections'][:20]:  # First 20 processed frames
        if frame_result['num_detections'] > 0:
            context = detector.get_frame_center_context(
                frame_result['detections'],
                results['video_info']['width'],
                results['video_info']['height']
            )
            if context['num_objects_near_center'] > 0:
                frames_with_center_objects += 1
    
    total_analyzed = min(20, len(results['frame_detections']))
    if total_analyzed > 0:
        center_percentage = (frames_with_center_objects / total_analyzed) * 100
        print(f"  Frames with objects near center: {frames_with_center_objects}/{total_analyzed} ({center_percentage:.1f}%)")
        
        if center_percentage < 50:
            print("  Recommendation: Consider adjusting camera framing to center objects more often")
        else:
            print("  Recommendation: Good framing with objects frequently near center")
    
    if results['output_path']:
        print(f"\nAnnotated video saved to: {results['output_path']}")
    
    # Save video results to JSON
    json_path = 'docs/example/video_detections.json'
    detector.save_detections_to_json(results, json_path)
    print(f"Video detection results saved to: {json_path}")
    
    return results


def example_batch_processing():
    """Example: Batch process multiple images."""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    detector = ObjectDetector(model_name='yolo26n.pt')
    
    # Find all images in media directory
    media_dir = Path('docs/media')
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_paths.extend(list(media_dir.glob(f'*{ext}')))
    
    if not image_paths:
        print("No images found in docs/media/ directory")
        return
    
    # Limit to 3 images for demonstration
    image_paths = [str(p) for p in image_paths[:3]]
    
    print(f"Batch processing {len(image_paths)} images:")
    for path in image_paths:
        print(f"  - {Path(path).name}")
    
    # Process all images
    batch_results = detector.detect_batch(image_paths, verbose=True)
    
    # Print summary
    print(f"\nBatch Processing Summary:")
    total_detections = 0
    for i, result in enumerate(batch_results):
        if 'error' in result:
            print(f"  Image {i+1}: ERROR - {result['error']}")
        else:
            detections = result['num_detections']
            total_detections += detections
            if detections > 0:
                classes = [d['class_name'] for d in result['detections'][:3]]  # First 3 classes
                classes_str = ', '.join(classes)
                if detections > 3:
                    classes_str += f" and {detections - 3} more"
                print(f"  Image {i+1}: {detections} objects - {classes_str}")
            else:
                print(f"  Image {i+1}: No objects detected")
    
    print(f"\nTotal objects detected across all images: {total_detections}")
    
    return batch_results


def example_custom_configuration():
    """Example: Custom configuration for specific use cases."""
    print("\n" + "=" * 60)
    print("CUSTOM CONFIGURATION EXAMPLE")
    print("=" * 60)
    
    # Example 1: Person detection only
    print("\n1. Person Detection Only:")
    person_detector = ObjectDetector(
        model_name='yolo26m.pt',
        classes=[0],  # Class 0 is 'person' in COCO dataset
        conf_threshold=0.4
    )
    print("   Configured to detect only persons with higher confidence threshold")
    
    # Example 2: Vehicle detection for traffic analysis
    print("\n2. Vehicle Detection for Traffic Analysis:")
    vehicle_classes = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck
    vehicle_detector = ObjectDetector(
        model_name='yolo11l.pt',
        classes=vehicle_classes,
        conf_threshold=0.3,
        iou_threshold=0.5
    )
    print("   Configured for vehicle detection with specific classes")
    
    # Example 3: High-precision detection for critical applications
    print("\n3. High-Precision Detection:")
    precision_detector = ObjectDetector(
        model_name='yolo26x.pt',
        conf_threshold=0.5,  # Higher threshold for fewer false positives
        iou_threshold=0.3    # Lower IoU for better separation of close objects
    )
    print("   Configured for high precision with larger model")
    
    return {
        'person_detector': person_detector,
        'vehicle_detector': vehicle_detector,
        'precision_detector': precision_detector
    }


def main():
    """Run all examples."""
    print("Pavo Engine - Object Detection Module Examples")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    os.makedirs('docs/example', exist_ok=True)
    
    try:
        # Run examples
        image_results = example_image_detection()
        video_results = example_video_detection()
        batch_results = example_batch_processing()
        config_examples = example_custom_configuration()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nOutput files created in docs/example/:")
        print("  - annotated_image.jpg (if image detection worked)")
        print("  - annotated_video.mp4 (if video detection worked)")
        print("  - detections.json")
        print("  - video_detections.json")
        print("\nNext steps:")
        print("  1. Install ultralytics: pip install ultralytics")
        print("  2. Run this example: python docs/example/OBJECT_DETECTION_EXAMPLE.py")
        print("  3. Modify parameters for your specific use case")
        
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install ultralytics opencv-python pillow numpy")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nMake sure you have sample media files in docs/media/")
        print("You can download sample images/videos or use your own.")


if __name__ == "__main__":
    main()