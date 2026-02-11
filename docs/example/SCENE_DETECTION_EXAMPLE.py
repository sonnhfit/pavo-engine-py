"""
Scene Detection Example for Pavo Engine

This example demonstrates how to use the SceneDetector module for detecting
scene changes in videos and splitting videos into individual scenes.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import pavo modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pavo.perception.scene import SceneDetector


def example_basic_detection():
    """Example: Basic scene detection in a video."""
    print("=" * 60)
    print("BASIC SCENE DETECTION EXAMPLE")
    print("=" * 60)
    
    # Initialize detector with AdaptiveDetector (default)
    detector = SceneDetector(
        detector_type='adaptive',
        threshold=30.0,
        min_scene_len=15
    )
    
    # Use a sample video from the media directory
    video_path = 'docs/media/in.mp4'
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Sample video not found: {video_path}")
        print("Please ensure you have sample videos in docs/media/")
        return
    
    print(f"Detecting scenes in: {video_path}")
    print("This may take a while depending on video length...")
    
    # Detect scenes
    scenes = detector.detect(
        video_path,
        show_progress=True
    )
    
    # Print results
    print(f"\nDetection Results:")
    print(f"  Number of scenes detected: {len(scenes)}")
    
    if scenes:
        print(f"\nScene list:")
        for i, scene in enumerate(scenes[:5]):  # Show first 5 scenes
            print(f"  Scene {scene['scene_number']}:")
            print(f"    Start: {scene['start_timecode']} ({scene['start']:.2f}s)")
            print(f"    End: {scene['end_timecode']} ({scene['end']:.2f}s)")
            print(f"    Duration: {scene['duration']:.2f}s")
            print(f"    Frames: {scene['frame_count']}")
        
        if len(scenes) > 5:
            print(f"  ... and {len(scenes) - 5} more scenes")
    
    # Analyze scene statistics
    stats = detector.analyze_scene_statistics(scenes)
    
    print(f"\nScene Statistics:")
    print(f"  Total duration: {stats['total_duration']:.2f}s")
    print(f"  Average scene length: {stats['avg_duration']:.2f}s")
    print(f"  Shortest scene: {stats['min_duration']:.2f}s")
    print(f"  Longest scene: {stats['max_duration']:.2f}s")
    print(f"  Standard deviation: {stats['duration_std']:.2f}s")
    
    print(f"\nScene Distribution:")
    for category, count in stats['scene_distribution'].items():
        print(f"  {category.replace('_', ' ').title()}: {count} scenes")
    
    # Save results to JSON
    json_path = 'docs/example/scenes.json'
    detector.save_scenes_to_json(scenes, json_path)
    print(f"\nScene data saved to: {json_path}")
    
    return scenes


def example_video_splitting():
    """Example: Detect scenes and split video into individual files."""
    print("\n" + "=" * 60)
    print("VIDEO SPLITTING EXAMPLE")
    print("=" * 60)
    
    # Initialize detector with output directory
    output_dir = 'docs/example/scenes_output'
    detector = SceneDetector(
        detector_type='adaptive',
        threshold=30.0,
        min_scene_len=15,
        output_dir=output_dir
    )
    
    # Use a sample video from the media directory
    video_path = 'docs/media/in.mp4'
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Sample video not found: {video_path}")
        print("Please ensure you have sample videos in docs/media/")
        return
    
    print(f"Splitting video into scenes: {video_path}")
    print(f"Output directory: {output_dir}")
    print("This may take a while...")
    
    # Detect and split scenes
    result = detector.detect_and_split(
        video_path,
        output_pattern='scene_{scene_number:03d}.mp4',
        show_progress=True
    )
    
    # Print results
    print(f"\nVideo Splitting Results:")
    print(f"  Number of scenes: {result['num_scenes']}")
    print(f"  Output directory: {result['output_dir']}")
    
    if result['output_files']:
        print(f"\nGenerated scene files:")
        for i, file_path in enumerate(result['output_files'][:5]):  # Show first 5
            file_name = Path(file_path).name
            scene = result['scenes'][i]
            print(f"  {file_name}: {scene['duration']:.2f}s ({scene['frame_count']} frames)")
        
        if len(result['output_files']) > 5:
            print(f"  ... and {len(result['output_files']) - 5} more files")
    
    # Save splitting results to JSON
    json_path = 'docs/example/splitting_results.json'
    detector.save_scenes_to_json(result['scenes'], json_path)
    print(f"\nSplitting results saved to: {json_path}")
    
    return result


def example_comprehensive_report():
    """Example: Generate comprehensive scene analysis report."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE REPORT EXAMPLE")
    print("=" * 60)
    
    # Initialize detector with output directory
    output_dir = 'docs/example/scene_report'
    detector = SceneDetector(
        detector_type='adaptive',
        threshold=30.0,
        min_scene_len=15,
        output_dir=output_dir
    )
    
    # Use a sample video from the media directory
    video_path = 'docs/media/in.mp4'
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Sample video not found: {video_path}")
        print("Please ensure you have sample videos in docs/media/")
        return
    
    print(f"Generating scene report for: {video_path}")
    print(f"Report directory: {output_dir}")
    
    # Generate comprehensive report
    report = detector.generate_scene_report(video_path)
    
    print(f"\nReport Generated Successfully:")
    print(f"  JSON file: {report['json_path']}")
    print(f"  CSV file: {report['csv_path']}")
    print(f"  Total scenes: {len(report['scenes'])}")
    print(f"  Report directory: {report['output_dir']}")
    
    # Show statistics from report
    stats = report['statistics']
    print(f"\nReport Statistics:")
    print(f"  Average scene length: {stats['avg_duration']:.2f}s")
    print(f"  Scene length range: {stats['min_duration']:.2f}s to {stats['max_duration']:.2f}s")
    print(f"  Short scenes (<2s): {len(stats['short_scenes'])}")
    print(f"  Long scenes (>10s): {len(stats['long_scenes'])}")
    
    return report


def example_different_detectors():
    """Example: Compare different scene detection algorithms."""
    print("\n" + "=" * 60)
    print("DETECTOR COMPARISON EXAMPLE")
    print("=" * 60)
    
    video_path = 'docs/media/in.mp4'
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Sample video not found: {video_path}")
        print("Please ensure you have sample videos in docs/media/")
        return
    
    detectors_config = [
        ('Adaptive Detector', 'adaptive', 30.0),
        ('Content Detector', 'content', 27.0),
        ('Threshold Detector', 'threshold', 12.0)
    ]
    
    results = {}
    
    for detector_name, detector_type, threshold in detectors_config:
        print(f"\nTesting {detector_name}:")
        print(f"  Type: {detector_type}, Threshold: {threshold}")
        
        detector = SceneDetector(
            detector_type=detector_type,
            threshold=threshold,
            min_scene_len=15
        )
        
        try:
            scenes = detector.detect(video_path, show_progress=False)
            stats = detector.analyze_scene_statistics(scenes)
            
            results[detector_name] = {
                'num_scenes': len(scenes),
                'avg_duration': stats['avg_duration'],
                'min_duration': stats['min_duration'],
                'max_duration': stats['max_duration']
            }
            
            print(f"  Results: {len(scenes)} scenes, avg {stats['avg_duration']:.2f}s")
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            results[detector_name] = {'error': str(e)}
    
    print(f"\nDetector Comparison Summary:")
    print(f"{'Detector':<20} {'Scenes':<10} {'Avg Duration':<15} {'Min Duration':<15} {'Max Duration':<15}")
    print("-" * 75)
    
    for detector_name, result in results.items():
        if 'error' in result:
            print(f"{detector_name:<20} {'ERROR':<10} {'-':<15} {'-':<15} {'-':<15}")
        else:
            print(f"{detector_name:<20} {result['num_scenes']:<10} {result['avg_duration']:<15.2f} {result['min_duration']:<15.2f} {result['max_duration']:<15.2f}")
    
    return results


def example_custom_time_range():
    """Example: Detect scenes in a specific time range."""
    print("\n" + "=" * 60)
    print("CUSTOM TIME RANGE EXAMPLE")
    print("=" * 60)
    
    detector = SceneDetector(
        detector_type='adaptive',
        threshold=30.0,
        min_scene_len=10  # Shorter minimum for demo
    )
    
    video_path = 'docs/media/in.mp4'
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"Sample video not found: {video_path}")
        print("Please ensure you have sample videos in docs/media/")
        return
    
    # Define time range (first 30 seconds)
    start_time = 0.0
    end_time = 30.0
    
    print(f"Detecting scenes in first 30 seconds of: {video_path}")
    print(f"Time range: {start_time}s to {end_time}s")
    
    # Detect scenes in specific time range
    scenes = detector.detect(
        video_path,
        start_time=start_time,
        end_time=end_time,
        show_progress=True
    )
    
    print(f"\nResults for time range {start_time}s to {end_time}s:")
    print(f"  Number of scenes: {len(scenes)}")
    
    if scenes:
        print(f"\nDetected scenes:")
        for scene in scenes:
            print(f"  Scene {scene['scene_number']}: {scene['start']:.2f}s to {scene['end']:.2f}s ({scene['duration']:.2f}s)")
    
    return scenes


def main():
    """Run all examples."""
    print("Pavo Engine - Scene Detection Module Examples")
    print("=" * 60)
    
    # Create output directories if they don't exist
    os.makedirs('docs/example', exist_ok=True)
    os.makedirs('docs/example/scenes_output', exist_ok=True)
    os.makedirs('docs/example/scene_report', exist_ok=True)
    
    try:
        # Run examples
        basic_results = example_basic_detection()
        splitting_results = example_video_splitting()
        report_results = example_comprehensive_report()
        detector_comparison = example_different_detectors()
        time_range_results = example_custom_time_range()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nOutput files created:")
        print("  - docs/example/scenes.json")
        print("  - docs/example/splitting_results.json")
        print("  - docs/example/scene_report/ (directory with JSON and CSV)")
        print("  - docs/example/scenes_output/ (directory with split scene videos)")
        print("\nNext steps:")
        print("  1. Install scenedetect: pip install scenedetect[opencv]")
        print("  2. Run this example: python docs/example/SCENE_DETECTION_EXAMPLE.py")
        print("  3. Modify parameters for your specific use case")
        print("\nUse cases demonstrated:")
        print("  - Basic scene detection")
        print("  - Video splitting into scenes")
        print("  - Comprehensive scene analysis reports")
        print("  - Detector algorithm comparison")
        print("  - Scene detection in specific time ranges")
        
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install scenedetect[opencv]")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nMake sure you have sample video files in docs/media/")
        print("You can download sample videos or use your own.")


if __name__ == "__main__":
    main()