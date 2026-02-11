"""
Object detection module using YOLO models.

This module provides ObjectDetector class for detecting objects in images and videos
using Ultralytics YOLO models. It can output bounding boxes and positions of objects
in frames, which can be used as context for determining the center of the frame.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
import numpy as np
from PIL import Image
import cv2


class ObjectDetector:
    """
    A class for detecting objects in images and videos using YOLO models.
    
    This class handles object detection with support for various input formats
    and provides options for model selection, confidence threshold, and output format.
    
    Attributes:
        model_name (str): The YOLO model to use (e.g., 'yolo26n.pt', 'yolo11n.pt')
        device (str): Device to use for inference - 'cpu', 'cuda', or 'mps'
        conf_threshold (float): Confidence threshold for detections (0.0 to 1.0)
        iou_threshold (float): IoU threshold for NMS (0.0 to 1.0)
    
    Example:
        >>> detector = ObjectDetector(model_name='yolo26n.pt')
        >>> results = detector.detect_image('image.jpg')
        >>> for detection in results['detections']:
        >>>     print(f"Class: {detection['class_name']}, Confidence: {detection['confidence']}")
        >>>     print(f"Bounding box: {detection['bbox']}")
    """
    
    def __init__(
        self,
        model_name: str = 'yolo26n.pt',
        device: str = 'cpu',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None
    ):
        """
        Initialize the ObjectDetector.
        
        Args:
            model_name (str): YOLO model name or path. Options include:
                            - 'yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 'yolo26l.pt', 'yolo26x.pt'
                            - 'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'
                            - Path to custom trained model
                            Default is 'yolo26n.pt'.
            device (str): Device to use for inference - 'cpu', 'cuda', or 'mps'. Default is 'cpu'.
            conf_threshold (float): Confidence threshold for detections (0.0 to 1.0). Default is 0.25.
            iou_threshold (float): IoU threshold for NMS (0.0 to 1.0). Default is 0.45.
            classes (Optional[List[int]]): List of class IDs to detect. If None, detect all classes.
        
        Raises:
            ImportError: If ultralytics package is not installed.
            ValueError: If model_name or device is invalid.
        """
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics YOLO is not installed. Please install it using:\n"
                "pip install ultralytics"
            )
        
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self._model = None
        
        # Validate device
        if device not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu', 'cuda', or 'mps'.")
    
    def _load_model(self):
        """Load the YOLO model (lazy loading)."""
        if self._model is None:
            try:
                self._model = self.YOLO(self.model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}") from e
        return self._model
    
    def detect_image(
        self,
        image_path: Union[str, np.ndarray, Image.Image],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[List[int]] = None,
        return_image: bool = False,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Detect objects in an image.
        
        Args:
            image_path (Union[str, np.ndarray, Image.Image]): Path to image, numpy array, or PIL Image.
            conf (Optional[float]): Override confidence threshold for this detection.
            iou (Optional[float]): Override IoU threshold for this detection.
            classes (Optional[List[int]]): Override class filter for this detection.
            return_image (bool): If True, return annotated image with bounding boxes.
            verbose (bool): If True, print detection information.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'detections' (list): List of detection dictionaries, each containing:
                    - 'bbox' (list): [x1, y1, x2, y2] bounding box coordinates
                    - 'confidence' (float): Detection confidence score
                    - 'class_id' (int): Class ID
                    - 'class_name' (str): Class name
                    - 'center' (tuple): (x_center, y_center) of bounding box
                    - 'area' (float): Area of bounding box
                - 'image_info' (dict): Information about the image (width, height, channels)
                - 'annotated_image' (Optional[np.ndarray]): Image with bounding boxes if return_image=True
                - 'num_detections' (int): Total number of detections
        
        Raises:
            FileNotFoundError: If image file doesn't exist.
            ValueError: If image format is not supported.
            RuntimeError: If detection fails.
        """
        # Load image if path is provided
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Validate supported image format
            supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            file_ext = Path(image_path).suffix.lower()
            if file_ext not in supported_formats:
                raise ValueError(
                    f"Unsupported image format: {file_ext}. "
                    f"Supported formats: {', '.join(supported_formats)}"
                )
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        elif isinstance(image_path, np.ndarray):
            img_rgb = image_path
            if len(img_rgb.shape) != 3 or img_rgb.shape[2] not in [3, 4]:
                raise ValueError("Image array must be 3D with 3 or 4 channels (RGB/RGBA)")
            if img_rgb.shape[2] == 4:
                img_rgb = img_rgb[:, :, :3]  # Remove alpha channel
        
        elif isinstance(image_path, Image.Image):
            img_rgb = np.array(image_path)
            if len(img_rgb.shape) == 2:  # Grayscale
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
            elif img_rgb.shape[2] == 4:  # RGBA
                img_rgb = img_rgb[:, :, :3]
        
        else:
            raise TypeError("image_path must be str, np.ndarray, or PIL.Image")
        
        # Get image dimensions
        height, width = img_rgb.shape[:2]
        
        # Use provided parameters or fall back to instance defaults
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        classes = classes or self.classes
        
        try:
            # Load model
            model = self._load_model()
            
            # Run detection
            results = model(
                img_rgb,
                conf=conf,
                iou=iou,
                classes=classes,
                device=self.device,
                verbose=verbose
            )
            
            # Process results
            detections = []
            annotated_img = None
            
            if results and len(results) > 0:
                result = results[0]
                
                # Create annotated image if requested
                if return_image:
                    annotated_img = result.plot()
                    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                
                # Extract detections
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    # Get class names
                    class_names = result.names
                    
                    for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        area = (x2 - x1) * (y2 - y1)
                        
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence),
                            'class_id': int(class_id),
                            'class_name': class_names[class_id],
                            'center': (float(center_x), float(center_y)),
                            'area': float(area),
                            'normalized_center': (float(center_x / width), float(center_y / height))
                        }
                        detections.append(detection)
            
            return {
                'detections': detections,
                'image_info': {
                    'width': width,
                    'height': height,
                    'channels': img_rgb.shape[2] if len(img_rgb.shape) > 2 else 1
                },
                'annotated_image': annotated_img,
                'num_detections': len(detections)
            }
        
        except Exception as e:
            raise RuntimeError(f"Object detection failed: {str(e)}") from e
    
    def detect_video(
        self,
        video_path: str,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[List[int]] = None,
        frame_interval: int = 1,
        output_path: Optional[str] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Detect objects in a video file.
        
        Args:
            video_path (str): Path to the video file.
            conf (Optional[float]): Override confidence threshold for this detection.
            iou (Optional[float]): Override IoU threshold for this detection.
            classes (Optional[List[int]]): Override class filter for this detection.
            frame_interval (int): Process every Nth frame. Default is 1 (every frame).
            output_path (Optional[str]): Path to save annotated video. If None, no video is saved.
            verbose (bool): If True, print progress information.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'frame_detections' (list): List of detection results for each processed frame
                - 'video_info' (dict): Information about the video (fps, width, height, total_frames)
                - 'output_path' (Optional[str]): Path to saved annotated video if output_path was provided
                - 'summary' (dict): Summary statistics (total_detections, avg_detections_per_frame, etc.)
        
        Raises:
            FileNotFoundError: If video file doesn't exist.
            ValueError: If video format is not supported.
            RuntimeError: If detection fails.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Validate supported video format
        supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'}
        file_ext = Path(video_path).suffix.lower()
        if file_ext not in supported_formats:
            raise ValueError(
                f"Unsupported video format: {file_ext}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
        
        # Use provided parameters or fall back to instance defaults
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold
        classes = classes or self.classes
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_detections = []
        frame_count = 0
        processed_count = 0
        total_detections = 0
        
        try:
            # Load model
            model = self._load_model()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames based on interval
                if (frame_count - 1) % frame_interval != 0:
                    continue
                
                processed_count += 1
                
                if verbose and processed_count % 10 == 0:
                    print(f"Processing frame {frame_count}/{total_frames}...")
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run detection
                results = model(
                    frame_rgb,
                    conf=conf,
                    iou=iou,
                    classes=classes,
                    device=self.device,
                    verbose=False
                )
                
                # Process results
                detections = []
                annotated_frame = frame.copy()
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    # Extract detections
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)
                        class_names = result.names
                        
                        for i, (box, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                            x1, y1, x2, y2 = box
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            area = (x2 - x1) * (y2 - y1)
                            
                            detection = {
                                'frame_number': frame_count,
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(confidence),
                                'class_id': int(class_id),
                                'class_name': class_names[class_id],
                                'center': (float(center_x), float(center_y)),
                                'area': float(area),
                                'normalized_center': (float(center_x / width), float(center_y / height))
                            }
                            detections.append(detection)
                        
                        total_detections += len(detections)
                    
                    # Create annotated frame
                    if output_path:
                        annotated_frame = result.plot()
                        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                frame_result = {
                    'frame_number': frame_count,
                    'detections': detections,
                    'num_detections': len(detections)
                }
                frame_detections.append(frame_result)
                
                # Write annotated frame if writer exists
                if writer is not None:
                    writer.write(annotated_frame)
            
            # Release resources
            cap.release()
            if writer is not None:
                writer.release()
            
            # Calculate summary statistics
            avg_detections = total_detections / max(processed_count, 1)
            
            return {
                'frame_detections': frame_detections,
                'video_info': {
                    'fps': fps,
                    'width': width,
                    'height': height,
                    'total_frames': total_frames,
                    'processed_frames': processed_count
                },
                'output_path': output_path,
                'summary': {
                    'total_detections': total_detections,
                    'avg_detections_per_frame': avg_detections,
                    'frames_with_detections': sum(1 for fd in frame_detections if fd['num_detections'] > 0),
                    'total_frames_processed': processed_count
                }
            }
        
        except Exception as e:
            # Clean up resources
            cap.release()
            if writer is not None:
                writer.release()
            raise RuntimeError(f"Video detection failed: {str(e)}") from e
    
    def detect_batch(
        self,
        image_paths: List[str],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        classes: Optional[List[int]] = None,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in multiple images.
        
        Args:
            image_paths (List[str]): List of paths to image files.
            conf (Optional[float]): Override confidence threshold for this detection.
            iou (Optional[float]): Override IoU threshold for this detection.
            classes (Optional[List[int]]): Override class filter for this detection.
            verbose (bool): If True, print progress information.
        
        Returns:
            List[Dict[str, Any]]: List of detection results for each image.
        
        Raises:
            FileNotFoundError: If any image file doesn't exist.
            ValueError: If any image format is not supported.
            RuntimeError: If detection fails.
        """
        results = []
        for i, image_path in enumerate(image_paths):
            if verbose:
                print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                result = self.detect_image(
                    image_path,
                    conf=conf,
                    iou=iou,
                    classes=classes,
                    return_image=False,
                    verbose=False
                )
                results.append(result)
            except Exception as e:
                if verbose:
                    print(f"Failed to process {image_path}: {str(e)}")
                results.append({
                    'error': str(e),
                    'image_path': image_path,
                    'detections': [],
                    'num_detections': 0
                })
        
        return results
    
    def get_frame_center_context(
        self,
        detections: List[Dict[str, Any]],
        image_width: int,
        image_height: int
    ) -> Dict[str, Any]:
        """
        Analyze detections to provide context about the center of the frame.
        
        This method analyzes object positions relative to the frame center,
        which can be useful for determining if objects are centered, off-center,
        or if the frame needs adjustment.
        
        Args:
            detections (List[Dict[str, Any]]): List of detection dictionaries.
            image_width (int): Width of the image/frame.
            image_height (int): Height of the image/frame.
        
        Returns:
            Dict[str, Any]: Context information including:
                - 'frame_center' (tuple): (x_center, y_center) of the frame
                - 'objects_near_center' (list): Objects within center region
                - 'dominant_object' (Optional[dict]): Largest object by area
                - 'center_of_mass' (tuple): Weighted center of all objects
                - 'recommendation' (str): Suggested action based on analysis
        """
        frame_center_x = image_width / 2
        frame_center_y = image_height / 2
        
        # Define center region (20% of frame dimensions from center)
        center_region_width = image_width * 0.2
        center_region_height = image_height * 0.2
        
        objects_near_center = []
        total_area = 0
        weighted_x = 0
        weighted_y = 0
        largest_object = None
        max_area = 0
        
        for detection in detections:
            center_x, center_y = detection['center']
            area = detection['area']
            
            # Check if object is near center
            if (abs(center_x - frame_center_x) < center_region_width / 2 and
                abs(center_y - frame_center_y) < center_region_height / 2):
                objects_near_center.append(detection)
            
            # Track largest object
            if area > max_area:
                max_area = area
                largest_object = detection
            
            # Calculate weighted center (center of mass)
            total_area += area
            weighted_x += center_x * area
            weighted_y += center_y * area
        
        # Calculate center of mass
        center_of_mass = None
        if total_area > 0:
            center_of_mass = (weighted_x / total_area, weighted_y / total_area)
        
        # Generate recommendation
        recommendation = "Frame composition looks good."
        if not detections:
            recommendation = "No objects detected. Consider adjusting framing."
        elif not objects_near_center:
            recommendation = "No objects near center. Consider panning or zooming."
        elif len(objects_near_center) == 1:
            recommendation = "Single object near center. Good for focus."
        else:
            recommendation = "Multiple objects near center. Good composition."
        
        return {
            'frame_center': (frame_center_x, frame_center_y),
            'objects_near_center': objects_near_center,
            'dominant_object': largest_object,
            'center_of_mass': center_of_mass,
            'num_objects_near_center': len(objects_near_center),
            'total_objects': len(detections),
            'recommendation': recommendation
        }
    
    def save_detections_to_json(
        self,
        detections: Dict[str, Any],
        output_path: str
    ):
        """
        Save detection results to a JSON file.
        
        Args:
            detections (Dict[str, Any]): Detection results from detect_image or detect_video.
            output_path (str): Path to save JSON file.
        
        Raises:
            ValueError: If output_path is not a .json file.
            RuntimeError: If saving fails.
        """
        import json
        
        if not output_path.lower().endswith('.json'):
            raise ValueError("Output path must be a .json file")
        
        try:
            # Convert numpy types to Python native types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            serializable_detections = convert_to_serializable(detections)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_detections, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            raise RuntimeError(f"Failed to save detections to JSON: {str(e)}") from e
    
    def unload_model(self):
        """Unload the model to free up memory."""
        self._model = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model to free resources."""
        self.unload_model()
        return False
