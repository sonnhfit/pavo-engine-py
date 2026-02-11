"""
Object detection module using YOLO models.

This module provides ObjectDetector class for detecting objects in images and videos
using Ultralytics YOLO models.
"""

from .detector import ObjectDetector

__all__ = ['ObjectDetector']