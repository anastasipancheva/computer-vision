

"""
License Plate Detection Model Implementation
Uses YOLO architecture for real-time license plate detection
"""

import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectionResult:
    """Container for detection results"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str


class My_LicensePlate_Model:
    """
    YOLO-based license plate detection model.
    
    Attributes:
        model_path: Path to trained YOLO weights
        device: Device to run inference on ('cuda' or 'cpu')
        conf_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
    """
    
    def __init__(
        self,
        model_path: str = "weights/best.pt",
        device: str = "cuda",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize the license plate detector.
        
        Args:
            model_path: Path to YOLO weights file
            device: Device for inference ('cuda' or 'cpu')
            conf_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for non-maximum suppression
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLO model
        if not self.model_path.exists():
            self.logger.warning(f"Model not found at {model_path}, using pretrained")
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(str(self.model_path))
        
        # Move to appropriate device
        self.model.to(device)
        self.logger.info(f"Model loaded on {device}")
    
    def detect_plates(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect license plates in a single frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            List of dictionaries containing:
                - 'bbox': [x1, y1, x2, y2] coordinates
                - 'confidence': detection confidence score
                - 'class': class name (default 'license_plate')
        """
        if frame is None or frame.size == 0:
            self.logger.error("Empty frame provided")
            return []
        
        # Run inference
        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': confidence,
                    'class': self.model.names.get(class_id, 'license_plate')
                })
        
        self.logger.debug(f"Detected {len(detections)} license plates")
        return detections
    
    def detect_with_latency(self, frame: np.ndarray) -> Tuple[List[Dict], float]:
        """
        Detect plates and measure inference latency.
        
        Args:
            frame: Input image
            
        Returns:
            Tuple of (detections, latency_ms)
        """
        start_time = time.perf_counter()
        detections = self.detect_plates(frame)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return detections, latency_ms
    
    def process_stream(self, frame_callback, show_fps: bool = True):
        """
        Process a stream of frames (from video or camera).
        
        Args:
            frame_callback: Generator yielding frames
            show_fps: Whether to calculate and display FPS
        """
        frame_times = []
        
        for frame in frame_callback:
            detections, latency = self.detect_with_latency(frame)
            
            # Annotate frame
            annotated = self._annotate_frame(frame, detections)
            
            if show_fps:
                frame_times.append(latency)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                avg_latency = np.mean(frame_times)
                fps = 1000 / avg_latency if avg_latency > 0 else 0
                cv2.putText(
                    annotated, f"FPS: {fps:.1f} | Latency: {avg_latency:.1f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            
            yield annotated, detections
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes on frame"""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Plate: {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated