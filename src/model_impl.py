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
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


class My_LicensePlate_Model:
    """
    YOLO-based license plate detection model.
    """
    
    def __init__(
        self,
        model_path: str = "weights/best.pt",
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        self.logger = logging.getLogger(__name__)
        self.model_path = Path(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        if not self.model_path.exists():
            self.logger.warning(f"Model not found at {model_path}, using pretrained")
            self.model = YOLO("yolov8n.pt")
        else:
            self.model = YOLO(str(self.model_path))
        
        self.model.to(device)
        self.logger.info(f"Model loaded on {device}")
    
    def detect_plates(self, frame: np.ndarray) -> List[Dict]:
        if frame is None or frame.size == 0:
            self.logger.error("Empty frame provided")
            return []
        
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
        start_time = time.perf_counter()
        detections = self.detect_plates(frame)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return detections, latency_ms
    
    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes on frame"""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Draw rectangle only
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return annotated