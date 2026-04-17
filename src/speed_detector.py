"""
Vehicle speed detection using license plate tracking across frames
"""

import logging
from typing import List, Dict, Tuple, Optional
from collections import deque
import math

import numpy as np
import cv2


class SpeedDetector:
    """
    Estimate vehicle speed by tracking license plates across frames.
    
    Requires camera calibration (pixels to meters conversion) and frame rate.
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        pixels_per_meter: float = 50.0,  # Calibration: how many pixels = 1 meter
        tracking_frames: int = 15,  # Number of frames to track
        speed_limit_kmh: float = 60.0  # For speed violation detection
    ):
        """
        Initialize speed detector.
        
        Args:
            fps: Frames per second of video stream
            pixels_per_meter: Camera calibration factor
            tracking_frames: Number of frames to track for speed calculation
            speed_limit_kmh: Speed limit for violation alerts
        """
        self.logger = logging.getLogger(__name__)
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.tracking_frames = tracking_frames
        self.speed_limit_kmh = speed_limit_kmh
        
        # Track positions of plates by ID
        self.tracked_plates: Dict[str, deque] = {}  # plate_id -> deque of (frame_idx, center_x, center_y)
        self.frame_counter = 0
    
    def _get_plate_center(self, bbox: List[int]) -> Tuple[float, float]:
        """Calculate center of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_speed(
        self,
        positions: List[Tuple[int, Tuple[float, float]]],
        fps: float,
        pixels_per_meter: float
    ) -> float:
        """
        Calculate speed in km/h from tracked positions.
        
        Args:
            positions: List of (frame_idx, (x, y)) positions
            fps: Frames per second
            pixels_per_meter: Conversion factor
            
        Returns:
            Speed in km/h
        """
        if len(positions) < 2:
            return 0.0
        
        # Calculate total distance in pixels
        total_distance_px = 0.0
        for i in range(1, len(positions)):
            _, (x1, y1) = positions[i-1]
            _, (x2, y2) = positions[i]
            distance_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance_px += distance_px
        
        # Convert to meters
        total_distance_m = total_distance_px / pixels_per_meter
        
        # Calculate time in hours
        first_frame, _ = positions[0]
        last_frame, _ = positions[-1]
        time_seconds = (last_frame - first_frame) / fps
        time_hours = time_seconds / 3600
        
        # Calculate speed (convert m/s to km/h)
        if time_hours > 0:
            speed_kmh = (total_distance_m / 1000) / time_hours  # Convert m to km
        else:
            speed_kmh = 0.0
        
        return speed_kmh
    
    def update(
        self,
        detections: List[Dict],
        plate_ids: List[str]
    ) -> List[Dict]:
        """
        Update tracking and calculate speeds for detected plates.
        
        Args:
            detections: List of detection dictionaries with 'bbox' keys
            plate_ids: List of unique IDs for each plate (from OCR or tracking)
            
        Returns:
            Detections enriched with 'speed_kmh' field
        """
        self.frame_counter += 1
        
        for det, plate_id in zip(detections, plate_ids):
            if plate_id not in self.tracked_plates:
                self.tracked_plates[plate_id] = deque(maxlen=self.tracking_frames)
            
            # Get center of plate
            center = self._get_plate_center(det['bbox'])
            self.tracked_plates[plate_id].append((self.frame_counter, center))
            
            # Calculate speed if we have enough history
            if len(self.tracked_plates[plate_id]) >= 2:
                positions = list(self.tracked_plates[plate_id])
                speed = self._calculate_speed(
                    positions, self.fps, self.pixels_per_meter
                )
                det['speed_kmh'] = round(speed, 1)
                det['speed_violation'] = speed > self.speed_limit_kmh
            else:
                det['speed_kmh'] = 0.0
                det['speed_violation'] = False
        
        return detections
    
    def draw_speed_info(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """Draw speed information on frame"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            if 'speed_kmh' in det:
                speed = det['speed_kmh']
                color = (0, 0, 255) if det.get('speed_violation', False) else (255, 255, 0)
                text = f"Speed: {speed} km/h"
                # Draw background for better visibility
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y2 + 5), (x1 + text_w + 10, y2 + text_h + 15), (0, 0, 0), -1)
                cv2.putText(annotated, text, (x1 + 5, y2 + text_h + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw speed limit info on top
        limit_text = f"SPEED LIMIT: {self.speed_limit_kmh} km/h"
        (text_w, text_h), _ = cv2.getTextSize(limit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (10, 60), (10 + text_w + 20, 60 + text_h + 10), (0, 0, 0), -1)
        cv2.putText(annotated, limit_text, (20, 60 + text_h),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated
    
    def reset(self):
        """Reset all tracking data"""
        self.tracked_plates.clear()
        self.frame_counter = 0
        self.logger.info("Speed detector reset")
    
    def get_average_speed(self, plate_id: str) -> float:
        """
        Get average speed for a specific plate.
        
        Args:
            plate_id: Unique identifier for the plate
            
        Returns:
            Average speed in km/h, or 0 if not found
        """
        if plate_id not in self.tracked_plates:
            return 0.0
        
        positions = list(self.tracked_plates[plate_id])
        if len(positions) < 2:
            return 0.0
        
        return self._calculate_speed(positions, self.fps, self.pixels_per_meter)
