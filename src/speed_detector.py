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
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        pixels_per_meter: float = 25.0,  # УВЕЛИЧИЛ чувствительность (было 50)
        tracking_frames: int = 20,  # УВЕЛИЧИЛ количество кадров для отслеживания
        speed_limit_kmh: float = 60.0,
        min_speed: float = 5.0  # Минимальная скорость для отображения
    ):
        self.logger = logging.getLogger(__name__)
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.tracking_frames = tracking_frames
        self.speed_limit_kmh = speed_limit_kmh
        self.min_speed = min_speed
        
        self.tracked_plates: Dict[str, deque] = {}
        self.frame_counter = 0
        self.last_positions: Dict[str, Tuple[float, float]] = {}  # Для сглаживания
    
    def _get_plate_center(self, bbox: List[int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _calculate_speed(
        self,
        positions: List[Tuple[int, Tuple[float, float]]],
        fps: float,
        pixels_per_meter: float
    ) -> float:
        """Улучшенный расчет скорости"""
        if len(positions) < 3:  # Нужно минимум 3 точки для точности
            return 0.0
        
        # Используем скользящее среднее для сглаживания
        speeds = []
        for i in range(1, len(positions)):
            frame1, (x1, y1) = positions[i-1]
            frame2, (x2, y2) = positions[i]
            
            distance_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            time_seconds = (frame2 - frame1) / fps
            
            if time_seconds > 0 and distance_px > 5:  # Игнорируем маленькие перемещения
                speed_ms = (distance_px / pixels_per_meter) / time_seconds
                speed_kmh = speed_ms * 3.6
                if 0 < speed_kmh < 200:  # Реалистичный диапазон
                    speeds.append(speed_kmh)
        
        if not speeds:
            return 0.0
        
        # Возвращаем медиану (устойчива к выбросам)
        return round(np.median(speeds), 1)
    
    def update(
        self,
        detections: List[Dict],
        plate_ids: List[str]
    ) -> List[Dict]:
        """Обновление отслеживания с улучшенной логикой"""
        self.frame_counter += 1
        
        for det, plate_id in zip(detections, plate_ids):
            if not plate_id or plate_id == "":
                plate_id = f"unknown_{self.frame_counter}_{det['bbox'][0]}"
            
            if plate_id not in self.tracked_plates:
                self.tracked_plates[plate_id] = deque(maxlen=self.tracking_frames)
            
            center = self._get_plate_center(det['bbox'])
            self.tracked_plates[plate_id].append((self.frame_counter, center))
            
            # Расчет скорости
            if len(self.tracked_plates[plate_id]) >= 3:
                positions = list(self.tracked_plates[plate_id])
                speed = self._calculate_speed(
                    positions, self.fps, self.pixels_per_meter
                )
                det['speed_kmh'] = speed
                det['speed_violation'] = speed > self.speed_limit_kmh and speed > self.min_speed
                
                # Логирование для отладки
                if speed > 10:
                    self.logger.debug(f"Plate {plate_id}: speed={speed} km/h, positions={len(positions)}")
            else:
                det['speed_kmh'] = 0.0
                det['speed_violation'] = False
        
        return detections
    
    def draw_speed_info(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """Улучшенная отрисовка информации о скорости"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            if 'speed_kmh' in det and det['speed_kmh'] > 0:
                speed = det['speed_kmh']
                color = (0, 0, 255) if det.get('speed_violation', False) else (255, 255, 0)
                
                # Текст скорости
                text = f"{speed} km/h"
                
                # Рисуем фон для текста
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y2 + 5), (x1 + text_w + 10, y2 + text_h + 15), (0, 0, 0), -1)
                cv2.putText(annotated, text, (x1 + 5, y2 + text_h + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Иконка радара
                cv2.circle(annotated, (x1 - 10, y1 + 10), 5, (0, 255, 255), -1)
        
        # Информация о лимите скорости
        limit_text = f"SPEED LIMIT: {self.speed_limit_kmh} km/h"
        (text_w, text_h), _ = cv2.getTextSize(limit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (10, 60), (10 + text_w + 20, 60 + text_h + 10), (0, 0, 0), -1)
        cv2.putText(annotated, limit_text, (20, 60 + text_h),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return annotated
    
    def reset(self):
        self.tracked_plates.clear()
        self.frame_counter = 0
        self.last_positions.clear()
        self.logger.info("Speed detector reset")