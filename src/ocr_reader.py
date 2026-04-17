

"""
OCR module for license plate text recognition
Supports both Cyrillic and Latin characters
"""

import logging
from typing import List, Dict, Tuple, Optional
import re

import cv2
import numpy as np
import easyocr


class LicensePlateOCR:
    """
    OCR reader for extracting text from license plates.
    Supports Russian (Cyrillic) and English (Latin) characters.
    """
    
    def __init__(
        self,
        languages: List[str] = ['ru', 'en'],  # Russian first, then English
        gpu: bool = True,
        confidence_threshold: float = 0.4  # Lower threshold for better recognition
    ):
        """
        Initialize OCR reader.
        
        Args:
            languages: List of languages to recognize (ru for Russian, en for English)
            gpu: Use GPU acceleration if available
            confidence_threshold: Minimum confidence for text recognition
        """
        self.logger = logging.getLogger(__name__)
        self.languages = languages
        self.confidence_threshold = confidence_threshold
        
        # Cyrillic to Latin mapping for Russian plates
        self.cyrillic_to_latin = {
            'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M',
            'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T',
            'У': 'Y', 'Х': 'X'
        }
        
        # Latin to Cyrillic mapping
        self.latin_to_cyrillic = {v: k for k, v in self.cyrillic_to_latin.items()}
        
        # Allowed characters for Russian plates
        self.allowed_cyrillic = set('АВЕКМНОРСТУХ')
        self.allowed_latin = set('ABEKMHOPCTYX')
        
        # Initialize EasyOCR reader
        try:
            self.reader = easyocr.Reader(
                languages,
                gpu=gpu,
                verbose=False,
                model_storage_directory='./data/ocr_models'  # Cache models
            )
            self.logger.info(f"OCR initialized with languages: {languages}, GPU: {gpu}")
        except Exception as e:
            self.logger.error(f"Failed to initialize EasyOCR: {e}")
            self.reader = None
    
    def preprocess_plate(
        self,
        plate_image: np.ndarray,
        resize_factor: float = 2.0
    ) -> np.ndarray:
        """Preprocess license plate image for better OCR."""
        if plate_image is None or plate_image.size == 0:
            return np.array([])
        
        # Convert to grayscale
        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Resize
        height, width = denoised.shape
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        resized = cv2.resize(denoised, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def normalize_plate_text(self, text: str) -> str:
        """
        Normalize license plate text to standard format.
        Handles both Cyrillic and Latin characters.
        """
        text = text.upper().strip()
        
        # Remove spaces and special characters
        text = re.sub(r'[^A-ZА-Я0-9]', '', text)
        
        # Russian plate pattern: Letter + 3 digits + 2-3 letters + 2-3 digits
        # Example: А123ВС777 or A123BC777
        
        # Try to detect if it's a Russian plate
        has_cyrillic = any(c in self.allowed_cyrillic for c in text)
        has_latin = any(c in self.allowed_latin for c in text)
        
        if has_cyrillic:
            # Keep as Cyrillic (Russian)
            return self._format_russian_plate(text)
        elif has_latin:
            # Convert Latin to Cyrillic if it looks like Russian plate
            converted = self._latin_to_cyrillic(text)
            if self._is_valid_russian_format(converted):
                return converted
            return text
        else:
            return text
    
    def _latin_to_cyrillic(self, text: str) -> str:
        """Convert Latin characters to Cyrillic where applicable."""
        result = []
        for char in text:
            if char in self.latin_to_cyrillic:
                result.append(self.latin_to_cyrillic[char])
            else:
                result.append(char)
        return ''.join(result)
    
    def _cyrillic_to_latin(self, text: str) -> str:
        """Convert Cyrillic to Latin."""
        result = []
        for char in text:
            if char in self.cyrillic_to_latin:
                result.append(self.cyrillic_to_latin[char])
            else:
                result.append(char)
        return ''.join(result)
    
    def _is_valid_russian_format(self, text: str) -> bool:
        """Check if text matches Russian license plate format."""
        # Format: 1 letter + 3 digits + 2-3 letters + 2-3 digits
        pattern = r'^[АВЕКМНОРСТУХ]\d{3}[АВЕКМНОРСТУХ]{2}\d{2,3}$'
        return bool(re.match(pattern, text))
    
    def _format_russian_plate(self, text: str) -> str:
        """Format text as Russian license plate."""
        # Try to extract valid pattern
        pattern = r'([АВЕКМНОРСТУХ])(\d{3})([АВЕКМНОРСТУХ]{2})(\d{2,3})'
        match = re.search(pattern, text)
        
        if match:
            return f"{match.group(1)}{match.group(2)}{match.group(3)}{match.group(4)}"
        
        # If no pattern match, try to clean up
        # Keep only allowed characters
        cleaned = ''.join([c for c in text if c in self.allowed_cyrillic or c.isdigit()])
        
        # Try to reconstruct
        if len(cleaned) >= 6:
            # Find first letter
            first_letter = ''
            for i, c in enumerate(cleaned):
                if c in self.allowed_cyrillic:
                    first_letter = c
                    cleaned = cleaned[i:]
                    break
            
            # Find digits for region code at the end
            region = ''
            for i in range(len(cleaned)-1, -1, -1):
                if cleaned[i].isdigit():
                    region = cleaned[i] + region
                else:
                    break
            
            if len(region) >= 2:
                cleaned = cleaned[:len(cleaned)-len(region)]
                return f"{first_letter}{cleaned}{region}"
        
        return cleaned
    
    def read_plate(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Read text from a license plate image.
        Returns both original and normalized text.
        """
        if self.reader is None:
            self.logger.error("OCR reader not initialized")
            return "", 0.0
        
        if plate_image is None or plate_image.size == 0:
            return "", 0.0
        
        try:
            # Preprocess image
            processed = self.preprocess_plate(plate_image)
            
            if processed.size == 0:
                return "", 0.0
            
            # Run OCR with both languages
            results = self.reader.readtext(processed, paragraph=False)
            
            # Also try on original image
            if not results:
                results = self.reader.readtext(plate_image, paragraph=False)
            
            if results:
                # Get best result
                best_result = max(results, key=lambda x: x[2])
                text, confidence = best_result[1], best_result[2]
                
                if confidence >= self.confidence_threshold:
                    # Normalize text (handles both alphabets)
                    normalized = self.normalize_plate_text(text)
                    
                    # Also get Latin version
                    latin_version = self._cyrillic_to_latin(normalized)
                    
                    if normalized:
                        self.logger.debug(f"OCR: '{text}' -> '{normalized}' (latin: '{latin_version}', conf: {confidence:.2f})")
                        return normalized, confidence
            
            return "", 0.0
            
        except Exception as e:
            self.logger.error(f"OCR failed: {e}")
            return "", 0.0
    
    def read_plate_both(self, plate_image: np.ndarray) -> Tuple[str, str, float]:
        """
        Read license plate and return both Cyrillic and Latin versions.
        
        Returns:
            Tuple of (cyrillic_text, latin_text, confidence)
        """
        cyrillic_text, confidence = self.read_plate(plate_image)
        latin_text = self._cyrillic_to_latin(cyrillic_text)
        return cyrillic_text, latin_text, confidence
    
    def process_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        use_multiple_attempts: bool = False
    ) -> List[Dict]:
        """Process all detected plates in a frame."""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Add padding
            padding = 15
            h, w = frame.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            plate_crop = frame[y1:y2, x1:x2]
            
            if plate_crop.size > 0:
                # Get both Cyrillic and Latin versions
                cyrillic, latin, confidence = self.read_plate_both(plate_crop)
                det['plate_text'] = cyrillic
                det['plate_text_latin'] = latin
                det['ocr_confidence'] = confidence
            else:
                det['plate_text'] = ""
                det['plate_text_latin'] = ""
                det['ocr_confidence'] = 0.0
        
        return detections
    
    def draw_ocr_results(
        self,
        frame: np.ndarray,
        detections: List[Dict]
    ) -> np.ndarray:
        """Draw OCR results on frame (shows both versions)."""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            if 'plate_text' in det and det['plate_text']:
                cyrillic = det['plate_text']
                latin = det.get('plate_text_latin', cyrillic)
                confidence = det.get('ocr_confidence', 0.0)
                
                # Show both versions
                label = f"🚗 {cyrillic} / {latin} ({confidence:.2f})"
                
                # Draw background
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1 - 5), (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(annotated, label, (x1 + 5, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return annotated