








"""
Main CLI entry point for license plate detection system
Supports video file processing and live camera stream
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Generator, Optional

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model_impl import My_LicensePlate_Model
from speed_detector import SpeedDetector
from ocr_reader import LicensePlateOCR


def setup_logging(log_path: Path = Path("data/log_file.log")):
    """Setup logging configuration"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def video_frame_generator(video_path: str) -> Generator[np.ndarray, None, None]:
    """Generator that yields frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    
    cap.release()


def camera_frame_generator(camera_id: int = 0) -> Generator[np.ndarray, None, None]:
    """Generator that yields frames from webcam"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Cannot open camera {camera_id}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    
    cap.release()


def process_video_mode(
    input_path: str,
    output_path: Optional[str],
    model: My_LicensePlate_Model,
    speed_detector: Optional[SpeedDetector],
    ocr: Optional[LicensePlateOCR],
    logger: logging.Logger
):
    """Process video file mode"""
    logger.info(f"Processing video: {input_path}")
    
    # Get video properties for output
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    logger.info(f"Video info: {width}x{height}, {fps:.2f} fps, {total_frames} frames")
    
    # Setup video writer
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        logger.info(f"Output will be saved to: {output_path}")
    
    frame_count = 0
    frame_generator = video_frame_generator(input_path)
    total_latency = 0.0
    
    for frame in frame_generator:
        # Detect plates
        detections, latency = model.detect_with_latency(frame)
        total_latency += latency
        
        # Apply OCR if available
        if ocr:
            detections = ocr.process_detections(frame, detections)
        
        # Apply speed detection if available
        if speed_detector:
            # For speed tracking, use plate text or position-based ID
            plate_ids = []
            for i, d in enumerate(detections):
                if 'plate_text' in d and d['plate_text']:
                    plate_ids.append(d['plate_text'])
                else:
                    # Use position-based ID as fallback
                    x1, y1, x2, y2 = d['bbox']
                    plate_ids.append(f"pos_{x1}_{y1}_{i}")
            detections = speed_detector.update(detections, plate_ids)
        
        # Annotate frame
        annotated = model._annotate_frame(frame, detections)
        
        if ocr:
            annotated = ocr.draw_ocr_results(annotated, detections)
        if speed_detector:
            annotated = speed_detector.draw_speed_info(annotated, detections)
        
        # Add info panel
        info_text = f"Frame: {frame_count}/{total_frames} | Latency: {latency:.1f}ms"
        cv2.putText(annotated, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Write or display
        if out:
            out.write(annotated)
        else:
            cv2.imshow('License Plate Detection', annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        if frame_count % 100 == 0:
            avg_latency = total_latency / frame_count
            logger.info(f"Processed {frame_count}/{total_frames} frames (avg latency: {avg_latency:.1f}ms)")
    
    if out:
        out.release()
        logger.info(f"Output saved to {output_path}")
    else:
        cv2.destroyAllWindows()
    
    avg_latency = total_latency / frame_count if frame_count > 0 else 0
    logger.info(f"Finished processing {frame_count} frames (avg latency: {avg_latency:.1f}ms)")


def process_camera_mode(
    camera_id: int,
    model: My_LicensePlate_Model,
    speed_detector: Optional[SpeedDetector],
    ocr: Optional[LicensePlateOCR],
    logger: logging.Logger
):
    """Process live camera stream mode"""
    logger.info(f"Starting camera stream from camera {camera_id}")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Cannot open camera {camera_id}")
    
    # Get actual FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default assumption
        logger.warning(f"Could not detect FPS, using default: {fps}")
    
    logger.info(f"Camera FPS: {fps}")
    
    # Update speed detector with actual FPS
    if speed_detector:
        speed_detector.fps = fps
    
    frame_count = 0
    frame_times = []
    
    # Get frame height for info display
    ret, frame = cap.read()
    if ret:
        height = frame.shape[0]
    else:
        height = 480
    cap.release()
    
    try:
        for frame in camera_frame_generator(camera_id):
            # Detect plates with latency measurement
            import time
            start_time = time.perf_counter()
            detections = model.detect_plates(frame)
            latency = (time.perf_counter() - start_time) * 1000
            
            # Track FPS
            frame_times.append(latency)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_latency = sum(frame_times) / len(frame_times)
            current_fps = 1000 / avg_latency if avg_latency > 0 else 0
            
            # Apply OCR if available
            if ocr:
                detections = ocr.process_detections(frame, detections, use_multiple_attempts=False)
            
            # Apply speed detection if available
            if speed_detector:
                plate_ids = []
                for i, d in enumerate(detections):
                    if 'plate_text' in d and d['plate_text']:
                        plate_ids.append(d['plate_text'])
                    else:
                        x1, y1, x2, y2 = d['bbox']
                        plate_ids.append(f"cam_{x1}_{y1}_{i}")
                detections = speed_detector.update(detections, plate_ids)
            
            # Annotate frame
            annotated = model._annotate_frame(frame, detections)
            
            if ocr:
                annotated = ocr.draw_ocr_results(annotated, detections)
            if speed_detector:
                annotated = speed_detector.draw_speed_info(annotated, detections)
            
            # Add info panel
            info_text = f"LIVE | FPS: {current_fps:.1f} | Latency: {latency:.1f}ms | Frames: {frame_count}"
            cv2.putText(annotated, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add instructions
            cv2.putText(annotated, "Press 'q' to quit | 'r' to reset speed tracking",
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('License Plate Detection - LIVE', annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and speed_detector:
                speed_detector.reset()
                logger.info("Speed tracking reset")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cv2.destroyAllWindows()
        logger.info(f"Camera stream stopped. Processed {frame_count} frames")


def cli():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='License Plate Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video file with OCR and speed detection
  python src/main.py --mode video --input data/videos/test.mp4 --output result.mp4 --ocr --speed
  
  # Process video without saving (display only)
  python src/main.py --mode video --input data/videos/test.mp4 --ocr
  
  # Live camera stream with speed detection
  python src/main.py --mode camera --camera-id 0 --speed --speed-limit 50
  
  # Live camera with OCR (recognize plate numbers)
  python src/main.py --mode camera --ocr --conf 0.3
  
  # Using custom model weights
  python src/main.py --mode video --input test.mp4 --weights weights/best.pt
        """
    )
    
    parser.add_argument('--mode', choices=['video', 'camera'], required=True,
                       help='Processing mode: video file or live camera')
    parser.add_argument('--input', type=str, help='Input video file path (required for video mode)')
    parser.add_argument('--output', type=str, help='Output video file path (optional, for video mode)')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID (for camera mode, default: 0)')
    parser.add_argument('--weights', type=str, default='weights/best.pt',
                       help='Path to model weights (default: weights/best.pt)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for inference: cuda or cpu (default: cpu)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (default: 0.45)')
    
    # Feature flags
    parser.add_argument('--speed', action='store_true',
                       help='Enable speed detection')
    parser.add_argument('--ocr', action='store_true',
                       help='Enable OCR for plate text recognition')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable display (for headless environments)')
    
    # Speed detection parameters
    parser.add_argument('--speed-limit', type=float, default=60.0,
                       help='Speed limit in km/h (default: 60.0)')
    parser.add_argument('--pixels-per-meter', type=float, default=50.0,
                       help='Camera calibration: pixels per meter (default: 50.0)')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='FPS for speed calculation (default: 30.0)')
    
    # OCR parameters
    parser.add_argument('--ocr-languages', type=str, default='ru,en',
                       help='OCR languages (comma-separated, default: ru,en)')
    parser.add_argument('--ocr-multiple-attempts', action='store_true',
                       help='Use multiple preprocessing attempts for better OCR')
    parser.add_argument('--ocr-both', action='store_true',
                       help='Show both Cyrillic and Latin versions')
    
    args = parser.parse_args()
    
    # Validate video mode arguments
    if args.mode == 'video' and not args.input:
        parser.error("--input is required for video mode")
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("License Plate Detection System Started")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Confidence threshold: {args.conf}")
    
    # Initialize model
    try:
        model = My_LicensePlate_Model(
            model_path=args.weights,
            device=args.device,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        sys.exit(1)
    
    # Initialize speed detector if enabled
    speed_detector = None
    if args.speed:
        speed_detector = SpeedDetector(
            fps=args.fps,
            pixels_per_meter=args.pixels_per_meter,
            speed_limit_kmh=args.speed_limit
        )
        logger.info(f"Speed detection enabled (limit: {args.speed_limit} km/h, calibration: {args.pixels_per_meter} px/m)")
    
    # Initialize OCR if enabled
    ocr = None
    if args.ocr:
        languages = args.ocr_languages.split(',')
        try:
            ocr = LicensePlateOCR(
                languages=languages,
                gpu=(args.device == 'cuda')
            )
            logger.info(f"OCR enabled with languages: {languages}")
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {e}")
            logger.warning("Continuing without OCR")
    
    # Run selected mode
    try:
        if args.mode == 'video':
            process_video_mode(
                args.input, args.output, model,
                speed_detector, ocr, logger
            )
        else:  # camera mode
            process_camera_mode(
                args.camera_id, model,
                speed_detector, ocr, logger
            )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("System shutdown")
    logger.info("=" * 60)


if __name__ == "__main__":
    cli()