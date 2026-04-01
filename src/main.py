import cv2
import sys
from loguru import logger
from model_impl import My_LicensePlate_Model

logger.add("./data/log_file.log", rotation="10 MB")

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python main.py <path_to_video_or_camera_index>")
        return

    source = sys.argv[1]
    model = My_LicensePlate_Model("yolov8n.pt")
    
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.detect_plates(frame)
        
        for res in results:
            logger.info(f"Detected plate with probability: {res['probability']}")
            
        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()