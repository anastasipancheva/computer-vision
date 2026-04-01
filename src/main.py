import cv2
import sys
import os
from loguru import logger
from model_impl import My_LicensePlate_Model

if not os.path.exists('./data'):
    os.makedirs('./data')

logger.add("./data/log_file.log", rotation="10 MB", level="INFO")

def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python main.py <source>")
        return

    source = sys.argv[1]
    model = My_LicensePlate_Model("best.pt")
    
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    
    if not cap.isOpened():
        logger.error(f"Failed to open source: {source}")
        return

    logger.info(f"Starting detection on: {source}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.detect_plates(frame)
        
        for res in results:
            logger.info(f"Plate detected: {res['probability']:.2f}")
            box = res['bbox']
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        cv2.imshow("Result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Processing finished")

if __name__ == "__main__":
    main()