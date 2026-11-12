from pathlib import Path
import argparse
import cv2
from yolo_detector import YOLODetector

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Path to an image for detection")
args = parser.parse_args()

image_path = Path(args.image)

if not image_path.is_file():
    raise FileNotFoundError(f"Image not found: {image_path}")

img = cv2.imread(str(image_path))

if img is None:
    raise ValueError(f"Could not read image: {image_path}")

detector = YOLODetector()
detections = detector.detect(img)

print(detections)