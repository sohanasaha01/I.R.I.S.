import cv2
from detector import MockDetector

detector = MockDetector()
img = cv2.imread("test.jpg")  # any image
detections = detector.detect(img)
print(detections)
