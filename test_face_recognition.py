import cv2
from face_recognizer import FaceRecognizer


IMAGE_PATH = r"C:\border_project\faces\Deepika\Deepika1.jpeg"


print("========================================")
print("FACE RECOGNITION TEST")
print("========================================")
print()

recognizer = FaceRecognizer()

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Could not read image.")
    print(IMAGE_PATH)
    raise SystemExit(1)

print()
print("Testing image:")
print(IMAGE_PATH)
print()

name, score = recognizer.recognize(image)

print("========================================")
print("RESULT")
print("========================================")
print("Name:", name)
print(f"Similarity score: {score:.4f}")
print("========================================")