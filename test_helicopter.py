import cv2
from ultralytics import YOLO

MODEL_PATH = r"C:\border_project\runs\border_custom-5\weights\best.pt"
VIDEO_PATH = r"C:\clip folder\WhatsApp Video 2026-09-08 at 12.43.47 AM.mp4"

model = YOLO(MODEL_PATH)

print("MODEL CLASSES:")
print(model.names)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: VIDEO COULD NOT BE OPENED")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print()
print("VIDEO OPENED SUCCESSFULLY")
print("FPS:", fps)
print("TOTAL FRAMES:", total_frames)

frame_number = 0
helicopter_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Analyze every 30th frame
    if frame_number % 30 != 0:
        continue

    results = model.predict(
        source=frame,
        conf=0.20,
        verbose=False
    )

    result = results[0]

    if result.boxes is None:
        continue

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = model.names[class_id]

        if label == "helicopter":
            helicopter_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            print(
                f"HELICOPTER DETECTED | "
                f"Frame: {frame_number} | "
                f"Confidence: {confidence:.3f} | "
                f"Box: ({x1},{y1},{x2},{y2})"
            )

print()
print("====================================")
print("HELICOPTER TEST COMPLETED")
print("Frames checked:", frame_number)
print("Helicopter detections:", helicopter_count)
print("====================================")

cap.release()