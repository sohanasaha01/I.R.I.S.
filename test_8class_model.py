from ultralytics import YOLO
import cv2

MODEL_PATH = r"C:\border_project\runs\border_custom_8class\weights\best.pt"
VIDEO_PATH = r"C:\clip folder\WhatsApp Video 2026-09-08 at 12.41.35 AM.mp4"
OUTPUT_PATH = r"C:\border_project\8class_fast_test.mp4"

print("Loading 8-class YOLO model...")
model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print("Classes:")
print(model.names)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video opened successfully.")
print("FPS:", fps)
print("Resolution:", width, "x", height)

# Output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps / 60,
    (width, height)
)

frame_number = 0
processed = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # Process only every 60th frame
    if frame_number % 60 != 0:
        continue

    processed += 1

    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    print(
        "Processed sample:",
        processed,
        "| Original frame:",
        frame_number
    )

cap.release()
out.release()

print()
print("FAST TEST COMPLETED")
print("Original frames:", frame_number)
print("Frames analyzed:", processed)
print("Output:")
print(OUTPUT_PATH)