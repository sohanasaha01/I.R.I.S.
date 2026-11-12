import cv2
import time
from preprocessing import Preprocessor

# Load a test image (use your sample video to grab one frame)
img = cv2.imread("sample-5s.mp4")  # ⚠️ Replace with an actual image file for testing

pre = Preprocessor(target_size=(320, 320), use_clahe=True, fps_cap=10)

count = 0
start = time.time()

while count < 100:
    frame = cv2.imread("test.jpg")  # Replace with a real image file
    processed = pre.preprocess_frame(frame)
    if processed is None:
        continue
    print(f"Frame {count} | Shape: {processed.shape} | Dtype: {processed.dtype} | Min: {processed.min()} | Max: {processed.max()}")
    count += 1

end = time.time()
print(f"Processed {count} frames in {end-start:.2f} seconds")
