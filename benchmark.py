import time
import cv2
from yolo_detector import YOLODetector

def benchmark(video_path="sample-clip.mp4"):
    detector = YOLODetector("C:/border_project/yolo11.pt")
    cap = cv2.VideoCapture(video_path)

    timings = {stage: [] for stage in ["ingestion","preprocessing","queue","detection","tracking"]}
    frame_count = 0
    start_total = time.perf_counter()

    while cap.isOpened():
        t_ingest = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        t_ingest_end = time.perf_counter()
        timings["ingestion"].append((t_ingest_end - t_ingest) * 1000)

        # Preprocessing
        t_pre = time.perf_counter()
        resized = cv2.resize(frame, (640, 480))
        t_pre_end = time.perf_counter()
        timings["preprocessing"].append((t_pre_end - t_pre) * 1000)

        # Queue simulation
        t_queue = time.perf_counter()
        # (simulate queue push/pop)
        t_queue_end = time.perf_counter()
        timings["queue"].append((t_queue_end - t_queue) * 1000)

        # Detection
        t_det = time.perf_counter()
        detections = detector.detect(resized)
        t_det_end = time.perf_counter()
        timings["detection"].append((t_det_end - t_det) * 1000)

        # Tracking
        t_track = time.perf_counter()
        tracks = detector.track(resized)
        t_track_end = time.perf_counter()
        timings["tracking"].append((t_track_end - t_track) * 1000)

        frame_count += 1

    end_total = time.perf_counter()
    total_time = end_total - start_total
    fps = frame_count / total_time if total_time > 0 else 0

    print("\n--- Benchmark Results ---")
    for stage, vals in timings.items():
        avg = sum(vals) / len(vals) if vals else 0
        print(f"{stage.capitalize()}: {avg:.2f} ms")
    print(f"End-to-end FPS: {fps:.2f}")

    cap.release()

if __name__ == "__main__":
    benchmark("C:/border_project/test_walk.mp4")
