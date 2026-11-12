from frame_source import FileFrameSource, WebcamFrameSource, RTSPFrameSource
import cv2

def test_source(source, name):
    count = 0
    last_ts = 0
    while count < 100:
        frame, ts = source.get_frame()
        if frame is None:
            continue
        h, w, _ = frame.shape
        print(f"{name} | Frame {count} | Resolution: {w}x{h} | Timestamp: {ts}")
        assert ts >= last_ts, "Timestamps not monotonic!"
        last_ts = ts
        count += 1

if __name__ == "__main__":
    print("Testing File Source...")
    test_source(FileFrameSource("sample-5s.mp4"), "File")

    print("Testing Webcam Source...")
    test_source(WebcamFrameSource(0), "Webcam")

    print("Testing RTSP Source...")
    # Replace with a real RTSP URL if available
    test_source(RTSPFrameSource("rtsp://your_rtsp_stream_here"), "RTSP")
