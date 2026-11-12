import cv2
import time

class FrameSource:
    def get_frame(self):
        """Return (frame, timestamp). Must be implemented by subclasses."""
        raise NotImplementedError

class FileFrameSource(FrameSource):
    def __init__(self, filepath):
        self.cap = cv2.VideoCapture(filepath)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        ts = time.time()
        return frame, ts

class WebcamFrameSource(FrameSource):
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        ts = time.time()
        return frame, ts

class RTSPFrameSource(FrameSource):
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)

    def reconnect(self):
        self.cap.release()
        time.sleep(1)
        self.cap = cv2.VideoCapture(self.rtsp_url)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("RTSP dropped, reconnecting...")
            self.reconnect()
            return None, None
        ts = time.time()
        return frame, ts
