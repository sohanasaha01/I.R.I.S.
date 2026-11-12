import cv2
import numpy as np
import time

class Preprocessor:
    def __init__(self, target_size=(640, 640), use_clahe=False, fps_cap=None):
        """
        target_size: (width, height) to resize frames
        use_clahe: True/False to enable low-light correction
        fps_cap: maximum frames per second (e.g., 10). None = no throttling
        """
        self.target_size = target_size
        self.use_clahe = use_clahe
        self.fps_cap = fps_cap
        self.last_time = 0

    def preprocess_frame(self, frame):
        # --- Resize ---
        frame = cv2.resize(frame, self.target_size)

        # --- Normalize (0-1 float) ---
        frame = frame.astype(np.float32) / 255.0

        # --- Optional CLAHE (low-light correction) ---
        if self.use_clahe:
            # Convert to LAB color space
            lab = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0

        # --- Frame-rate throttling ---
        if self.fps_cap:
            now = time.time()
            min_interval = 1.0 / self.fps_cap
            if now - self.last_time < min_interval:
                return None  # Skip frame
            self.last_time = now

        return frame
