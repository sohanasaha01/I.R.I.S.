from abc import ABC, abstractmethod

class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, frame):
        """Return list of detections: [ { 'bbox': (x1,y1,x2,y2), 'label': str, 'confidence': float } ]"""
        pass


import random
from detector import ObjectDetector

class MockDetector(ObjectDetector):
    def detect(self, frame):
        # Always return one fake box
        h, w, _ = frame.shape
        return [{
            "bbox": (int(w*0.3), int(h*0.3), int(w*0.6), int(h*0.6)),
            "label": "person",
            "confidence": round(random.uniform(0.5, 0.9), 2)
        }]
