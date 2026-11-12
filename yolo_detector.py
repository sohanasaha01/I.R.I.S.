import cv2
import numpy as np
from ultralytics import YOLO
from detector import ObjectDetector


class YOLODetector(ObjectDetector):

    def __init__(
        self,
        model_path=r"C:\border_project\best.pt"
    ):

        print("Loading custom YOLO11 model...")
        print("Model:", model_path)

        self.model = YOLO(
            model_path
        )

        print(
            "Custom YOLO11 model loaded successfully."
        )

        print(
            "Classes:",
            self.model.names
        )


    def detect(self, frame):

        # ----------------------------------------------------
        # IMPORTANT
        #
        # The preprocessing pipeline can produce float32
        # images in the 0-1 range.
        #
        # YOLO detection should receive a normal uint8
        # BGR image.
        # ----------------------------------------------------

        if frame is None:

            return []


        if frame.size == 0:

            return []


        # ----------------------------------------------------
        # CONVERT FLOAT IMAGE TO UINT8
        # ----------------------------------------------------

        if frame.dtype != "uint8":

            frame = np.clip(
                frame * 255.0,
                0,
                255
            ).astype(
                "uint8"
            )


        # ----------------------------------------------------
        # YOLO TRACKING
        # ----------------------------------------------------

        results = self.model.track(

            source=frame,

            persist=True,

            tracker="bytetrack.yaml",

            conf=0.25,

            iou=0.50,

            imgsz=640,

            verbose=False
        )


        detections = []


        # ----------------------------------------------------
        # READ RESULTS
        # ----------------------------------------------------

        for result in results:

            if result.boxes is None:

                continue


            for box in result.boxes:

                # --------------------------------------------
                # BOUNDING BOX
                # --------------------------------------------

                x1, y1, x2, y2 = (
                    box.xyxy[0].tolist()
                )


                # --------------------------------------------
                # CONFIDENCE
                # --------------------------------------------

                confidence = float(
                    box.conf[0]
                )


                # --------------------------------------------
                # CLASS
                # --------------------------------------------

                class_id = int(
                    box.cls[0]
                )


                class_name = (
                    self.model.names[
                        class_id
                    ]
                )


                # --------------------------------------------
                # TRACK ID
                # --------------------------------------------

                if box.id is not None:

                    track_id = int(
                        box.id[0]
                    )

                else:

                    track_id = None


                # --------------------------------------------
                # SAVE DETECTION
                # --------------------------------------------

                detections.append({

                    "id":
                        track_id,

                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2)
                    ),

                    "label":
                        class_name,

                    "confidence":
                        confidence
                })


        return detections