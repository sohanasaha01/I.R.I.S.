
import cv2
import time
import threading
import queue
import math
import numpy as np
import os
import json

from geofence import Geofence
from preprocessing import Preprocessor
from yolo_detector import YOLODetector
from face_recognizer import FaceRecognizer
from army_uniform_classifier import ArmyUniformClassifier
from backend import VideoSource

from edge.threat_engine.threat_engine import (
    ThreatEvidence,
    compute_threat
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"C:\border_project\best.pt"
ZONE_FOLDER = r"C:\border_project\zones"

QUEUE_SIZE = 10

DWELL_TIME_LIMIT = 30.0
SPEED_LIMIT = 5.0

FACE_RECOGNITION_INTERVAL = 10

ARMY_CLASSIFICATION_INTERVAL = 10
ARMY_CLASSIFICATION_THRESHOLD = 0.60

THREAT_ALERT_THRESHOLD = 50.0

LOITER_THRESHOLD_PX = 15.0
LOITER_MIN_SECONDS = 10.0


# ============================================================
# AUTHORIZED OFFICERS
# ============================================================

AUTHORIZED_OFFICERS = {
    "Sohana Saha",
    "Adrija Pramanik",
    "Deepika",
    "Kritika Gupta",
    "Nisha Yadav",
    "Shazia Parveen",
    "Sudeshna Saha",
    "Sumit Dey",
    "Swastika"
}


# ============================================================
# GLOBAL TRACKING / CACHE
# ============================================================

track_history = {}
face_cache = {}
uniform_cache = {}
identity_alert_state = {}
zone_state = {}
inside_start_time = {}

next_fallback_id = 100000


# ============================================================
# SOURCE NAME
# ============================================================

def get_source_name(source_type, source):

    if source_type == "video":

        source_clean = (
            str(source)
            .strip()
            .strip('"')
            .strip("'")
        )

        filename = os.path.basename(
            source_clean
        )

        return os.path.splitext(
            filename
        )[0]

    elif source_type == "webcam":

        return f"webcam_{source}"

    elif source_type == "cctv":

        return "cctv_1"

    return "unknown_source"


# ============================================================
# ZONE PATH
# ============================================================

def get_zone_path(
    source_type,
    source
):

    os.makedirs(
        ZONE_FOLDER,
        exist_ok=True
    )

    source_name = get_source_name(
        source_type,
        source
    )

    return os.path.join(
        ZONE_FOLDER,
        f"{source_name}.json"
    )


# ============================================================
# SAVE GEOFENCE
# ============================================================

def save_geofence(
    source_type,
    source,
    normalized_points
):

    zone_path = get_zone_path(
        source_type,
        source
    )

    data = {
        "source_type": source_type,
        "source": str(source),
        "points": normalized_points
    }

    with open(
        zone_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )

    print()
    print("========================================")
    print("GEOFENCE SAVED")
    print("========================================")
    print(zone_path)
    print()


# ============================================================
# SOURCE SELECTION
# ============================================================

def select_source():

    print()
    print("========================================")
    print("       BORDER SURVEILLANCE SYSTEM")
    print("========================================")
    print()

    print("Select source:")
    print()

    print("1. Video Clip")
    print("2. Webcam / Laptop Camera")
    print("3. CCTV / RTSP")

    print()

    choice = input(
        "Enter choice (1/2/3): "
    ).strip()

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    if choice == "1":

        source = input(
            "Enter video path: "
        ).strip().strip('"').strip("'")

        if not os.path.exists(source):

            print()
            print("ERROR: Video file not found.")
            print(source)
            print()

            return None, None

        return "video", source

    # --------------------------------------------------------
    # WEBCAM
    # --------------------------------------------------------

    if choice == "2":

        source = input(
            "Enter webcam index (usually 0): "
        ).strip()

        try:

            source = int(source)

        except ValueError:

            print()
            print(
                "ERROR: Webcam index must be a number."
            )
            print()

            return None, None

        return "webcam", source

    # --------------------------------------------------------
    # CCTV
    # --------------------------------------------------------

    if choice == "3":

        source = input(
            "Enter CCTV / RTSP URL: "
        ).strip().strip('"').strip("'")

        return "cctv", source

    print()
    print("Invalid choice.")
    print()

    return None, None


# ============================================================
# GEOFENCE SELECTION
# ============================================================

def select_geofence(frame):

    points = []

    display_frame = frame.copy()

    window_name = "Draw Geofence"

    print()
    print("========================================")
    print("          GEOFENCE SELECTION")
    print("========================================")
    print()

    print("LEFT CLICK = Add point")
    print("S = Save")
    print("R = Reset")
    print("Q = Cancel")

    print()

    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            points.append(
                (x, y)
            )

            cv2.circle(
                display_frame,
                (x, y),
                5,
                (0, 255, 255),
                -1
            )

            if len(points) > 1:

                cv2.line(
                    display_frame,
                    points[-2],
                    points[-1],
                    (0, 255, 255),
                    2
                )

    cv2.namedWindow(
        window_name
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )

    while True:

        cv2.imshow(
            window_name,
            display_frame
        )

        key = cv2.waitKey(1) & 0xFF

        # SAVE
        if key == ord("s"):

            if len(points) < 3:

                print(
                    "Need at least 3 points."
                )

                continue

            break

        # RESET
        if key == ord("r"):

            points.clear()

            display_frame = frame.copy()

            print(
                "Geofence reset."
            )

        # CANCEL
        if key == ord("q"):

            points.clear()

            break

    cv2.destroyWindow(
        window_name
    )

    if len(points) < 3:

        return None

    height, width = frame.shape[:2]

    normalized_points = []

    for x, y in points:

        normalized_points.append(
            (
                round(
                    x / width,
                    6
                ),
                round(
                    y / height,
                    6
                )
            )
        )

    return normalized_points


# ============================================================
# IOU
# ============================================================

def calculate_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(
        ax1,
        bx1
    )

    inter_y1 = max(
        ay1,
        by1
    )

    inter_x2 = min(
        ax2,
        bx2
    )

    inter_y2 = min(
        ay2,
        by2
    )

    inter_width = max(
        0,
        inter_x2 - inter_x1
    )

    inter_height = max(
        0,
        inter_y2 - inter_y1
    )

    intersection = (
        inter_width *
        inter_height
    )

    area_a = (
        max(
            0,
            ax2 - ax1
        )
        *
        max(
            0,
            ay2 - ay1
        )
    )

    area_b = (
        max(
            0,
            bx2 - bx1
        )
        *
        max(
            0,
            by2 - by1
        )
    )

    union = (
        area_a +
        area_b -
        intersection
    )

    if union <= 0:

        return 0.0

    return intersection / union


# ============================================================
# WEAPON / PERSON ASSOCIATION
# ============================================================

def weapon_belongs_to_person(
    weapon_bbox,
    person_bbox
):

    iou = calculate_iou(
        weapon_bbox,
        person_bbox
    )

    if iou > 0:

        return True

    wx1, wy1, wx2, wy2 = weapon_bbox

    px1, py1, px2, py2 = person_bbox

    weapon_cx = (
        wx1 + wx2
    ) / 2

    weapon_cy = (
        wy1 + wy2
    ) / 2

    person_cx = (
        px1 + px2
    ) / 2

    person_cy = (
        py1 + py2
    ) / 2

    distance = math.sqrt(
        (
            weapon_cx -
            person_cx
        ) ** 2
        +
        (
            weapon_cy -
            person_cy
        ) ** 2
    )

    person_width = (
        px2 - px1
    )

    person_height = (
        py2 - py1
    )

    person_diagonal = math.sqrt(
        person_width ** 2 +
        person_height ** 2
    )

    if person_diagonal <= 0:

        return False

    normalized_distance = (
        distance /
        person_diagonal
    )

    return normalized_distance < 1.25


# ============================================================
# FIND ASSOCIATED PERSON
# ============================================================

def find_associated_person(
    weapon_bbox,
    person_detections
):

    best_person = None

    best_distance = float("inf")

    wx1, wy1, wx2, wy2 = weapon_bbox

    weapon_cx = (
        wx1 + wx2
    ) / 2

    weapon_cy = (
        wy1 + wy2
    ) / 2

    for person in person_detections:

        person_bbox = person["bbox"]

        if not weapon_belongs_to_person(
            weapon_bbox,
            person_bbox
        ):

            continue

        px1, py1, px2, py2 = person_bbox

        person_cx = (
            px1 + px2
        ) / 2

        person_cy = (
            py1 + py2
        ) / 2

        distance = math.sqrt(
            (
                weapon_cx -
                person_cx
            ) ** 2
            +
            (
                weapon_cy -
                person_cy
            ) ** 2
        )

        if distance < best_distance:

            best_distance = distance

            best_person = person

    return best_person


# ============================================================
# LOITERING
# ============================================================

def calculate_loitering(
    history,
    timestamp
):

    trajectory = history.get(
        "trajectory",
        []
    )

    if len(trajectory) < 5:

        return False, 0.0

    recent_points = trajectory[-10:]

    current_x, current_y = (
        recent_points[-1]
    )

    max_distance = 0.0

    for x, y in recent_points:

        distance = math.sqrt(
            (
                current_x - x
            ) ** 2
            +
            (
                current_y - y
            ) ** 2
        )

        max_distance = max(
            max_distance,
            distance
        )

    if max_distance <= LOITER_THRESHOLD_PX:

        if history.get(
            "loiter_start_time"
        ) is None:

            history[
                "loiter_start_time"
            ] = timestamp

        loiter_seconds = (
            timestamp -
            history[
                "loiter_start_time"
            ]
        )

        return (
            loiter_seconds >=
            LOITER_MIN_SECONDS,
            loiter_seconds
        )

    history[
        "loiter_start_time"
    ] = None

    return False, 0.0


# ============================================================
# FALLBACK TRACK ID
# ============================================================

def assign_fallback_id():

    global next_fallback_id

    track_id = next_fallback_id

    next_fallback_id += 1

    return track_id


# ============================================================
# PRODUCER
# ============================================================

def producer(
    video_source,
    preprocessor,
    frame_queue,
    stop_event,
    producer_done
):

    frame_number = 0

    while not stop_event.is_set():

        ret, frame = (
            video_source.read()
        )

        if not ret:

            print()
            print(
                "[INFO] Source ended or frame could not be read."
            )
            print()

            break

        frame_number += 1

        processed = (
            preprocessor.preprocess_frame(
                frame
            )
        )

        if processed is None:

            continue

        timestamp = time.time()

        item = (
            processed,
            timestamp,
            frame_number
        )

        try:

            frame_queue.put_nowait(
                item
            )

        except queue.Full:

            try:

                frame_queue.get_nowait()

            except queue.Empty:

                pass

            try:

                frame_queue.put_nowait(
                    item
                )

            except queue.Full:

                pass

    producer_done.set()


# ============================================================
# CONSUMER
# ============================================================

def consumer(
    frame_queue,
    stop_event,
    producer_done,
    geofence
):

    print()
    print("========================================")
    print("          STARTING ANALYSIS")
    print("========================================")
    print()

    # ========================================================
    # YOLO
    # ========================================================

    print(
        "[INFO] Loading YOLO model:"
    )

    print(
        MODEL_PATH
    )

    detector = YOLODetector(
        MODEL_PATH
    )

    # ========================================================
    # FACE RECOGNITION
    # ========================================================

    print()
    print(
        "[INFO] Loading face recognition system..."
    )

    face_recognizer = FaceRecognizer()

    # ========================================================
    # ARMY UNIFORM
    # ========================================================

    print()
    print(
        "[INFO] Loading Army Uniform Classifier..."
    )

    army_classifier = (
        ArmyUniformClassifier(
            confidence_threshold=
            ARMY_CLASSIFICATION_THRESHOLD
        )
    )

    print()
    print(
        "[INFO] Analysis system ready."
    )
    print()

    # ========================================================
    # FRAME COUNTER FOR DEBUG
    # ========================================================

    last_detection_report = 0

    while True:

        # ----------------------------------------------------
        # GET FRAME
        # ----------------------------------------------------

        try:

            (
                processed,
                timestamp,
                frame_number
            ) = frame_queue.get(
                timeout=0.1
            )

        except queue.Empty:

            if (
                producer_done.is_set()
                and
                frame_queue.empty()
            ):

                break

            continue

        # ----------------------------------------------------
        # PREPARE FRAME
        #
        # preprocessing.py should return normal uint8 BGR.
        #
        # DO NOT multiply by 255 before YOLO.
        # ----------------------------------------------------

        frame = processed.copy()

        # ====================================================
        # YOLO DETECTION
        # ====================================================

        detections = detector.detect(
            frame
        )

        # ====================================================
        # FALLBACK IDS
        # ====================================================

        tracked_detections = []

        for detection in detections:

            if detection.get("id") is None:

                detection["id"] = (
                    assign_fallback_id()
                )

                print(
                    f"[TRACKER] Fallback ID assigned: "
                    f"{detection['id']} | "
                    f"{detection['label']}"
                )

            tracked_detections.append(
                detection
            )

        # ====================================================
        # PERIODIC DETECTION DEBUG
        # ====================================================

        if (
            frame_number -
            last_detection_report
            >= 30
        ):

            labels = [
                d["label"]
                for d in tracked_detections
            ]

            print(
                f"[DETECTION] "
                f"Frame: {frame_number} | "
                f"Objects: {len(tracked_detections)} | "
                f"Labels: {labels}"
            )

            last_detection_report = (
                frame_number
            )

        # ====================================================
        # DISPLAY FRAME
        #
        # IMPORTANT:
        # OpenCV drawing functions require CV_8U / uint8.
        # ====================================================

        display_frame = processed.copy()

        if display_frame.dtype != np.uint8:

            if display_frame.max() <= 1.0:

                display_frame = (
                    display_frame * 255.0
                ).clip(
                    0,
                    255
                ).astype(
                    np.uint8
                )

            else:

                display_frame = (
                    display_frame
                ).clip(
                    0,
                    255
                ).astype(
                    np.uint8
                )

        # ====================================================
        # DRAW GEOFENCE
        # ====================================================

        height, width = frame.shape[:2]

        if geofence is not None:

            polygon_points = (
                geofence.to_pixel_points(
                    width,
                    height
                )
            )

            polygon_array = np.array(
                polygon_points,
                dtype=np.int32
            )

            cv2.polylines(
                display_frame,
                [polygon_array],
                True,
                (0, 255, 255),
                2
            )

        # ====================================================
        # SEPARATE PERSONS / WEAPONS
        # ====================================================

        person_detections = []

        weapon_detections = []

        for detection in tracked_detections:

            if detection["label"] == "person":

                person_detections.append(
                    detection
                )

            elif detection["label"] in [
                "gun",
                "bomb",
                "knife"
            ]:

                weapon_detections.append(
                    detection
                )

        # ====================================================
        # PROCESS EACH OBJECT
        # ====================================================

        for detection in tracked_detections:

            track_id = detection["id"]

            bbox = detection["bbox"]

            label = detection["label"]

            confidence = detection["confidence"]

            x1, y1, x2, y2 = bbox

            # ------------------------------------------------
            # CLAMP BOX
            # ------------------------------------------------

            x1 = max(
                0,
                min(
                    int(x1),
                    width - 1
                )
            )

            y1 = max(
                0,
                min(
                    int(y1),
                    height - 1
                )
            )

            x2 = max(
                0,
                min(
                    int(x2),
                    width - 1
                )
            )

            y2 = max(
                0,
                min(
                    int(y2),
                    height - 1
                )
            )

            if x2 <= x1 or y2 <= y1:

                continue

            # =================================================
            # GEOFENCE
            # =================================================

            inside = False

            if geofence is not None:

                inside = geofence.is_inside(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    width,
                    height
                )

            # =================================================
            # CENTER
            # =================================================

            center_x = (
                x1 + x2
            ) / 2

            center_y = (
                y1 + y2
            ) / 2

            # =================================================
            # CREATE TRACK HISTORY
            # =================================================

            if track_id not in track_history:

                track_history[track_id] = {

                    "first_seen":
                        timestamp,

                    "last_seen":
                        timestamp,

                    "last_center":
                        (
                            center_x,
                            center_y
                        ),

                    "inside":
                        inside,

                    "entered_time":
                        (
                            timestamp
                            if inside
                            else None
                        ),

                    "trajectory":
                        [],

                    "previous_speed":
                        0.0,

                    "current_speed":
                        0.0,

                    "loiter_start_time":
                        None
                }

            history = (
                track_history[
                    track_id
                ]
            )

            # =================================================
            # SPEED
            # =================================================

            old_time = history[
                "last_seen"
            ]

            old_center = history[
                "last_center"
            ]

            previous_speed = history.get(
                "current_speed",
                0.0
            )

            dt = (
                timestamp -
                old_time
            )

            if dt > 0:

                distance = math.sqrt(
                    (
                        center_x -
                        old_center[0]
                    ) ** 2
                    +
                    (
                        center_y -
                        old_center[1]
                    ) ** 2
                )

                speed = (
                    distance /
                    dt
                )

            else:

                speed = 0.0

            history[
                "previous_speed"
            ] = previous_speed

            history[
                "current_speed"
            ] = speed

            history[
                "last_seen"
            ] = timestamp

            history[
                "last_center"
            ] = (
                center_x,
                center_y
            )

            # =================================================
            # TRAJECTORY
            # =================================================

            history[
                "trajectory"
            ].append(
                (
                    int(center_x),
                    int(center_y)
                )
            )

            if len(
                history["trajectory"]
            ) > 50:

                history[
                    "trajectory"
                ] = history[
                    "trajectory"
                ][-50:]

            # =================================================
            # GEOFENCE STATE
            # =================================================

            previous_inside = (
                zone_state.get(
                    track_id,
                    history.get(
                        "inside",
                        inside
                    )
                )
            )

            if track_id not in zone_state:

                zone_state[
                    track_id
                ] = inside

                if inside:

                    inside_start_time[
                        track_id
                    ] = timestamp

                    print(
                        f"[GEOFENCE] ENTER | "
                        f"Track ID: {track_id}"
                    )

            elif (
                not previous_inside
                and
                inside
            ):

                zone_state[
                    track_id
                ] = True

                inside_start_time[
                    track_id
                ] = timestamp

                print(
                    f"[GEOFENCE] ENTER | "
                    f"Track ID: {track_id}"
                )

            elif (
                previous_inside
                and
                not inside
            ):

                zone_state[
                    track_id
                ] = False

                inside_start_time.pop(
                    track_id,
                    None
                )

                print(
                    f"[GEOFENCE] EXIT | "
                    f"Track ID: {track_id}"
                )

            else:

                zone_state[
                    track_id
                ] = inside

            history[
                "inside"
            ] = inside

            # =================================================
            # DWELL
            # =================================================

            dwell_time = 0.0

            if inside:

                if history[
                    "entered_time"
                ] is None:

                    history[
                        "entered_time"
                    ] = timestamp

                dwell_time = (
                    timestamp -
                    history[
                        "entered_time"
                    ]
                )

            else:

                history[
                    "entered_time"
                ] = None

            # =================================================
            # INSIDE TIMER
            # =================================================

            inside_duration = 0.0

            if inside:

                if (
                    inside_start_time
                    .get(track_id)
                    is not None
                ):

                    inside_duration = (
                        timestamp -
                        inside_start_time[
                            track_id
                        ]
                    )

            # =================================================
            # LOITERING
            # =================================================

            (
                loitering,
                loiter_seconds
            ) = calculate_loitering(
                history,
                timestamp
            )

            # =================================================
            # DEFAULT VALUES
            # =================================================

            display_name = "Unknown"

            face_score = 0.0

            uniform_label = "unknown"

            uniform_score = 0.0

            threat_score = 0.0

            threat_severity = "UNKNOWN"

            associated_person = None

            associated_person_name = "Unknown"

            # =================================================
            # PERSON PROCESSING
            # =================================================

            if label == "person":

                print(
                    f"[PERSON DETECTED] "
                    f"Track ID: {track_id} | "
                    f"Confidence: {confidence:.3f}"
                )

                # =============================================
                # FACE RECOGNITION
                # =============================================

                should_recognize = (
                    track_id not in face_cache
                    or
                    frame_number %
                    FACE_RECOGNITION_INTERVAL
                    == 0
                )

                if should_recognize:

                    crop = frame[
                        y1:y2,
                        x1:x2
                    ]

                    if crop.size > 0:

                        print(
                            f"[FACE DEBUG] "
                            f"Running face recognition "
                            f"for Track ID: {track_id}"
                        )

                        try:

                            (
                                name,
                                score
                            ) = (
                                face_recognizer.recognize(
                                    crop
                                )
                            )

                            print(
                                f"[FACE RESULT] "
                                f"Track ID: {track_id} | "
                                f"Name: {name} | "
                                f"Score: {score:.3f}"
                            )

                            # ---------------------------------
                            # Do not overwrite a known person
                            # with Unknown.
                            # ---------------------------------

                            if name != "Unknown":

                                face_cache[
                                    track_id
                                ] = {

                                    "name":
                                        name,

                                    "score":
                                        score
                                }

                            elif (
                                track_id
                                not in face_cache
                            ):

                                face_cache[
                                    track_id
                                ] = {

                                    "name":
                                        "Unknown",

                                    "score":
                                        score
                                }

                        except Exception as error:

                            print(
                                f"[FACE ERROR] "
                                f"Track ID: {track_id} | "
                                f"{error}"
                            )

                # =============================================
                # GET FACE CACHE
                # =============================================

                if track_id in face_cache:

                    display_name = (
                        face_cache[
                            track_id
                        ]["name"]
                    )

                    face_score = (
                        face_cache[
                            track_id
                        ]["score"]
                    )

                # =============================================
                # ARMY UNIFORM
                # =============================================

                should_classify_uniform = (
                    track_id not in uniform_cache
                    or
                    frame_number %
                    ARMY_CLASSIFICATION_INTERVAL
                    == 0
                )

                if should_classify_uniform:

                    crop = frame[
                        y1:y2,
                        x1:x2
                    ]

                    if crop.size > 0:

                        try:

                            (
                                uniform_result,
                                uniform_confidence
                            ) = (
                                army_classifier.classify(
                                    crop
                                )
                            )

                            uniform_cache[
                                track_id
                            ] = {

                                "label":
                                    uniform_result,

                                "score":
                                    uniform_confidence
                            }

                        except Exception as error:

                            print(
                                f"[UNIFORM ERROR] "
                                f"Track ID: {track_id} | "
                                f"{error}"
                            )

                # =============================================
                # GET UNIFORM CACHE
                # =============================================

                if track_id in uniform_cache:

                    uniform_label = (
                        uniform_cache[
                            track_id
                        ]["label"]
                    )

                    uniform_score = (
                        uniform_cache[
                            track_id
                        ]["score"]
                    )

                print(
                    f"[UNIFORM] "
                    f"Track ID: {track_id} | "
                    f"{uniform_label} | "
                    f"Confidence: "
                    f"{uniform_score:.3f}"
                )

                # =============================================
                # IDENTITY ALERT
                # =============================================

                previous_identity = (
                    identity_alert_state.get(
                        track_id
                    )
                )

                if display_name == "Unknown":

                    if (
                        previous_identity
                        !=
                        "Unknown"
                    ):

                        print(
                            f"[ALERT] "
                            f"UNKNOWN PERSON DETECTED | "
                            f"Track ID: {track_id} | "
                            f"Face Score: "
                            f"{face_score:.3f}"
                        )

                    identity_alert_state[
                        track_id
                    ] = "Unknown"

                else:

                    if (
                        previous_identity
                        !=
                        display_name
                    ):

                        print(
                            f"[IDENTIFIED] "
                            f"{display_name} | "
                            f"Track ID: {track_id} | "
                            f"Face Score: "
                            f"{face_score:.3f}"
                        )

                    identity_alert_state[
                        track_id
                    ] = display_name

            # =================================================
            # WEAPON ASSOCIATION
            # =================================================

            if label in [
                "gun",
                "bomb",
                "knife"
            ]:

                associated_person = (
                    find_associated_person(
                        bbox,
                        person_detections
                    )
                )

                if associated_person is not None:

                    person_id = (
                        associated_person["id"]
                    )

                    person_info = (
                        face_cache.get(
                            person_id,
                            {
                                "name":
                                    "Unknown",

                                "score":
                                    0.0
                            }
                        )
                    )

                    associated_person_name = (
                        person_info["name"]
                    )

                    if (
                        associated_person_name
                        in
                        AUTHORIZED_OFFICERS
                    ):

                        print(
                            f"[AUTHORIZED] "
                            f"{associated_person_name} "
                            f"carrying {label}. "
                            f"No alert."
                        )

                    else:

                        print(
                            f"[ALERT] "
                            f"{label.upper()} detected "
                            f"near unknown/"
                            f"unauthorized person | "
                            f"Person ID: "
                            f"{person_id}"
                        )

                else:

                    print(
                        f"[ALERT] "
                        f"{label.upper()} DETECTED | "
                        f"No associated person"
                    )

            # =================================================
            # DWELL ALERT
            # =================================================

            if (
                inside
                and
                dwell_time >
                DWELL_TIME_LIMIT
            ):

                print(
                    f"[ALERT] "
                    f"EXCESSIVE DWELL | "
                    f"Track ID: {track_id} | "
                    f"{dwell_time:.1f} seconds"
                )

            # =================================================
            # SPEED ALERT
            # =================================================

            if speed > SPEED_LIMIT:

                print(
                    f"[ALERT] "
                    f"HIGH SPEED | "
                    f"Track ID: {track_id} | "
                    f"Speed: {speed:.2f}"
                )

            # =================================================
            # THREAT ENGINE
            # =================================================

            threat_person_name = "Unknown"

            if label == "person":

                threat_person_name = (
                    display_name
                )

            elif associated_person is not None:

                threat_person_name = (
                    associated_person_name
                )

            threat_evidence = ThreatEvidence(

                track_id=track_id,

                object_type=label,

                person_name=(
                    threat_person_name
                ),

                inside_zone=inside,

                dwell_seconds=dwell_time,

                zone_name="Restricted Zone",

                tripwire_crossings=0,

                timestamp=None,

                is_night=None,

                speed_px_s=speed,

                previous_speed_px_s=(
                    previous_speed
                ),

                trajectory=(
                    history["trajectory"]
                ),

                loitering=loitering,

                loiter_threshold_px=(
                    LOITER_THRESHOLD_PX
                ),

                loiter_seconds=(
                    loiter_seconds
                )
            )

            try:

                threat_result = (
                    compute_threat(
                        threat_evidence
                    )
                )

                threat_score = (
                    threat_result.score
                )

                threat_severity = (
                    threat_result.severity
                )

                if (
                    threat_score
                    >=
                    THREAT_ALERT_THRESHOLD
                ):

                    print()

                    print(
                        f"[THREAT ALERT] "
                        f"Track ID: {track_id} | "
                        f"Score: "
                        f"{threat_score:.1f}/100 | "
                        f"Severity: "
                        f"{threat_severity}"
                    )

                    print(
                        f"[THREAT REASON] "
                        f"{threat_result.reason}"
                    )

                    print()

            except Exception as error:

                print(
                    f"[THREAT ENGINE ERROR] "
                    f"{error}"
                )

                threat_score = 0.0

                threat_severity = "UNKNOWN"

            # =================================================
            # DRAW TRAJECTORY
            # =================================================

            trajectory = history[
                "trajectory"
            ]

            if len(trajectory) > 1:

                for i in range(
                    1,
                    len(trajectory)
                ):

                    cv2.line(
                        display_frame,
                        trajectory[i - 1],
                        trajectory[i],
                        (255, 0, 0),
                        2
                    )

            # =================================================
            # BOX COLOR
            # =================================================

            if label == "person":

                if display_name == "Unknown":

                    box_color = (
                        0,
                        0,
                        255
                    )

                    display_text = (
                        "UNKNOWN PERSON | "
                        f"ID: {track_id}"
                    )

                else:

                    box_color = (
                        0,
                        255,
                        0
                    )

                    display_text = (
                        f"{display_name} | "
                        f"ID: {track_id}"
                    )

                if uniform_label == "army":

                    display_text += (
                        " | ARMY"
                    )

                elif uniform_label == "non_army":

                    display_text += (
                        " | NON-ARMY"
                    )

            elif label in [
                "gun",
                "bomb",
                "knife"
            ]:

                box_color = (
                    0,
                    0,
                    255
                )

                display_text = (
                    f"{label.upper()} | "
                    f"ID: {track_id}"
                )

            elif label in [
                "tank",
                "helicopter"
            ]:

                box_color = (
                    255,
                    0,
                    255
                )

                display_text = (
                    f"{label.upper()} | "
                    f"ID: {track_id}"
                )

            else:

                box_color = (
                    255,
                    255,
                    0
                )

                display_text = (
                    f"{label} | "
                    f"ID: {track_id}"
                )

            # =================================================
            # DRAW BOX
            # =================================================

            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            # =================================================
            # OBJECT LABEL
            # =================================================

            cv2.putText(
                display_frame,
                display_text,
                (
                    x1,
                    max(
                        20,
                        y1 - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )

            # =================================================
            # CONFIDENCE
            # =================================================

            confidence_text = (
                f"Conf: {confidence:.2f}"
            )

            cv2.putText(
                display_frame,
                confidence_text,
                (
                    x1,
                    min(
                        height - 10,
                        y2 + 20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1
            )

            # =================================================
            # THREAT
            # =================================================

            threat_text = (
                f"Threat: "
                f"{threat_severity}"
            )

            cv2.putText(
                display_frame,
                threat_text,
                (
                    x1,
                    min(
                        height - 30,
                        y2 + 40
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1
            )

            # =================================================
            # GEOFENCE STATUS
            # =================================================

            status_text = (
                "INSIDE ZONE"
                if inside
                else
                "OUTSIDE ZONE"
            )

            cv2.putText(
                display_frame,
                status_text,
                (
                    x1,
                    min(
                        height - 50,
                        y2 + 60
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1
            )

            # =================================================
            # INSIDE TIMER
            # =================================================

            if inside:

                minutes = int(
                    inside_duration // 60
                )

                seconds = int(
                    inside_duration % 60
                )

                timer_text = (
                    f"INSIDE FOR: "
                    f"{minutes:02d}:"
                    f"{seconds:02d}"
                )

                cv2.putText(
                    display_frame,
                    timer_text,
                    (
                        x1,
                        min(
                            height - 10,
                            y2 + 80
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    box_color,
                    1
                )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "Border Surveillance",
            display_frame
        )

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )

        if key == ord("q"):

            print()

            print(
                "[INFO] Stopping analysis..."
            )

            stop_event.set()

            break

    print()

    print(
        "[INFO] Analysis consumer stopped."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global next_fallback_id

    # --------------------------------------------------------
    # RESET ALL STATE
    # --------------------------------------------------------

    track_history.clear()

    face_cache.clear()

    uniform_cache.clear()

    identity_alert_state.clear()

    zone_state.clear()

    inside_start_time.clear()

    next_fallback_id = 100000

    # --------------------------------------------------------
    # SOURCE
    # --------------------------------------------------------

    source_type, source = (
        select_source()
    )

    if source_type is None:

        return

    # --------------------------------------------------------
    # CONNECT SOURCE
    # --------------------------------------------------------

    print()

    print(
        "[INFO] Connecting to source..."
    )

    video_source = VideoSource(
        source_type,
        source
    )

    if not video_source.connect():

        print()

        print(
            "[ERROR] Could not connect to source."
        )

        print()

        return

    print(
        "[INFO] Source connected successfully."
    )

    # --------------------------------------------------------
    # FIRST FRAME
    # --------------------------------------------------------

    ret, first_frame = (
        video_source.read()
    )

    if not ret:

        print()

        print(
            "[ERROR] Could not read first frame."
        )

        print()

        video_source.release()

        return

    print()

    print(
        "Original frame size:",
        first_frame.shape
    )

    print()

    # --------------------------------------------------------
    # PREPROCESSOR
    #
    # 640x640
    # CLAHE disabled
    # uint8 output
    # --------------------------------------------------------

    preprocessor = Preprocessor(
        target_size=(640, 640),
        use_clahe=False,
        fps_cap=None
    )

    # --------------------------------------------------------
    # PROCESS FIRST FRAME
    # --------------------------------------------------------

    first_processed = (
        preprocessor.preprocess_frame(
            first_frame
        )
    )

    if first_processed is None:

        print(
            "[ERROR] First frame preprocessing failed."
        )

        video_source.release()

        return

    # --------------------------------------------------------
    # PREPARE FIRST DISPLAY FRAME
    # --------------------------------------------------------

    first_display = (
        first_processed.copy()
    )

    if first_display.dtype != np.uint8:

        if first_display.max() <= 1.0:

            first_display = (
                first_display * 255.0
            ).clip(
                0,
                255
            ).astype(
                np.uint8
            )

        else:

            first_display = (
                first_display
            ).clip(
                0,
                255
            ).astype(
                np.uint8
            )

    # --------------------------------------------------------
    # NEW GEOFENCE
    # --------------------------------------------------------

    print()

    print("========================================")

    print(
        "          NEW GEOFENCE REQUIRED"
    )

    print("========================================")

    print()

    print(
        "Select the restricted area."
    )

    print(
        "The saved zone for this source "
        "will be replaced."
    )

    normalized_points = (
        select_geofence(
            first_display
        )
    )

    if normalized_points is None:

        print()

        print(
            "[INFO] Geofence selection cancelled."
        )

        print()

        video_source.release()

        cv2.destroyAllWindows()

        return

    # --------------------------------------------------------
    # SAVE GEOFENCE
    # --------------------------------------------------------

    save_geofence(
        source_type,
        source,
        normalized_points
    )

    geofence = Geofence(
        normalized_points
    )

    print(
        "[INFO] New geofence selected successfully."
    )

    # --------------------------------------------------------
    # QUEUE
    # --------------------------------------------------------

    frame_queue = queue.Queue(
        maxsize=QUEUE_SIZE
    )

    stop_event = threading.Event()

    producer_done = threading.Event()

    # --------------------------------------------------------
    # PRODUCER THREAD
    # --------------------------------------------------------

    producer_thread = threading.Thread(
        target=producer,
        args=(
            video_source,
            preprocessor,
            frame_queue,
            stop_event,
            producer_done
        ),
        daemon=True
    )

    # --------------------------------------------------------
    # CONSUMER THREAD
    # --------------------------------------------------------

    consumer_thread = threading.Thread(
        target=consumer,
        args=(
            frame_queue,
            stop_event,
            producer_done,
            geofence
        ),
        daemon=True
    )

    print()

    print(
        "[INFO] Starting producer..."
    )

    producer_thread.start()

    print(
        "[INFO] Starting analysis..."
    )

    consumer_thread.start()

    # --------------------------------------------------------
    # WAIT
    # --------------------------------------------------------

    try:

        consumer_thread.join()

    except KeyboardInterrupt:

        print()

        print(
            "[INFO] Keyboard interrupt received."
        )

        stop_event.set()

    # --------------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------------

    stop_event.set()

    producer_thread.join(
        timeout=2
    )

    consumer_thread.join(
        timeout=2
    )

    video_source.release()

    cv2.destroyAllWindows()

    print()

    print("========================================")

    print(
        "       SYSTEM SHUTDOWN COMPLETE"
    )

    print("========================================")

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
