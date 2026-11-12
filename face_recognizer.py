import cv2
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

DETECTOR_MODEL = (
    MODEL_DIR / "face_detection_yunet_2023mar.onnx"
)

RECOGNITION_MODEL = (
    MODEL_DIR / "face_recognition_sface_2021dec.onnx"
)

FACE_DIR = BASE_DIR / "faces"


# ============================================================
# FACE RECOGNIZER
# ============================================================

class FaceRecognizer:

    def __init__(self):

        # ----------------------------------------------------
        # CHECK MODEL FILES
        # ----------------------------------------------------

        print()
        print("========================================")
        print("LOADING FACE RECOGNITION SYSTEM")
        print("========================================")
        print()

        print(
            "YuNet model:",
            DETECTOR_MODEL
        )

        print(
            "SFace model:",
            RECOGNITION_MODEL
        )

        print(
            "Face database:",
            FACE_DIR
        )

        print()

        if not DETECTOR_MODEL.exists():

            raise FileNotFoundError(
                f"YuNet model not found:\n"
                f"{DETECTOR_MODEL}"
            )

        if not RECOGNITION_MODEL.exists():

            raise FileNotFoundError(
                f"SFace model not found:\n"
                f"{RECOGNITION_MODEL}"
            )

        # ----------------------------------------------------
        # LOAD YUNET
        # ----------------------------------------------------

        print(
            "Loading YuNet face detector..."
        )

        self.detector = cv2.FaceDetectorYN.create(
            str(DETECTOR_MODEL),
            "",
            (320, 320),
            0.8,
            0.3,
            5000
        )

        print(
            "YuNet loaded successfully."
        )

        # ----------------------------------------------------
        # LOAD SFACE
        # ----------------------------------------------------

        print(
            "Loading SFace face recognition model..."
        )

        self.recognizer = cv2.FaceRecognizerSF.create(
            str(RECOGNITION_MODEL),
            ""
        )

        print(
            "SFace loaded successfully."
        )

        # ----------------------------------------------------
        # KNOWN FACES
        # ----------------------------------------------------

        self.known_faces = {}

        self.load_known_faces()

        print()
        print(
            "Face recognition system ready."
        )
        print()


    # ========================================================
    # LOAD KNOWN FACES
    # ========================================================

    def load_known_faces(self):

        if not FACE_DIR.exists():

            print()
            print(
                "WARNING: Face database folder not found:"
            )

            print(
                FACE_DIR
            )

            return

        print()
        print(
            "========================================"
        )

        print(
            "LOADING KNOWN FACES"
        )

        print(
            "========================================"
        )

        print()

        total_embeddings = 0

        # ----------------------------------------------------
        # LOOP THROUGH PERSON FOLDERS
        # ----------------------------------------------------

        for person_folder in sorted(
            FACE_DIR.iterdir()
        ):

            if not person_folder.is_dir():

                continue

            person_name = (
                person_folder.name
            )

            print(
                f"Person: {person_name}"
            )

            embeddings = []

            # ------------------------------------------------
            # LOOP THROUGH IMAGES
            # ------------------------------------------------

            for image_path in sorted(
                person_folder.iterdir()
            ):

                if image_path.suffix.lower() not in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]:

                    continue

                image = cv2.imread(
                    str(image_path)
                )

                if image is None:

                    print(
                        f"  ERROR reading: "
                        f"{image_path.name}"
                    )

                    continue

                # --------------------------------------------
                # CREATE EMBEDDING
                # --------------------------------------------

                embedding = (
                    self.get_embedding(
                        image
                    )
                )

                if embedding is not None:

                    embeddings.append(
                        embedding
                    )

                    print(
                        f"  Loaded: "
                        f"{image_path.name}"
                    )

                else:

                    print(
                        f"  WARNING: "
                        f"No face detected in "
                        f"{image_path.name}"
                    )

            # ------------------------------------------------
            # SAVE PERSON EMBEDDINGS
            # ------------------------------------------------

            if embeddings:

                self.known_faces[
                    person_name
                ] = embeddings

                total_embeddings += (
                    len(embeddings)
                )

                print(
                    f"  {len(embeddings)} "
                    f"face(s) loaded for "
                    f"{person_name}"
                )

            else:

                print(
                    f"  WARNING: No usable "
                    f"faces found for "
                    f"{person_name}"
                )

            print()

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        print(
            "========================================"
        )

        print(
            "FACE DATABASE SUMMARY"
        )

        print(
            "========================================"
        )

        print()

        print(
            "Known people:",
            list(
                self.known_faces.keys()
            )
        )

        print(
            "Total people:",
            len(
                self.known_faces
            )
        )

        print(
            "Total embeddings:",
            total_embeddings
        )

        print()


    # ========================================================
    # DETECT FACE
    # ========================================================

    def detect_face(self, image):

        if image is None:

            return None

        if image.size == 0:

            return None

        # ----------------------------------------------------
        # IMAGE SIZE
        # ----------------------------------------------------

        height, width = (
            image.shape[:2]
        )

        if width <= 0 or height <= 0:

            return None

        # ----------------------------------------------------
        # SET YUNET INPUT SIZE
        # ----------------------------------------------------

        self.detector.setInputSize(
            (
                width,
                height
            )
        )

        # ----------------------------------------------------
        # DETECT
        # ----------------------------------------------------

        try:

            _, faces = (
                self.detector.detect(
                    image
                )
            )

        except Exception as e:

            print(
                "[FACE ERROR] "
                f"YuNet detection failed: {e}"
            )

            return None

        # ----------------------------------------------------
        # NO FACES
        # ----------------------------------------------------

        if faces is None:

            return None

        if len(faces) == 0:

            return None

        # ----------------------------------------------------
        # SELECT LARGEST FACE
        # ----------------------------------------------------

        largest_face = max(
            faces,
            key=lambda face:
                face[2] * face[3]
        )

        return largest_face


    # ========================================================
    # CREATE FACE EMBEDDING
    # ========================================================

    def get_embedding(self, image):

        face = (
            self.detect_face(
                image
            )
        )

        if face is None:

            return None

        # ----------------------------------------------------
        # ALIGN FACE
        # ----------------------------------------------------

        try:

            aligned_face = (
                self.recognizer.alignCrop(
                    image,
                    face
                )
            )

        except Exception as e:

            print(
                "[FACE ERROR] "
                f"Face alignment failed: {e}"
            )

            return None

        if aligned_face is None:

            return None

        # ----------------------------------------------------
        # CREATE SFACE FEATURE
        # ----------------------------------------------------

        try:

            feature = (
                self.recognizer.feature(
                    aligned_face
                )
            )

        except Exception as e:

            print(
                "[FACE ERROR] "
                f"SFace feature extraction failed: {e}"
            )

            return None

        return feature


    # ========================================================
    # RECOGNIZE FACE
    # ========================================================

    def recognize(self, image):

        # ----------------------------------------------------
        # DETECT / EMBED FACE
        # ----------------------------------------------------

        embedding = (
            self.get_embedding(
                image
            )
        )

        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        if embedding is None:

            print(
                "[FACE DEBUG] "
                "No face detected in person crop."
            )

            return (
                "Unknown",
                0.0
            )

        # ----------------------------------------------------
        # DATABASE EMPTY
        # ----------------------------------------------------

        if not self.known_faces:

            print(
                "[FACE DEBUG] "
                "Known face database is empty."
            )

            return (
                "Unknown",
                0.0
            )

        # ----------------------------------------------------
        # BEST MATCH
        # ----------------------------------------------------

        best_name = "Unknown"

        best_score = -1.0

        # ----------------------------------------------------
        # COMPARE AGAINST EVERY STORED EMBEDDING
        # ----------------------------------------------------

        for person_name, known_embeddings in (
            self.known_faces.items()
        ):

            person_best_score = -1.0

            for known_embedding in (
                known_embeddings
            ):

                try:

                    score = (
                        self.recognizer.match(
                            embedding,
                            known_embedding,
                            cv2.FaceRecognizerSF_FR_COSINE
                        )
                    )

                except Exception as e:

                    print(
                        "[FACE ERROR] "
                        f"SFace comparison failed: {e}"
                    )

                    continue

                score = float(
                    score
                )

                # --------------------------------------------
                # KEEP BEST SCORE FOR THIS PERSON
                # --------------------------------------------

                if score > person_best_score:

                    person_best_score = score

                # --------------------------------------------
                # KEEP GLOBAL BEST MATCH
                # --------------------------------------------

                if score > best_score:

                    best_score = score

                    best_name = (
                        person_name
                    )

            # ------------------------------------------------
            # DEBUG
            # ------------------------------------------------

            print(
                f"[FACE DEBUG] "
                f"{person_name}: "
                f"best score = "
                f"{person_best_score:.4f}"
            )

        # ====================================================
        # FINAL DEBUG
        # ====================================================

        print(
            f"[FACE DEBUG] "
            f"BEST MATCH = {best_name} | "
            f"BEST SCORE = {best_score:.4f}"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT make the final authorized/unknown decision
        # here.
        #
        # pipeline.py performs the final decision using:
        #
        #   name in AUTHORIZED_OFFICERS
        #
        # and:
        #
        #   score >= FACE_RECOGNITION_THRESHOLD
        #
        # ----------------------------------------------------

        if best_score < 0:

            return (
                "Unknown",
                0.0
            )

        return (
            best_name,
            best_score
        )