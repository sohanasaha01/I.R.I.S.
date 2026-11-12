import cv2


# ============================================================
# VIDEO SOURCE CLASS
# ============================================================

class VideoSource:

    def __init__(self, source_type, source):

        self.source_type = source_type
        self.source = source
        self.cap = None


    # ========================================================
    # CONNECT TO SOURCE
    # ========================================================

    def connect(self):

        # ----------------------------------------------------
        # VIDEO CLIP
        # ----------------------------------------------------

        if self.source_type == "video":

            self.cap = cv2.VideoCapture(
                self.source
            )


        # ----------------------------------------------------
        # LAPTOP / WEBCAM
        # ----------------------------------------------------

        elif self.source_type == "webcam":

            self.cap = cv2.VideoCapture(
                int(self.source)
            )


        # ----------------------------------------------------
        # CCTV / RTSP
        # ----------------------------------------------------

        elif self.source_type == "cctv":

            self.cap = cv2.VideoCapture(
                self.source
            )


        # ----------------------------------------------------
        # INVALID SOURCE TYPE
        # ----------------------------------------------------

        else:

            raise ValueError(
                "Invalid source type. "
                "Use 'video', 'webcam', or 'cctv'."
            )


        # ----------------------------------------------------
        # CHECK CONNECTION
        # ----------------------------------------------------

        if not self.cap.isOpened():

            print(
                f"Could not connect to {self.source_type}."
            )

            return False


        print(
            f"{self.source_type} connected successfully."
        )

        return True


    # ========================================================
    # READ FRAME
    # ========================================================

    def read(self):

        if self.cap is None:

            return False, None


        return self.cap.read()


    # ========================================================
    # RELEASE SOURCE
    # ========================================================

    def release(self):

        if self.cap is not None:

            self.cap.release()

            self.cap = None


# ============================================================
# GET FIRST FRAME
# ============================================================

def get_first_frame(source_type, source):

    """
    Connects to the selected source,
    reads the first frame,
    and releases the source.

    Returns:
        frame if successful
        None if unsuccessful
    """

    video_source = VideoSource(
        source_type,
        source
    )


    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    if not video_source.connect():

        return None


    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    ret, frame = video_source.read()


    # --------------------------------------------------------
    # RELEASE
    # --------------------------------------------------------

    video_source.release()


    # --------------------------------------------------------
    # CHECK FRAME
    # --------------------------------------------------------

    if not ret:

        print(
            "Could not read frame from source."
        )

        return None


    print(
        "First frame received successfully."
    )


    return frame