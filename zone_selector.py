import cv2
import json
import os

from backend import VideoSource


# ============================================================
# SELECTED SOURCE
# ============================================================

# For now, test with your video.
# Later the frontend will provide these values.

SOURCE_TYPE = "video"

SOURCE = "C:/border_project/test2.mp4"


# ============================================================
# ZONE FILE
# ============================================================

ZONE_FOLDER = "C:/border_project/zones"

os.makedirs(
    ZONE_FOLDER,
    exist_ok=True
)


# ============================================================
# CREATE SOURCE NAME
# ============================================================

if SOURCE_TYPE == "video":

    source_name = os.path.splitext(
        os.path.basename(SOURCE)
    )[0]

elif SOURCE_TYPE == "webcam":

    source_name = f"webcam_{SOURCE}"

else:

    source_name = "cctv_1"


ZONE_FILE = os.path.join(
    ZONE_FOLDER,
    source_name + ".json"
)


# ============================================================
# CONNECT TO SOURCE
# ============================================================

video_source = VideoSource(
    SOURCE_TYPE,
    SOURCE
)


if not video_source.connect():

    print("Could not connect to source.")

    exit()


# ============================================================
# GET FIRST FRAME
# ============================================================

ret, frame = video_source.read()


if not ret:

    print("Could not read frame.")

    video_source.release()

    exit()


# ============================================================
# FRAME DIMENSIONS
# ============================================================

frame_height, frame_width = frame.shape[:2]


print()
print("==============================================")
print("           GEOFENCE SELECTION")
print("==============================================")
print()

print(f"Source type : {SOURCE_TYPE}")
print(f"Source      : {SOURCE}")
print(f"Resolution  : {frame_width} x {frame_height}")
print(f"Zone file   : {ZONE_FILE}")

print()

print("Left Click  : Add point")
print("Right Click : Remove last point")
print("ENTER       : Save fence")
print("ESC         : Cancel")

print()


# ============================================================
# POINT STORAGE
# ============================================================

points = []


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    # --------------------------------------------------------
    # LEFT CLICK → ADD POINT
    # --------------------------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append(
            (x, y)
        )

        print(
            f"Point {len(points)}: ({x}, {y})"
        )


    # --------------------------------------------------------
    # RIGHT CLICK → REMOVE POINT
    # --------------------------------------------------------

    elif event == cv2.EVENT_RBUTTONDOWN:

        if points:

            removed = points.pop()

            print(
                f"Removed point: {removed}"
            )


# ============================================================
# CREATE WINDOW
# ============================================================

window_name = "Select Geofence"

cv2.namedWindow(
    window_name
)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)


# ============================================================
# SELECTION LOOP
# ============================================================

while True:

    display = frame.copy()


    # --------------------------------------------------------
    # DRAW POINTS
    # --------------------------------------------------------

    for i, (x, y) in enumerate(points):

        cv2.circle(
            display,
            (x, y),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display,
            str(i + 1),
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # --------------------------------------------------------
    # DRAW LINES
    # --------------------------------------------------------

    if len(points) >= 2:

        for i in range(
            len(points) - 1
        ):

            cv2.line(
                display,
                points[i],
                points[i + 1],
                (255, 0, 0),
                3
            )


    # --------------------------------------------------------
    # CLOSE POLYGON
    # --------------------------------------------------------

    if len(points) >= 3:

        cv2.line(
            display,
            points[-1],
            points[0],
            (255, 0, 0),
            3
        )


    # --------------------------------------------------------
    # SHOW FRAME
    # --------------------------------------------------------

    cv2.imshow(
        window_name,
        display
    )


    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # ENTER → SAVE
    # ========================================================

    if key == 13:

        if len(points) < 3:

            print()
            print(
                "Please select at least 3 points."
            )
            print()

            continue


        # ----------------------------------------------------
        # CONVERT TO NORMALIZED COORDINATES
        # ----------------------------------------------------

        normalized_points = []


        for x, y in points:

            nx = round(
                x / frame_width,
                6
            )

            ny = round(
                y / frame_height,
                6
            )


            normalized_points.append(
                [
                    nx,
                    ny
                ]
            )


        # ----------------------------------------------------
        # CREATE ZONE DATA
        # ----------------------------------------------------

        zone_data = {

            "source_type": SOURCE_TYPE,

            "source": str(SOURCE),

            "frame_width": frame_width,

            "frame_height": frame_height,

            "zone": normalized_points

        }


        # ----------------------------------------------------
        # SAVE JSON
        # ----------------------------------------------------

        with open(
            ZONE_FILE,
            "w"
        ) as f:

            json.dump(
                zone_data,
                f,
                indent=4
            )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        print()
        print("==============================================")
        print("       GEOFENCE SAVED SUCCESSFULLY")
        print("==============================================")
        print()

        print(
            f"Saved to: {ZONE_FILE}"
        )

        print()

        print(
            "Normalized coordinates:"
        )

        for point in normalized_points:

            print(point)

        print()


        break


    # ========================================================
    # ESC → CANCEL
    # ========================================================

    elif key == 27:

        print()
        print("Selection cancelled.")
        print()

        break


# ============================================================
# CLEANUP
# ============================================================

video_source.release()

cv2.destroyAllWindows()