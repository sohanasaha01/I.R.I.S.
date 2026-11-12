import cv2
import os
import sys
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\border_project"

IMAGE_OUTPUT_DIR = os.path.join(
    BASE_DIR, "dataset", "images", "all"
)

LABEL_OUTPUT_DIR = os.path.join(
    BASE_DIR, "dataset", "labels", "all"
)

# ============================================================
# YOLO CLASS IDs
# ============================================================

CLASS_IDS = {
    "person": 0,
    "car": 1,
    "bomb": 2,
    "knife": 3,
    "gun": 4,
    "number_plate": 5,
    "helicopter": 6,
    "tank": 7
}

WINDOW_NAME = "YOLO Annotation"

# Smaller window so it fits on the screen
MAX_WIDTH = 900
MAX_HEIGHT = 600

# Allow small images to be enlarged
MAX_SCALE = 4.0


# ============================================================
# CHECK COMMAND-LINE ARGUMENTS
# ============================================================

if len(sys.argv) != 3:

    print()
    print("Usage:")
    print("python annotate.py person 0")
    print("python annotate.py car 1")
    print("python annotate.py bomb 2")
    print("python annotate.py knife 3")
    print("python annotate.py gun 4")
    print("python annotate.py number_plate 5")
    print("python annotate.py helicopter 6")
    print("python annotate.py tank 7")
    print()

    sys.exit(1)


CLASS_NAME = sys.argv[1]

try:

    CLASS_ID = int(sys.argv[2])

except ValueError:

    print("Class ID must be a number.")
    sys.exit(1)


# ============================================================
# VALIDATE CLASS
# ============================================================

if CLASS_NAME not in CLASS_IDS:

    print(f"Invalid class name: {CLASS_NAME}")
    sys.exit(1)


if CLASS_IDS[CLASS_NAME] != CLASS_ID:

    print(
        f"Class mismatch: {CLASS_NAME} should use "
        f"class ID {CLASS_IDS[CLASS_NAME]}, not {CLASS_ID}."
    )

    sys.exit(1)


# ============================================================
# SOURCE DIRECTORY
# ============================================================

SOURCE_DIR = os.path.join(
    BASE_DIR,
    "raw_images",
    CLASS_NAME
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(
    IMAGE_OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    LABEL_OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CHECK SOURCE DIRECTORY
# ============================================================

if not os.path.exists(SOURCE_DIR):

    print()
    print("Source directory not found:")
    print(SOURCE_DIR)
    print()

    sys.exit(1)


# ============================================================
# GET IMAGE FILES
# ============================================================

image_files = []

for filename in os.listdir(SOURCE_DIR):

    file_path = os.path.join(
        SOURCE_DIR,
        filename
    )

    if not os.path.isfile(file_path):
        continue

    extension = os.path.splitext(filename)[1].lower()

    if extension in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    ]:

        image_files.append(filename)


image_files.sort()


if len(image_files) == 0:

    print()
    print("No images found in:")
    print(SOURCE_DIR)
    print()

    sys.exit(1)


# ============================================================
# INFORMATION
# ============================================================

print()
print("========================================")
print("YOLO MULTI-BOX ANNOTATION TOOL")
print("========================================")
print(f"Class      : {CLASS_NAME}")
print(f"Class ID   : {CLASS_ID}")
print(f"Images     : {len(image_files)}")
print(f"Source     : {SOURCE_DIR}")
print("========================================")
print()
print("Controls:")
print("  Drag mouse = Draw bounding box")
print("  S          = Save all boxes + next")
print("  R          = Reset all boxes")
print("  U          = Undo last box")
print("  N          = Skip image")
print("  Q          = Quit")
print()


# ============================================================
# GLOBAL VARIABLES
# ============================================================

drawing = False

start_x = 0
start_y = 0

end_x = 0
end_y = 0

boxes = []

canvas = None

offset_x = 0
offset_y = 0

original_width = 0
original_height = 0

display_width = 0
display_height = 0

scale = 1.0


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global drawing
    global start_x
    global start_y
    global end_x
    global end_y

    if event == cv2.EVENT_LBUTTONDOWN:

        if (
            x < offset_x
            or x >= offset_x + display_width
            or y < offset_y
            or y >= offset_y + display_height
        ):

            return

        drawing = True

        start_x = x
        start_y = y

        end_x = x
        end_y = y


    elif event == cv2.EVENT_MOUSEMOVE:

        if drawing:

            end_x = x
            end_y = y


    elif event == cv2.EVENT_LBUTTONUP:

        if drawing:

            drawing = False

            end_x = x
            end_y = y

            x1 = min(
                start_x,
                end_x
            )

            y1 = min(
                start_y,
                end_y
            )

            x2 = max(
                start_x,
                end_x
            )

            y2 = max(
                start_y,
                end_y
            )

            if (
                abs(x2 - x1) > 5
                and abs(y2 - y1) > 5
            ):

                boxes.append(
                    (
                        x1,
                        y1,
                        x2,
                        y2
                    )
                )

                print(
                    f"Box {len(boxes)} added: "
                    f"({x1}, {y1}) - ({x2}, {y2})"
                )


# ============================================================
# CREATE WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    MAX_WIDTH,
    MAX_HEIGHT
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

index = 0


while index < len(image_files):

    filename = image_files[index]

    image_path = os.path.join(
        SOURCE_DIR,
        filename
    )


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"Could not load image: {filename}"
        )

        index += 1

        continue


    original_height, original_width = image.shape[:2]


    # ========================================================
    # CALCULATE DISPLAY SCALE
    # ========================================================

    scale_x = MAX_WIDTH / original_width

    scale_y = MAX_HEIGHT / original_height

    scale = min(
        scale_x,
        scale_y
    )

    # Allow small images to enlarge

    scale = min(
        scale,
        MAX_SCALE
    )


    display_width = max(
        1,
        int(original_width * scale)
    )

    display_height = max(
        1,
        int(original_height * scale)
    )


    # ========================================================
    # RESIZE IMAGE
    # ========================================================

    resized = cv2.resize(
        image,
        (
            display_width,
            display_height
        ),
        interpolation=cv2.INTER_AREA
    )


    # ========================================================
    # CREATE CANVAS
    # ========================================================

    canvas = np.ones(
        (
            MAX_HEIGHT,
            MAX_WIDTH,
            3
        ),
        dtype=np.uint8
    ) * 255


    # ========================================================
    # CENTER IMAGE
    # ========================================================

    offset_x = (
        MAX_WIDTH - display_width
    ) // 2

    offset_y = (
        MAX_HEIGHT - display_height
    ) // 2


    canvas[
        offset_y:
        offset_y + display_height,

        offset_x:
        offset_x + display_width
    ] = resized


    # ========================================================
    # RESET BOXES
    # ========================================================

    boxes = []

    drawing = False

    start_x = 0
    start_y = 0

    end_x = 0
    end_y = 0


    # ========================================================
    # IMAGE INFORMATION
    # ========================================================

    print()

    print(
        f"[{index + 1}/{len(image_files)}] "
        f"{filename}"
    )

    print(
        f"Original size: "
        f"{original_width} x {original_height}"
    )

    print(
        f"Display size: "
        f"{display_width} x {display_height}"
    )


    # ========================================================
    # DISPLAY LOOP
    # ========================================================

    while True:

        frame = canvas.copy()


        # ----------------------------------------------------
        # DRAW SAVED BOXES
        # ----------------------------------------------------

        for box_number, box in enumerate(boxes):

            x1, y1, x2, y2 = box

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                str(box_number + 1),
                (
                    x1 + 5,
                    y1 + 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


        # ----------------------------------------------------
        # DRAW CURRENT BOX
        # ----------------------------------------------------

        if drawing:

            cv2.rectangle(
                frame,
                (
                    start_x,
                    start_y
                ),
                (
                    end_x,
                    end_y
                ),
                (0, 0, 255),
                2
            )


        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"Class: {CLASS_NAME} | Boxes: {len(boxes)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            "S=Save  R=Reset  U=Undo  N=Skip  Q=Quit",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


        # ----------------------------------------------------
        # SHOW
        # ----------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(20) & 0xFF


        # ====================================================
        # SAVE
        # ====================================================

        if key == ord("s"):

            if len(boxes) == 0:

                print(
                    "No bounding boxes drawn."
                )

                continue


            label_lines = []


            # ------------------------------------------------
            # CONVERT EVERY BOX
            # ------------------------------------------------

            for box in boxes:

                (
                    x1_display,
                    y1_display,
                    x2_display,
                    y2_display
                ) = box


                # Remove canvas offset

                x1_display -= offset_x
                y1_display -= offset_y

                x2_display -= offset_x
                y2_display -= offset_y


                # Convert to original image coordinates

                x1 = int(
                    x1_display / scale
                )

                y1 = int(
                    y1_display / scale
                )

                x2 = int(
                    x2_display / scale
                )

                y2 = int(
                    y2_display / scale
                )


                # Clip coordinates

                x1 = max(
                    0,
                    min(
                        x1,
                        original_width - 1
                    )
                )

                y1 = max(
                    0,
                    min(
                        y1,
                        original_height - 1
                    )
                )

                x2 = max(
                    0,
                    min(
                        x2,
                        original_width - 1
                    )
                )

                y2 = max(
                    0,
                    min(
                        y2,
                        original_height - 1
                    )
                )


                # Sort coordinates

                x1, x2 = sorted(
                    [
                        x1,
                        x2
                    ]
                )

                y1, y2 = sorted(
                    [
                        y1,
                        y2
                    ]
                )


                box_width = x2 - x1

                box_height = y2 - y1


                if (
                    box_width <= 0
                    or box_height <= 0
                ):

                    continue


                # ------------------------------------------------
                # YOLO FORMAT
                # ------------------------------------------------

                center_x = (
                    x1 + x2
                ) / 2

                center_y = (
                    y1 + y2
                ) / 2


                yolo_x = (
                    center_x /
                    original_width
                )

                yolo_y = (
                    center_y /
                    original_height
                )

                yolo_width = (
                    box_width /
                    original_width
                )

                yolo_height = (
                    box_height /
                    original_height
                )


                # Safety clipping

                yolo_x = max(
                    0.0,
                    min(
                        1.0,
                        yolo_x
                    )
                )

                yolo_y = max(
                    0.0,
                    min(
                        1.0,
                        yolo_y
                    )
                )

                yolo_width = max(
                    0.0,
                    min(
                        1.0,
                        yolo_width
                    )
                )

                yolo_height = max(
                    0.0,
                    min(
                        1.0,
                        yolo_height
                    )
                )


                label_lines.append(
                    f"{CLASS_ID} "
                    f"{yolo_x:.6f} "
                    f"{yolo_y:.6f} "
                    f"{yolo_width:.6f} "
                    f"{yolo_height:.6f}"
                )


            # =================================================
            # SAVE IMAGE
            # =================================================

            output_image_path = os.path.join(
                IMAGE_OUTPUT_DIR,
                filename
            )

            success = cv2.imwrite(
                output_image_path,
                image
            )

            if not success:

                print(
                    f"Could not save image: "
                    f"{output_image_path}"
                )

                continue


            # =================================================
            # SAVE LABEL FILE
            # =================================================

            label_filename = (
                os.path.splitext(filename)[0]
                + ".txt"
            )

            label_path = os.path.join(
                LABEL_OUTPUT_DIR,
                label_filename
            )


            with open(
                label_path,
                "w"
            ) as f:

                f.write(
                    "\n".join(label_lines)
                    + "\n"
                )


            # =================================================
            # PRINT RESULTS
            # =================================================

            print()

            print(
                f"Saved: {filename}"
            )

            print(
                f"Number of boxes: "
                f"{len(label_lines)}"
            )

            for line in label_lines:

                print(
                    " ",
                    line
                )

            print()


            index += 1

            break


        # ====================================================
        # RESET
        # ====================================================

        elif key == ord("r"):

            boxes = []

            print(
                "All boxes reset."
            )


        # ====================================================
        # UNDO
        # ====================================================

        elif key == ord("u"):

            if boxes:

                removed = boxes.pop()

                print(
                    f"Removed box: {removed}"
                )

            else:

                print(
                    "No boxes to undo."
                )


        # ====================================================
        # SKIP
        # ====================================================

        elif key == ord("n"):

            print(
                f"Skipped: {filename}"
            )

            index += 1

            break


        # ====================================================
        # QUIT
        # ====================================================

        elif key == ord("q"):

            cv2.destroyAllWindows()

            print()

            print(
                "Annotation stopped."
            )

            sys.exit(0)


# ============================================================
# FINISHED
# ============================================================

cv2.destroyAllWindows()

print()

print("========================================")
print("ANNOTATION COMPLETED")
print("========================================")

print()

print("Images saved to:")
print(IMAGE_OUTPUT_DIR)

print()

print("Labels saved to:")
print(LABEL_OUTPUT_DIR)

print()

print("========================================")