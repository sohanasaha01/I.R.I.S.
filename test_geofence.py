from geofence import Geofence


# Create a test zone
zone = Geofence([
    (0.25, 0.25),
    (0.75, 0.25),
    (0.75, 0.75),
    (0.25, 0.75)
])


# Test frame size
W = 320
H = 320


# Test bounding boxes
test_boxes = [

    # Frame 1 - outside
    (10, 10, 50, 50),

    # Frame 2 - outside
    (40, 40, 80, 80),

    # Frame 3 - inside
    (140, 140, 180, 180),

    # Frame 4 - inside
    (150, 150, 190, 190),

    # Frame 5 - inside
    (180, 180, 220, 220),

    # Frame 6 - outside
    (260, 260, 300, 300),
]


previous = False


for frame_no, bbox in enumerate(test_boxes, start=1):

    inside = zone.is_inside(
        bbox,
        W,
        H
    )

    if not previous and inside:

        print(
            f"Frame {frame_no}: ENTER"
        )

    elif previous and not inside:

        print(
            f"Frame {frame_no}: EXIT"
        )

    else:

        print(
            f"Frame {frame_no}: inside={inside}"
        )

    previous = inside