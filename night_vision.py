import cv2


def enhance_night_frame(frame):
    """
    Enhance a dark/low-light frame for better visibility.
    """

    # Convert BGR to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Split channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_l = clahe.apply(l_channel)

    # Merge channels
    enhanced_lab = cv2.merge(
        (enhanced_l, a_channel, b_channel)
    )

    # Convert back to BGR
    enhanced_frame = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    # Mild brightness adjustment
    enhanced_frame = cv2.convertScaleAbs(
        enhanced_frame,
        alpha=1.15,
        beta=10
    )

    return enhanced_frame