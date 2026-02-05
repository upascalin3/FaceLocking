import numpy as np

def detect_horizontal_movement(prev_center, curr_center, threshold: int = 25):
    """
    Detect coarse horizontal movement between two centers.

    threshold: pixel threshold (default 25). If dx > threshold -> MOVE_RIGHT,
    if dx < -threshold -> MOVE_LEFT, else None.
    """
    dx = int(curr_center[0] - prev_center[0])
    if dx > threshold:
        return "MOVE_RIGHT"
    if dx < -threshold:
        return "MOVE_LEFT"
    return None


def eye_aspect_ratio(eye):
    """
    Compute Eye Aspect Ratio (EAR) from 6 eye landmarks in the order:
    [p0, p1, p2, p3, p4, p5] where p0..p5 are 2D points.

    If `eye` is only a single 2D center point (shape (2,) or (1,2)), or fewer
    than 6 points are provided, return a large EAR (no blink) to avoid false
    positives because we don't have enough data to compute EAR.
    """
    arr = np.asarray(eye)
    if arr.ndim == 1 and arr.size == 2:
        # only center point provided -> can't compute EAR
        return 1.0
    if arr.ndim == 2 and arr.shape[0] >= 6 and arr.shape[1] >= 2:
        A = np.linalg.norm(arr[1] - arr[5])
        B = np.linalg.norm(arr[2] - arr[4])
        C = np.linalg.norm(arr[0] - arr[3])
        # avoid division by zero
        if C <= 1e-6:
            return 1.0
        return float((A + B) / (2.0 * C))

    # fallback for unexpected shapes
    return 1.0


def detect_blink(ear, blink_threshold: float = 0.20):
    """Return True when EAR (eye aspect ratio) drops below threshold."""
    return float(ear) < float(blink_threshold)


def detect_smile(mouth_width, face_width, ratio: float = 0.50):
    """Detect smile by comparing mouth corner distance to inter-eye distance.

    The ratio defaults to 0.50 (i.e., mouth width >= 50% of eye distance).
    """
    if face_width <= 0:
        return False
    return (mouth_width / face_width) > float(ratio)