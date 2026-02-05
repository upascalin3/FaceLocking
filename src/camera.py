import os
import glob
import cv2
from typing import Optional


def open_video_capture(device: Optional[str] = None, max_index: int = 4, preferred_index: int = 2) -> cv2.VideoCapture:
    """
    Open a cv2.VideoCapture with a preference for a fixed index using V4L2.

    Behavior:
    - First try opening `preferred_index` with `cv2.CAP_V4L2` (e.g. index 2),
      because some systems work better with this flag for USB cameras.
    - If `device` is provided, try it (as an int index or device path).
    - If env CAMERA_DEVICE / CAMERA_INDEX is set, try that.
    - Try integer indices 0..max_index.
    - Fallback: try any `/dev/video*` entries.
    - Returns a cv2.VideoCapture (may be not opened).
    """

    # 1) Preferred explicit open using CAP_V4L2
    try:
        cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    except Exception:
        pass

    # 2) explicit arg > env > default
    if device is None:
        device = os.environ.get("CAMERA_DEVICE") or os.environ.get("CAMERA_INDEX")

    def try_open(d):
        # try index
        try:
            cap = cv2.VideoCapture(int(d))
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            pass

        # try as path (string)
        try:
            cap = cv2.VideoCapture(d)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            pass

        return None

    if device:
        cap = try_open(device)
        if cap:
            return cap

    # 3) try indices 0..max_index (without CAP_V4L2 flag)
    for i in range(0, max_index + 1):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            continue

    # 4) fallback: try any /dev.video* (Linux)
    for p in sorted(glob.glob("/dev/video*")):
        cap = try_open(p)
        if cap:
            return cap

    # give up: return an unopened capture
    return cv2.VideoCapture(-1)


def main():
    cap = open_video_capture()

    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try setting CAMERA_DEVICE or CAMERA_INDEX.")

    print("Camera test. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        cv2.imshow("Camera Test", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()