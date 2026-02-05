import numpy as np
import time
from typing import Optional
from .face_tracker import FaceTracker
from .history_logger import start_history, log_action
from .actions import detect_horizontal_movement, detect_smile, detect_blink, eye_aspect_ratio

import os

# Allow configuring the target identity via environment variable TARGET_IDENTITY
TARGET_IDENTITY = os.environ.get("TARGET_IDENTITY", "pascaline")

tracker = FaceTracker(target_name=TARGET_IDENTITY)
history_file = None


def handle_face(name, bbox, landmarks=None, left_eye=None, right_eye=None):
    """
    Process a detected face.
    Only the locked person will have actions logged.
    """
    global history_file
    detected_action = None

    status = tracker.update(name, bbox)

    # Start history for locked person
    if status == "LOCKED" and history_file is None:
        # start_history creates a file named <face>_history_<timestamp>.txt in ./history
        history_file = start_history(name)
        log_action(history_file, "LOCKED", "Face successfully locked")

    # If we transitioned to unlocked, close any open history file
    if status == "UNLOCKED" and history_file is not None:
        log_action(history_file, "UNLOCKED", "Face unlocked / lost")
        try:
            history_file.close()
        except Exception:
            pass
        history_file = None

    # Only process actions for locked person
    if tracker.locked and name == tracker.target_name:
        # compute face center
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        current_center = (cx, cy)

        # Horizontal movement: use relative threshold based on face width
        face_w = bbox[2] - bbox[0]
        # relative threshold (fraction of face width)
        rel_thresh = max(20, int(0.12 * max(1.0, face_w)))

        if tracker.prev_center is not None:
            now = time.time()
            dt = now - (tracker.prev_time or now)
            dx = float(current_center[0] - tracker.prev_center[0])
            # velocity px/sec
            vel = dx / max(dt, 1e-6)

            # require both distance and speed thresholds to trigger
            moved_right = dx > rel_thresh and vel > tracker.move_speed_thresh
            moved_left = dx < -rel_thresh and vel < -tracker.move_speed_thresh

            if moved_right:
                tracker.action_counters["MOVE_RIGHT"] += 1
                tracker.action_counters["MOVE_LEFT"] = 0
            elif moved_left:
                tracker.action_counters["MOVE_LEFT"] += 1
                tracker.action_counters["MOVE_RIGHT"] = 0
            else:
                tracker.action_counters["MOVE_LEFT"] = 0
                tracker.action_counters["MOVE_RIGHT"] = 0

            # confirm and log MOVE_RIGHT
            if (
                tracker.action_counters["MOVE_RIGHT"] >= tracker.action_confirm_frames
                and now - tracker.last_action_times["MOVE_RIGHT"] > tracker.action_cooldown_s
            ):
                log_action(history_file, "MOVE_RIGHT", f"dx={dx:.1f} vel={vel:.1f}")
                print("[ACTION] MOVE_RIGHT")
                detected_action = "MOVE_RIGHT"
                tracker.last_action_times["MOVE_RIGHT"] = now
                tracker.action_counters["MOVE_RIGHT"] = 0

            # confirm and log MOVE_LEFT
            if (
                tracker.action_counters["MOVE_LEFT"] >= tracker.action_confirm_frames
                and now - tracker.last_action_times["MOVE_LEFT"] > tracker.action_cooldown_s
            ):
                log_action(history_file, "MOVE_LEFT", f"dx={dx:.1f} vel={vel:.1f}")
                print("[ACTION] MOVE_LEFT")
                detected_action = "MOVE_LEFT"
                tracker.last_action_times["MOVE_LEFT"] = now
                tracker.action_counters["MOVE_LEFT"] = 0

        # update smoothed prev_center and prev_time using EMA
        if tracker.prev_center is None:
            tracker.prev_center = np.array(current_center, dtype=np.float32)
        else:
            alpha = tracker.center_smooth_alpha
            tracker.prev_center = (alpha * np.array(current_center, dtype=np.float32)) + (
                (1.0 - alpha) * tracker.prev_center
            )
        tracker.prev_time = time.time()

        # Smile (require consecutive frames + cooldown)
        if landmarks is not None:
            mouth_width = float(np.linalg.norm(landmarks[3] - landmarks[4]))
            face_width = float(np.linalg.norm(landmarks[0] - landmarks[1]))
            now = time.time()
            if detect_smile(mouth_width, face_width):
                tracker.action_counters["SMILE"] += 1
            else:
                tracker.action_counters["SMILE"] = 0

            if (
                tracker.action_counters["SMILE"] >= tracker.action_confirm_frames
                and now - tracker.last_action_times["SMILE"] > tracker.action_cooldown_s
            ):
                ratio = mouth_width / max(1e-6, face_width)
                log_action(history_file, "SMILE", f"Smile detected (ratio={ratio:.3f})")
                detected_action = "SMILE"
                tracker.last_action_times["SMILE"] = now
                tracker.action_counters["SMILE"] = 0

        # Blink: only compute EAR if the provided eye landmarks are detailed
        # (i.e., have at least 6 points). Our 5-point pipeline may pass only
        # eye centers (shape (2,)), in which case skip blink detection to avoid
        # false positives or exceptions.
        def _has_detailed_eye(e):
            try:
                a = np.asarray(e)
                return a.ndim == 2 and a.shape[0] >= 6
            except Exception:
                return False

        # Blink detection with debounce + cooldown
        if _has_detailed_eye(left_eye) and _has_detailed_eye(right_eye):
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
            now = time.time()
            if detect_blink(ear):
                tracker.action_counters["BLINK"] += 1
            else:
                tracker.action_counters["BLINK"] = 0

            if (
                tracker.action_counters["BLINK"] >= 1
                and now - tracker.last_action_times["BLINK"] > tracker.action_cooldown_s
            ):
                log_action(history_file, "BLINK", f"Blink detected (EAR={ear:.3f})")
                detected_action = "BLINK"
                tracker.last_action_times["BLINK"] = now
                tracker.action_counters["BLINK"] = 0

    return detected_action


def get_current_locked() -> Optional[str]:
    """Return the name of the currently locked person, or None if nobody is locked."""
    return tracker.target_name if tracker.locked else None