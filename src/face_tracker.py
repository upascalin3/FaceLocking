import time

class FaceTracker:
    def __init__(self, target_name, max_missing_time=2.0):
        import os

        self.target_name = target_name
        self.locked = False
        self.last_seen_time = None
        self.last_bbox = None
        self.max_missing_time = max_missing_time
        # Action tracking state (used by face_locking_controller)
        self.prev_center = None  # will hold np.array([x, y]) smoothed
        self.prev_time = None
        # counters to require N consecutive frames before confirming action
        self.action_counters = {
            "MOVE_LEFT": 0,
            "MOVE_RIGHT": 0,
            "SMILE": 0,
            "BLINK": 0,
        }
        # last time an action was logged (to avoid floods)
        self.last_action_times = {k: 0.0 for k in self.action_counters}

        # configuration: confirmation frames and cooldown seconds (overridable via env)
        self.action_confirm_frames = int(os.environ.get("ACTION_CONFIRM_FRAMES", "3"))
        self.action_cooldown_s = float(os.environ.get("ACTION_COOLDOWN_S", "1.2"))

        # smoothing for center (EMA) to reduce jitter
        self.center_smooth_alpha = float(os.environ.get("CENTER_SMOOTH_ALPHA", "0.6"))
        # speed threshold (px/sec) to complement distance threshold
        self.move_speed_thresh = float(os.environ.get("MOVE_SPEED_THRESH", "30.0"))

    def update(self, recognized_name, bbox):
        now = time.time()

        if recognized_name == self.target_name:
            self.locked = True
            self.last_seen_time = now
            self.last_bbox = bbox
            return "LOCKED"

        if self.locked:
            if now - self.last_seen_time <= self.max_missing_time:
                return "TRACKING"
            else:
                self.reset()
                return "UNLOCKED"

        return "SEARCHING"

    def reset(self):
        self.locked = False
        self.last_seen_time = None
        self.last_bbox = None