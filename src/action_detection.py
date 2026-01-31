prev_nose_x = None
blink_counter = 0

def detect_head_movement(nose_x, threshold):
    global prev_nose_x
    action = None

    if prev_nose_x is not None:
        diff = nose_x - prev_nose_x
        if diff > threshold:
            action = "Moved Right"
        elif diff < -threshold:
            action = "Moved Left"

    prev_nose_x = nose_x
    return action


def detect_blink(eye_top, eye_bottom, threshold):
    global blink_counter
    ear = abs(eye_top[1] - eye_bottom[1])

    if ear < threshold:
        blink_counter += 1
    else:
        if blink_counter > 2:
            blink_counter = 0
            return "Blink"
        blink_counter = 0
    return None


def detect_smile(mouth_left, mouth_right, threshold):
    width = abs(mouth_right[0] - mouth_left[0])
    if width > threshold:
        return "Smile"
    return None
