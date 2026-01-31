import cv2
from src.config import *
from src.face_lock import FaceLock
from src.action_detection import *
from src.history_logger import HistoryLogger

from src.recognize import recognize_faces   # your existing function

face_lock = FaceLock(TARGET_IDENTITY, MAX_LOST_FRAMES)
logger = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Your recognition returns multiple faces
    faces = recognize_faces(frame)
    print("Detected faces:", faces)

    # Expected format per face:
    # {
    #   "name": identity_name,
    #   "similarity": similarity_score,
    #   "bbox": (x1,y1,x2,y2),
    #   "landmarks": [left_eye, right_eye, nose, mouth_left, mouth_right]
    # }

    for face in faces:
        name = face["name"]
        similarity = face["similarity"]
        bbox = face["bbox"]
        landmarks = face["landmarks"]

        locked_now = face_lock.try_lock(name, similarity, LOCK_THRESHOLD, bbox)

        if locked_now and logger is None:
            logger = HistoryLogger(name)

        face_lock.update_tracking(name, bbox)

        if face_lock.locked and name == face_lock.locked_id:
            left_eye, right_eye, nose, mouth_left, mouth_right = landmarks

            move = detect_head_movement(nose[0], MOVEMENT_THRESHOLD)
            blink = detect_blink(left_eye, right_eye, BLINK_EAR_THRESHOLD)
            smile = detect_smile(mouth_left, mouth_right, SMILE_WIDTH_THRESHOLD)

            for action in [move, blink, smile]:
                if action:
                    print("[ACTION]", action)
                    logger.log(action)

            cv2.putText(frame, "LOCKED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Face Locking System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
