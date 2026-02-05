# src/detect.py
import cv2
import numpy as np
from .recognize import recognize_face
from .face_locking_controller import handle_face, get_current_locked
from .camera import open_video_capture
from .camera import open_video_capture

def main():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade")

    cap = open_video_capture()
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Set CAMERA_DEVICE or CAMERA_INDEX or pass device env.")

    print("Face recognition + face locking running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]

            # 🔹 Recognition (ArcFace + 5pt alignment)
            name, confidence, landmarks = recognize_face(face_img)

            # Optional: placeholders for eye landmarks (blink detection disabled)
            left_eye = None
            right_eye = None

            # 🔹 Face locking logic (only locked person actions are logged)
            action = handle_face(name, (x, y, x + w, y + h), landmarks, left_eye, right_eye)

            # 🔹 Draw face box and name
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{name}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # Show locked/unlocked sign on the box for recognized identities
            locked_name = get_current_locked()
            if name != "Unknown":
                if locked_name and name == locked_name:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 4)
                    cv2.putText(frame, "LOCKED", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                else:
                    cv2.putText(frame, "UNLOCKED", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 200), 2)

            # 🔹 Draw action if any
            if action:
                cv2.putText(
                    frame,
                    f"ACTION: {action}",
                    (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        cv2.imshow("Face Locking System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()