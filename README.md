* Face Recognition & Face Locking System

(5-Point Landmark + ArcFace ONNX + Behavior Tracking)

 *Overview*

This project implements a CPU-friendly multi-face recognition and tracking system using OpenCV, MediaPipe FaceMesh, and ArcFace ONNX embeddings.

It supports:

✔ Enrollment of new users
✔ Real-time multi-face recognition
✔ Face Locking (identity-based tracking)
✔ Detection of simple facial actions
✔ Logging a timeline of user actions

The system moves beyond recognition and into behavior tracking over time.

* Full Processing Pipeline

Camera → Haar Face Detection → FaceMesh 5-Point Landmarks → Face Alignment (112×112) → ArcFace ONNX Embedding → Cosine Similarity Matching → Face Locking → Action Detection → History Logging

*Requirements*

Python 3.12

opencv-python

numpy

onnxruntime

mediapipe

Install dependencies:

pip install opencv-python numpy onnxruntime mediapipe

* Project Structure
face-recognition-5pt/
│
├─ data/
│ ├─ enroll/                # Raw enrollment images
│ └─ db/                    # Face embedding database (face_db.npz)
│
├─ models/
│ └─ embedder_arcface.onnx  # ArcFace ONNX embedding model
│
├─ src/
│ ├─ camera.py
│ ├─ detect.py
│ ├─ landmarks.py
│ ├─ align.py
│ ├─ embed.py
│ ├─ enroll.py
│ ├─ recognize.py           # Multi-face recognition
│ ├─ evaluate.py
│ ├─ haar_5pt.py
│ │
│ ├─ config.py              # 🔒 Face Lock settings
│ ├─ face_lock.py           # 🔒 Locking & tracking logic
│ ├─ action_detection.py    # 😊 Blink, smile, head movement detection
│ └─ history_logger.py      # 📝 Action history recording
│
├─ main_lock.py             # 🚀 Face Locking system runner
└─ README.md

* PART 1 — FACE ENROLLMENT
Run enrollment:
py -3.12 -m src.enroll

Controls:
Key	Action
SPACE	Capture current frame
a	Auto-capture multiple frames
s	Save captured embeddings
r	Reset current session
q	Quit

Faces are aligned using 5-point landmarks and converted into ArcFace embeddings stored in:

data/db/face_db.npz


More samples = better recognition.

* PART 2 — FACE RECOGNITION
Run recognition:
py -3.12 -m src.recognize


System detects multiple faces and identifies them in real time.

Controls:
Key	Action
q	Quit
r	Reload database
+/-	Adjust recognition threshold
d	Toggle debug overlay

Matching uses cosine distance:

distance = 1 - cosine_similarity(embedding, db_embedding)



* PART 3 — FACE LOCKING & BEHAVIOR TRACKING

This extends recognition into identity-based tracking over time.

Run Face Locking System:
py -3.12 main_lock.py

 What Face Locking Does

When the selected identity appears:

1️⃣ The system recognizes the person
2️⃣ It locks onto that identity
3️⃣ It tracks that same face across frames
4️⃣ It ignores other faces
5️⃣ It detects simple facial actions
6️⃣ It records a timeline of actions to a file

* Locking Behavior

The system uses a state-based lock mechanism:

* State	Description
 Recognition Mode	All faces are analyzed normally
 Locked Mode	Only the selected identity is tracked

The lock remains active even if recognition briefly fails, and only releases if the face disappears for several frames.

* Detected Actions (While Locked)
Action	How It’s Detected
Move Left	Nose X coordinate decreases
Move Right	Nose X coordinate increases
Blink	Eye landmark vertical distance shrinks
Smile	Mouth corner distance increases

These are simple, explainable heuristics — high precision is not required.

* Action History Recording

When a face is locked, a history file is created automatically:

<name>_history_<timestamp>.txt

Each entry contains:

Timestamp

Action type

Description (if needed)

This demonstrates behavior tracking over time.


*  Final Result

This project demonstrates the transition from:

Face Recognition → Intelligent Face Tracking System

It not only identifies a person, but also follows their behavior over time and logs