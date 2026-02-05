```markdown
# Face Lock & Behavior Tracker  
**ArcFace ONNX + MediaPipe 5-Point + Identity Locking**

A lightweight, CPU-only system that goes beyond face recognition:  
it locks onto one person and tracks their basic behavior over time.

## Key Features

- Enroll users with multiple face samples  
- Real-time multi-face recognition  
- **Face Lock** mode — tracks only the selected identity  
- Detects simple actions: blink 😐 • smile 😊 • look left/right  
- Automatically logs timestamped behavior history

## Quick Start

```bash
# Step 1: Enroll faces (more samples = better accuracy)
python -m src.enroll
# SPACE = capture  |  a = auto-capture  |  s = save  |  q = quit

# Step 2: Test basic recognition
python -m src.recognize
# + / - = threshold  |  d = debug overlay  |  q = quit

# Step 3: Run Face Lock + behavior tracking
python main_lock.py
```

## Requirements

```bash
pip install opencv-python numpy onnxruntime mediapipe
```

Place `embedder_arcface.onnx` inside the `models/` folder.

## How Face Lock Works

1. Recognizes people normally  
2. You choose/lock one identity  
3. Switches to **Locked Mode** — ignores everyone else  
4. Keeps tracking even during short recognition drops  
5. Shows detected actions live  
6. Saves timeline → `data/<name>_history_YYYYMMDD_HHMMSS.txt`

Example log:
```
2025-10-15 14:32:07    BLINK      Eye aspect ratio < 0.22
2025-10-15 14:32:11    HEAD_LEFT  Nose moved -22 px
2025-10-15 14:32:18    SMILE      Mouth corners raised
```

## Folder Structure

```
face-recognition-5pt/
├── data/
│   ├── enroll/           ← put enrollment photos here
│   └── db/               ← face_db.npz (auto-created)
├── models/
│   └── embedder_arcface.onnx
├── src/                  ← all logic modules
└── main_lock.py          ← main Face Lock application
```

Simple. Fast. No GPU. Just a webcam. 🚀
```