# src/recognize.py
"""
Multi-face recognition (CPU-friendly) using your now-stable pipeline:
Haar -> FaceMesh 5pt (ROI) -> align -> ArcFace -> cosine distance
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

from .haar_5pt import align_face_5pt
from .camera import open_video_capture
from .face_locking_controller import handle_face, get_current_locked


# -------------------------
# Data
# -------------------------
@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # (5,2)


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


# -------------------------
# Lazy single-image recognizer API
# -------------------------
_DET: Optional["HaarFaceMesh5pt"] = None
_EMBEDDER: Optional["ArcFaceEmbedderONNX"] = None
_MATCHER: Optional["FaceDBMatcher"] = None
_DB_PATH = Path("data/db/face_db.npz")


def _lazy_init_singleton() -> None:
    """Initialize global detector/embedder/matcher once for per-frame use.

    This is used by `recognize_face` so that `detect.py` can call into a
    simple function without reloading models on every frame.
    """
    global _DET, _EMBEDDER, _MATCHER

    if _DET is None:
        _DET = HaarFaceMesh5pt()
    if _EMBEDDER is None:
        _EMBEDDER = ArcFaceEmbedderONNX()
    if _MATCHER is None:
        db = load_db_npz(_DB_PATH)
        _MATCHER = FaceDBMatcher(db, dist_thresh=0.34)


def recognize_face(face_img) -> Tuple[str, float, Optional[np.ndarray]]:
    """Recognize a single face image.

    Parameters
    ----------
    face_img : np.ndarray (H, W, 3) BGR
        Cropped face region from the original frame.

    Returns
    -------
    name : str
        Matched person name or "Unknown".
    confidence : float
        Similarity score in [0, 1]. Higher is better.
    kps : np.ndarray | None
        5 facial landmarks (if detected), else None.
    """

    _lazy_init_singleton()

    assert _DET is not None and _EMBEDDER is not None and _MATCHER is not None

    # We run the 5-point detection on the provided crop. If it fails,
    # we fall back to "Unknown".
    faces = _DET.detect(face_img, max_faces=1)
    if not faces:
        return "Unknown", 0.0, None

    f = faces[0]
    aligned, _ = align_face_5pt(face_img, f.kps)
    emb = _EMBEDDER.embed(aligned)
    mr = _MATCHER.match(emb)

    name = mr.name if mr.name is not None else "Unknown"
    confidence = mr.similarity
    return name, confidence, f.kps


# -------------------------
# Math helpers
# -------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


def _clip_xyxy(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int
) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    k = kps.astype(np.float32)
    x_min, x_max = k[:, 0].min(), k[:, 0].max()
    y_min, y_max = k[:, 1].min(), k[:, 1].max()
    w, h = max(1.0, x_max - x_min), max(1.0, y_max - y_min)

    return np.array(
        [
            x_min - pad_x * w,
            y_min - pad_y_top * h,
            x_max + pad_x * w,
            y_max + pad_y_bot * h,
        ],
        dtype=np.float32,
    )


def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    le, re, no, lm, rm = kps.astype(np.float32)
    if np.linalg.norm(re - le) < min_eye_dist:
        return False
    if not (lm[1] > no[1] and rm[1] > no[1]):
        return False
    return True


# -------------------------
# DB helpers
# -------------------------
def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    return {k: np.asarray(data[k], dtype=np.float32).reshape(-1) for k in data.files}


# -------------------------
# Embedder
# -------------------------
class ArcFaceEmbedderONNX:
    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        self.in_w, self.in_h = input_size
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.debug = debug

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        if img.shape[:2] != (self.in_h, self.in_w):
            img = cv2.resize(img, (self.in_w, self.in_h))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        return np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    def embed(self, img: np.ndarray) -> np.ndarray:
        x = self._preprocess(img)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        return self._l2_normalize(y.reshape(-1).astype(np.float32))


# -------------------------
# Haar + FaceMesh detector
# -------------------------
class HaarFaceMesh5pt:
    def __init__(
        self,
        min_size: Tuple[int, int] = (70, 70),
        debug: bool = False,
    ):
        if mp is None:
            raise RuntimeError(_MP_IMPORT_ERROR)

        self.debug = debug
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.min_size = min_size
        self.idxs = [33, 263, 1, 61, 291]

    def detect(self, frame: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=self.min_size
        )

        out: List[FaceDet] = []
        for (x, y, w, h) in faces[:max_faces]:
            roi = frame[y : y + h, x : x + w]
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue

            lm = res.multi_face_landmarks[0].landmark
            kps = np.array(
                [[lm[i].x * w + x, lm[i].y * h + y] for i in self.idxs],
                dtype=np.float32,
            )

            if not _kps_span_ok(kps, max(10.0, 0.18 * w)):
                continue

            bb = _bbox_from_5pt(kps)
            x1, y1, x2, y2 = _clip_xyxy(*bb, W, H)

            out.append(FaceDet(x1, y1, x2, y2, 1.0, kps))

        return out


# -------------------------
# Matcher
# -------------------------
class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float):
        self.db = db
        self.dist_thresh = dist_thresh
        self.names = sorted(db.keys())
        self.mat = (
            np.stack([db[n] for n in self.names]).astype(np.float32)
            if self.names
            else None
        )

    def reload(self, path: Path):
        self.__init__(load_db_npz(path), self.dist_thresh)

    def match(self, emb: np.ndarray) -> MatchResult:
        if self.mat is None:
            return MatchResult(None, 1.0, 0.0, False)

        # (N, D) @ (D,) -> (N,) similarity vector
        sims = self.mat @ emb.reshape(-1)
        i = int(np.argmax(sims))
        sim = float(sims[i])
        dist = 1.0 - sim
        ok = dist <= self.dist_thresh

        return MatchResult(self.names[i] if ok else None, dist, sim, ok)


# -------------------------
# Demo
# -------------------------
def main():
    db_path = Path("data/db/face_db.npz")

    det = HaarFaceMesh5pt()
    embedder = ArcFaceEmbedderONNX()
    matcher = FaceDBMatcher(load_db_npz(db_path), dist_thresh=0.34)

    cap = open_video_capture()
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    print("Recognize: q quit | r reload | +/- threshold")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        faces = det.detect(frame)

        for f in faces:
            aligned, _ = align_face_5pt(frame, f.kps)
            emb = embedder.embed(aligned)
            mr = matcher.match(emb)
            color = (0, 255, 0) if mr.accepted else (0, 0, 255)
            label = mr.name if mr.name else "Unknown"

            # draw bounding box + label
            cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
            cv2.putText(
                vis,
                f"{label} d={mr.distance:.3f}",
                (f.x1, f.y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # Build bbox and landmarks for action detection
            bbox = (f.x1, f.y1, f.x2, f.y2)
            # f.kps order: left_eye, right_eye, nose, mouth_left, mouth_right
            detected_action = handle_face(label if label != "Unknown" else None, bbox, landmarks=f.kps, left_eye=f.kps[0], right_eye=f.kps[1])
            if detected_action:
                cv2.putText(vis, detected_action, (f.x1, f.y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                print(f"[DETECTED ACTION] {detected_action} for {label}")

            # Show locked/unlocked sign on the face box for recognized identities
            locked_name = get_current_locked()
            if label != "Unknown":
                if locked_name and label == locked_name:
                    # Emphasize locked face
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 200, 0), 4)
                    cv2.putText(vis, "LOCKED", (f.x1, f.y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                else:
                    # recognized but not locked
                    cv2.putText(vis, "UNLOCKED", (f.x1, f.y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

        # Display current locked identity at top-left
        locked_name = get_current_locked()
        if locked_name:
            lock_text = f"LOCKED: {locked_name}"
            lock_color = (0, 200, 0)
        else:
            lock_text = "LOCKED: None"
            lock_color = (0, 0, 200)
        cv2.putText(vis, lock_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, lock_color, 2)

        cv2.imshow("recognize", vis)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            matcher.reload(db_path)
            print("DB reloaded")
        elif key in (ord("+"), ord("=")):
            matcher.dist_thresh += 0.01
            print("thr =", matcher.dist_thresh)
        elif key == ord("-"):
            matcher.dist_thresh -= 0.01
            print("thr =", matcher.dist_thresh)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()