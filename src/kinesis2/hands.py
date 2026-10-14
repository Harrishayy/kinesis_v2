"""Hand tracks from phone video.

MediaPipe hand landmarks per frame -> a compact per-frame track:
    t          seconds
    uv         wrist position in normalised image coords (x right, y down)
    scale      palm width in normalised image units (proxy for camera distance)
    pinch      thumb-tip to index-tip distance / palm width (grip proxy, 0 = closed)
    ok         1 if a hand was detected this frame

Monocular, no calibration. Depth is deliberately not reconstructed: for a table-top
pick-and-place seen from above, in-plane motion is what matters and palm scale is a
good-enough proxy for height. See docs/DESIGN_DECISIONS.md.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, PINKY_MCP = 0, 4, 5, 8, 17


@dataclass
class HandTrack:
    t: np.ndarray       # (N,)
    uv: np.ndarray      # (N, 2)
    scale: np.ndarray   # (N,)
    pinch: np.ndarray   # (N,)
    ok: np.ndarray      # (N,) bool
    width: int
    height: int
    source: str

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **{k: v for k, v in asdict(self).items()})

    @classmethod
    def load(cls, path: Path) -> "HandTrack":
        d = np.load(path, allow_pickle=False)
        return cls(
            t=d["t"], uv=d["uv"], scale=d["scale"], pinch=d["pinch"], ok=d["ok"].astype(bool),
            width=int(d["width"]), height=int(d["height"]), source=str(d["source"]),
        )


def features_from_landmarks(lm: np.ndarray) -> tuple[np.ndarray, float, float]:
    """lm: (21, 2|3) normalised image landmarks. Returns (wrist_uv, palm_scale, pinch)."""
    lm = np.asarray(lm, dtype=np.float64)[:, :2]
    palm = np.linalg.norm(lm[INDEX_MCP] - lm[PINKY_MCP])
    palm = max(palm, 1e-6)
    pinch = np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP]) / palm
    return lm[WRIST].copy(), float(palm), float(pinch)


def ensure_model(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    p = models_dir / "hand_landmarker.task"
    if not p.exists():
        urllib.request.urlretrieve(MODEL_URL, p)
    return p


def extract_track(video: Path, model_path: Path, max_fps: float = 30.0) -> HandTrack:
    """Run MediaPipe HandLandmarker over a video in VIDEO mode."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(round(fps / max_fps)))

    ts, uvs, scales, pinches, oks = [], [], [], [], []
    with vision.HandLandmarker.create_from_options(opts) as lmk:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % stride:
                i += 1
                continue
            t_ms = int(round(1000.0 * i / fps))
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = lmk.detect_for_video(img, t_ms)
            ts.append(t_ms / 1000.0)
            if res.hand_landmarks:
                lm = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]])
                uv, sc, pn = features_from_landmarks(lm)
                uvs.append(uv); scales.append(sc); pinches.append(pn); oks.append(True)
            else:
                uvs.append([np.nan, np.nan]); scales.append(np.nan); pinches.append(np.nan); oks.append(False)
            i += 1
    cap.release()
    return HandTrack(
        t=np.array(ts), uv=np.array(uvs, dtype=np.float64), scale=np.array(scales),
        pinch=np.array(pinches), ok=np.array(oks, dtype=bool),
        width=width, height=height, source=video.name,
    )


def interpolate_gaps(track: HandTrack, max_gap_s: float = 0.5) -> HandTrack:
    """Linearly fill short detection dropouts; leave long gaps as NaN."""
    t, ok = track.t, track.ok
    if ok.sum() < 2:
        return track
    good = np.where(ok)[0]
    def fill(a: np.ndarray) -> np.ndarray:
        a = a.copy()
        if a.ndim == 1:
            a[~ok] = np.interp(t[~ok], t[good], a[good])
        else:
            for c in range(a.shape[1]):
                a[~ok, c] = np.interp(t[~ok], t[good], a[good, c])
        return a
    uv, scale, pinch = fill(track.uv), fill(track.scale), fill(track.pinch)
    # re-NaN anything inside a gap longer than max_gap_s
    gaps = np.diff(good)
    for k, g in enumerate(gaps):
        if t[good[k + 1]] - t[good[k]] > max_gap_s:
            sl = slice(good[k] + 1, good[k + 1])
            uv[sl] = np.nan; scale[sl] = np.nan; pinch[sl] = np.nan
    return HandTrack(t, uv, scale, pinch, ok, track.width, track.height, track.source)


def summary(track: HandTrack) -> dict:
    ok = track.ok
    return {
        "source": track.source,
        "frames": int(len(track.t)),
        "detected_frac": float(ok.mean()) if len(ok) else 0.0,
        "duration_s": float(track.t[-1] - track.t[0]) if len(track.t) > 1 else 0.0,
        "pinch_min": float(np.nanmin(track.pinch)) if ok.any() else None,
        "pinch_max": float(np.nanmax(track.pinch)) if ok.any() else None,
    }


def write_summary(rows: list[dict], path: Path) -> None:
    path.write_text(json.dumps(rows, indent=2))
