"""
Server-side configuration (Chapter 3.5 / 4.7): the browser posts frames, the
server runs the identical landmark + feature + classification pipeline used in
training, and returns scores. Built primarily as the comparison baseline so its
bandwidth / latency / load costs can be measured directly.

Endpoints
  POST /infer          one frame (base64 JPEG) -> updates that session's window,
                       returns current tiredness / boredom scores + timing
  POST /session/reset  clear a session's temporal window
  GET  /health

The server keeps a per-session TemporalAggregator so the client can stream frames.
Only DERIVED measurements are persisted (SQLite); frames are never stored
(privacy-by-design, Chapter 4.8).

Run:
  uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import base64
import time
import io
import os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.features.landmarks import LandmarkExtractor
from src.features.smoothing import LandmarkSmoother
from src.features.geometry import per_frame_features, FRAME_FEATURE_ORDER
from src.features.temporal import TemporalAggregator
from src.models.boredom_fusion import preprocess_face
from server.db import Database

app = FastAPI(title="Engagement Detector (server-side baseline)")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
    allow_headers=["*"],
)

# --- lazy singletons ---
_extractor: LandmarkExtractor | None = None
_tiredness_model = None
_boredom_model = None
_db = Database(os.environ.get("DB_PATH", "eval.db"))

# per-session state
_sessions: dict[str, dict] = {}

FPS = float(os.environ.get("FPS", "10"))
WINDOW_S = float(os.environ.get("WINDOW_SECONDS", "5"))


def get_extractor() -> LandmarkExtractor:
    global _extractor
    if _extractor is None:
        _extractor = LandmarkExtractor(static_image_mode=True)
    return _extractor


def get_models():
    global _tiredness_model, _boredom_model
    if _tiredness_model is None:
        import tensorflow as tf
        t_path = os.environ.get("TIREDNESS_MODEL",
                                "models_out/tiredness/tiredness_keras.keras")
        b_path = os.environ.get("BOREDOM_MODEL",
                                "models_out/boredom/boredom_keras.keras")
        _tiredness_model = tf.keras.models.load_model(t_path) \
            if os.path.exists(t_path) else None
        _boredom_model = tf.keras.models.load_model(b_path) \
            if os.path.exists(b_path) else None
    return _tiredness_model, _boredom_model


def session(sid: str) -> dict:
    if sid not in _sessions:
        _sessions[sid] = {
            "agg": TemporalAggregator(fps=FPS, window_seconds=WINDOW_S),
            "smoother": LandmarkSmoother(alpha=0.5),
        }
    return _sessions[sid]


class InferRequest(BaseModel):
    session_id: str
    image_b64: str            # base64-encoded JPEG/PNG frame
    record: bool = False      # persist derived measurement for the eval study
    self_report: str | None = None


class InferResponse(BaseModel):
    tiredness: float | None
    boredom: float | None
    face_present: bool
    window_ready: bool
    server_ms: float


def decode_image(b64: str):
    import cv2
    raw = base64.b64decode(b64.split(",")[-1])
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@app.get("/health")
def health():
    t, b = get_models()
    return {"status": "ok",
            "tiredness_model": t is not None,
            "boredom_model": b is not None}


@app.post("/session/reset")
def reset(session_id: str):
    s = session(session_id)
    s["agg"].reset()
    s["smoother"].reset()
    return {"reset": True}


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    t0 = time.perf_counter()
    s = session(req.session_id)
    agg: TemporalAggregator = s["agg"]
    smoother: LandmarkSmoother = s["smoother"]

    frame = decode_image(req.image_b64)
    h, w = frame.shape[:2]
    lm = smoother.smooth(get_extractor().extract(frame))

    face_present = lm is not None
    if not face_present:
        agg.update({k: 0.0 for k in FRAME_FEATURE_ORDER}, face_present=False)
    else:
        agg.update(per_frame_features(lm, w, h), face_present=True)

    tiredness = boredom = None
    if agg.ready():
        t_model, b_model = get_models()
        if t_model is not None:
            seq = agg.window_sequence()[None, ...]
            tiredness = float(t_model.predict(seq, verbose=0).ravel()[0])
        if b_model is not None and face_present:
            face = preprocess_face(frame)[None, ...]
            feats = agg.window_features()[None, ...]
            boredom = float(b_model.predict(
                {"image": face, "features": feats}, verbose=0).ravel()[0])

    server_ms = (time.perf_counter() - t0) * 1000.0

    if req.record:
        _db.log(session_id=req.session_id, tiredness=tiredness,
                boredom=boredom, face_present=face_present,
                server_ms=server_ms, self_report=req.self_report)

    return InferResponse(
        tiredness=tiredness, boredom=boredom, face_present=face_present,
        window_ready=agg.ready(), server_ms=round(server_ms, 2))
