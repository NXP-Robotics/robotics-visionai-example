#!/usr/bin/env python3
#
# Copyright 2026 NXP
#
# NXP Proprietary. This software is owned or controlled by NXP
# and may only be used strictly in accordance with the applicable license terms.
# By expressly accepting such terms or by downloading, installing, activating
# and/or otherwise using the software, you are agreeing that you have read, and
# that you agree to comply with and are bound by, such license terms.
# If you do not agree to be bound by the applicable license terms, then you may
# not retain, install, activate or otherwise use the software.
#

import datetime
import importlib.resources
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from robotics_visionai_example.app import cameras

logger = logging.getLogger("robotics_visionai_example.app.server")


app = FastAPI(title="robotics-visionai-example", docs_url="/docs")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

_INDEX_HTML = (
    importlib.resources.files("robotics_visionai_example.app.ui")
    .joinpath("index.html")
    .read_text()
)

# Set by app/cli.py once the VisionStream instance exists, so the routes
# (running in uvicorn worker threads) can reach the live pipeline.
_stream_instance = None


def set_stream_instance(stream):
    global _stream_instance
    _stream_instance = stream


def _switchable_models():
    """Models the running engine can hot-swap to, for the active task.

    Any engine exposing list_models() participates, so ARA (.dvm) and the
    TFLite backends are treated identically. Returns [] when the engine has no
    model registry.
    """
    if _stream_instance is None:
        return []

    engine = _stream_instance.engine
    lister = getattr(engine, "list_models", None)
    if lister is None:
        return []

    try:
        return lister(models_dir=engine.models_dir, tasks=(engine.task.NAME,))
    except Exception as exc:
        logger.warning("could not list models: %s", exc)
        return []



class ModelSwitchRequest(BaseModel):
    model: str


class CameraSwitchRequest(BaseModel):
    # The libcamera camera id, i.e. exactly what `cam -l` reports and what
    # libcamerasrc's camera-name= is given.
    camera_id: str


@app.get("/")
def index():
    return HTMLResponse(_INDEX_HTML, headers={"Cache-Control": "no-store"})


@app.get("/stream")
def stream():
    if _stream_instance is None:
        raise HTTPException(status_code=503, detail="stream not started")
    return StreamingResponse(
        _stream_instance.state.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/status")
def status():
    if _stream_instance is None:
        raise HTTPException(status_code=503, detail="stream not started")
    snap = _stream_instance.state.snapshot()

    task = snap["task"]
    result = snap["result"]

    det_list = []
    kpt_list = []
    if task is not None and result is not None:
        payload = task.to_json(result)
        if task.NAME == "detection":
            det_list = payload
        elif task.NAME == "pose":
            kpt_list = payload

    return {
        "fps": round(snap["fps"], 1),
        "invoke_ms": round(snap["invoke_ms"], 1),
        "backend": snap["backend"],
        "camera_type": snap["camera_type"],
        "camera_id": snap["camera_id"],
        "camera_index": snap["camera_index"],
        "model": snap["active_model"],
        "ara_model": snap["active_model"],
        "frame_count": snap["frame_count"],

        "keypoints": kpt_list,
        "num_keypoints_detected": sum(1 for k in kpt_list if k["valid"]),
        "detections": det_list,
        "timestamp": datetime.datetime.now().isoformat(),
        "timings": {k: round(v, 2) for k, v in snap["timings"].items()},
        "switch_state": snap["switch_state"],
        "switch_error": snap["switch_error"],
        "camera_switch_state": snap["camera_switch_state"],
        "camera_switch_error": snap["camera_switch_error"],
    }


@app.get("/api/options")
def api_options():
    """List what the UI's selectors can offer: models and attached cameras.

    "models" holds the models the running engine can switch to for its active
    task, which covers every backend with a model registry (ARA and the TFLite
    ones alike) and is empty otherwise. The camera list comes from `cam -l`,
    i.e. libcamera's own view of what is attached, which is what the CAMERA
    pipeline actually selects from.
    """
    return {
        "models": _switchable_models(),
        "detected_cameras": cameras.describe_cameras(),
    }


@app.post("/api/model")
def api_model(req: ModelSwitchRequest):
    """Hot-swap the engine's model without restarting the stream."""
    if _stream_instance is None:
        raise HTTPException(status_code=503, detail="stream not started")

    allowed = _switchable_models()
    if not allowed:
        raise HTTPException(
            status_code=400,
            detail=(f"the {_stream_instance.backend} backend has no switchable "
                    "models for the active task"),
        )

    if req.model not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(f"model '{req.model}' is not a supported model. "
                    f"Available: {', '.join(allowed)}"),
        )

    try:
        _stream_instance.switch_model(req.model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"ok": True, "model": req.model, "ara_model": req.model}



@app.post("/api/camera")
def api_camera(req: CameraSwitchRequest):
    """Switch which attached camera the capture pipeline reads from.

    The sensor type is resolved from `cam -l`, so the caller only names the
    camera and the correct per-sensor pipeline is selected automatically.
    Rebuilds the GStreamer pipeline; the engine and active model are left
    untouched.
    """
    if _stream_instance is None:
        raise HTTPException(status_code=503, detail="stream not started")

    if _stream_instance.source != "CAMERA":
        raise HTTPException(
            status_code=409,
            detail=("camera switching is only available with --source CAMERA "
                    f"(current source: {_stream_instance.source})"),
        )

    allowed = {c["camera_id"] for c in cameras.describe_cameras()}
    if req.camera_id not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(f"camera '{cameras.short_id(req.camera_id)}' is not an "
                    "available camera. Available: "
                    f"{', '.join(sorted(cameras.short_id(c) for c in allowed)) or '(none found)'}"),
        )

    try:
        _stream_instance.switch_camera(req.camera_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "ok": True,
        "camera_id": _stream_instance.camera_id,
        "camera_type": _stream_instance.camera_type,
        "camera_index": _stream_instance.camera_index,
    }
