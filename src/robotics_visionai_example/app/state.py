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

import queue
import threading
import time


# Per-frame timing breakdown keys (exponential moving average, alpha=0.2).
TIMING_KEYS = (
    "capture_ms",     # buffer map + copy
    "preprocess_ms",  # engine preprocess (resize/scale/CHW/quantize)
    "quantize_ms",    # quantize float32 -> int8 (ARA only; 0 elsewhere)
    "infer_ms",       # hardware/delegate inference
    "dequantize_ms",  # dequantize integer outputs -> float32 (0 when float)

    "nms_ms",         # task decode (NMS for detection, rescale for pose)
    "draw_ms",        # task.draw() + exposure correction
    "encode_ms",      # JPEG encode
    "total_ms",       # wall time for the whole frame
)


class StreamState:
    """Thread-safe container for the current inference result plus stats,
    consumed by app/server.py's routes.
    """
    def __init__(self, backend="CPU", active_model=None,
                 camera_type=None, camera_id=None, camera_index=None):
        self.lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=2)

        self.task = None          # active TaskProcessor instance
        self.result = None        # its most recent Result

        self.fps = 0.0
        self.invoke_ms = 0.0
        self.frame_count = 0
        self.backend = backend

        # What the capture pipeline is reading from. camera_id is the
        # libcamera identifier passed to libcamerasrc's camera-name=,
        # camera_index its 1-based position in `cam -l`.
        self.camera_type = camera_type   # e.g. "imx708" (None for TEST)
        self.camera_id = camera_id
        self.camera_index = camera_index

        self.active_model = active_model

        self.switch_state = "idle"   # "idle" | "switching" | "error"

        self.switch_error = None

        # Runtime camera switch status.
        self.camera_switch_state = "idle"   # "idle" | "switching" | "error"
        self.camera_switch_error = None

        self.timings = {k: 0.0 for k in TIMING_KEYS}

        self._last_tick = time.monotonic()

    def tick_fps(self):
        """Call once per completed frame; updates and returns (fps, dt_ms)."""
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now
        fps = 1.0 / dt if dt > 0 else 0.0
        return fps, dt * 1000.0

    def update_result(self, task, result):
        with self.lock:
            self.task = task
            self.result = result

    def set_camera(self, camera_type, camera_id, camera_index):
        """Publish the sensor the capture pipeline is now reading from."""
        with self.lock:
            self.camera_type = camera_type
            self.camera_id = camera_id
            self.camera_index = camera_index
            self.result = None

    def update_timings(self, new_timings, alpha=0.2):
        """Merge a partial timings dict into the EMA-smoothed self.timings."""
        with self.lock:
            old = self.timings
            merged = dict(old)
            for k in TIMING_KEYS:
                if k in new_timings:
                    merged[k] = old[k] * (1 - alpha) + new_timings[k] * alpha
            self.timings = merged

    def push_frame(self, jpeg_bytes):
        try:
            self.frame_queue.put_nowait(jpeg_bytes)
        except queue.Full:
            pass

    def mjpeg_generator(self):
        while True:
            try:
                jpeg_bytes = self.frame_queue.get(timeout=5.0)
            except queue.Empty:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + jpeg_bytes + b"\r\n")

    def snapshot(self):
        """Return everything /status needs as a dict, read under the lock in
        one shot so the JSON response is self-consistent.
        """
        with self.lock:
            return {
                "task": self.task,
                "result": self.result,
                "fps": self.fps,
                "invoke_ms": self.invoke_ms,
                "frame_count": self.frame_count,
                "backend": self.backend,
                "camera_type": self.camera_type,
                "camera_id": self.camera_id,
                "camera_index": self.camera_index,
                "active_model": self.active_model,
                "switch_state": self.switch_state,
                "switch_error": self.switch_error,
                "camera_switch_state": self.camera_switch_state,
                "camera_switch_error": self.camera_switch_error,
                "timings": dict(self.timings),
            }
