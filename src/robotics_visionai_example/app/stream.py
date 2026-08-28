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

import logging
import os
import queue
import threading
import time

import cv2
import gi
import numpy as np

from robotics_visionai_example.app import cameras
from robotics_visionai_example.app.state import StreamState

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib  # noqa: E402

logger = logging.getLogger("robotics_visionai_example.app.stream")

JPEG_QUALITY = 85


def correct_exposure(bgr: np.ndarray) -> np.ndarray:
    """Stretch contrast only when the frame is clearly under/overexposed.

    A correctly exposed sensor frame is left untouched, since stretching a
    noisy image only amplifies the noise. Only the extremes are corrected:
    mean < 40 (too dark) or mean > 210 (overexposed).
    """
    mean_val = float(np.mean(bgr))

    if mean_val < 40 or mean_val > 210:
        p_lo = float(np.percentile(bgr, 2))
        p_hi = float(np.percentile(bgr, 98))
        if p_hi > p_lo + 10:
            scale = 255.0 / (p_hi - p_lo)
            bgr = np.clip((bgr.astype(np.float32) - p_lo) * scale,
                          0, 255).astype(np.uint8)

    return bgr


def _put_drop_oldest(q, item):
    """Enqueue item, dropping the stalest entry if the queue is full.

    Keeps the pipeline working on the freshest frame available instead of
    blocking capture behind a slow stage.
    """
    try:
        q.put_nowait(item)
        return
    except queue.Full:
        pass
    try:
        q.get_nowait()
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        pass


class VisionStream:
    """GStreamer capture pipeline plus engine dispatch for the web viewer.

    Owns the pipeline (CAMERA or TEST source), the StreamState, and the
    optional 3-stage threaded worker pipeline that overlaps preprocess,
    inference and postprocess so throughput is bounded by the slowest single
    stage rather than their sum.

    engine: an InferenceEngine instance; VisionStream calls engine.init().
    threaded_pipeline: run the 3-stage pipeline instead of inferring inline in
        the capture callback. Defaults to True only for ARA, whose blocking
        hardware round-trip is what makes the overlap worthwhile.
    """
    def __init__(self, engine, source="CAMERA", width=640, height=480,
                 camera_type="auto", camera_index=None, threaded_pipeline=None):
        self.engine = engine
        self.backend = engine.NAME
        self.source = source
        self.video_width = width
        self.video_height = height

        # Which sensor profile to build the pipeline for. "auto" asks `cam -l`
        # what is attached; an explicit name skips detection. Only meaningful
        # for the CAMERA source, since TEST never touches libcamerasrc.
        self.camera_type = camera_type
        self.camera_index = camera_index
        self.camera_profile = None
        self.camera_id = None
        if self.source == "CAMERA":
            self._resolve_camera(camera_type, camera_index)

        self.threaded_pipeline = (
            threaded_pipeline if threaded_pipeline is not None
            else (self.backend == "ARA")
        )

        # Every engine reports the model it loaded, so the viewer can show and
        # switch it regardless of backend.
        active_model = getattr(engine, "model_name", None)
        self.state = StreamState(
            backend=self.backend,

            active_model=active_model,
            camera_type=(self.camera_profile.name if self.camera_profile else None),
            # TEST has no sensor to name, so report the element doing the work
            camera_id=(self.camera_id if self.source == "CAMERA" else "videotestsrc"),
            camera_index=self.camera_index,
        )

        self.mainloop = None
        self.pipeline = None
        self.running = False

        # 3-stage pipeline queues. Each holds at most 2 in-flight frames and
        # drops the oldest when full, so a slow stage never stalls capture.
        self._pipe_pre_q = queue.Queue(maxsize=2)
        self._pipe_inf_q = queue.Queue(maxsize=2)
        self._pipe_post_q = queue.Queue(maxsize=2)
        self._pipe_threads = []
        self._pipe_stop = threading.Event()

        # Serializes overlapping POST /api/model requests against each other;
        # switch_model() already parks the worker threads itself.
        self._model_switch_lock = threading.Lock()

        self._camera_switch_lock = threading.Lock()

        self._pipeline_generation = 0
        self._bus = None

        self.engine.init()
        logger.info("%s engine ready (model=%s, threaded=%s)", self.backend,
                    getattr(engine, "model_name", "?"), self.threaded_pipeline)

    def _resolve_camera(self, camera_type, camera_index):
        """Fill in camera_profile / camera_id / camera_index for the CAMERA source."""
        if camera_type == "auto":
            self.camera_profile, self.camera_id, self.camera_index = (
                cameras.detect_camera_type(index=camera_index))
            logger.info("Camera auto-detected: %s #%s (%s)", self.camera_profile.name,
                        self.camera_index, self.camera_id)
            return

        self.camera_profile = cameras.get_profile(camera_type)
        if camera_index is not None:
            _, self.camera_id, self.camera_index = (
                cameras.detect_camera_type(index=camera_index))
        else:
            self.camera_index, self.camera_id = cameras.find_camera_by_type(camera_type)
        logger.info("Camera type: %s #%s (%s) (explicit --camera-type)",
                    self.camera_profile.name, self.camera_index, self.camera_id)

    # ------------------------------------------------------------------
    # GStreamer pipeline
    # ------------------------------------------------------------------

    def _build_pipeline_string(self) -> str:
        if self.source == "CAMERA":
            camera_name_clause = (
                f'camera-name="{self.camera_id}" ' if self.camera_id else ""
            )
            return (
                f"libcamerasrc {camera_name_clause}! "
                f"{self.camera_profile.sensor_caps} ! "
                "imxvideoconvert_g2d ! "
                f"video/x-raw,format=BGRx,width={self.video_width},"
                f"height={self.video_height} ! "
                "appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false"
            )

        if self.source == "TEST":
            logger.warning("Using videotestsrc (no real camera)")
            return (
                f"videotestsrc pattern=ball ! "
                f"video/x-raw,width={self.video_width},"
                f"height={self.video_height},format=RGB ! "
                f"videoconvert ! video/x-raw,format=BGR ! "
                f"appsink name=appsink emit-signals=true max-buffers=2 drop=true"
            )

        raise ValueError("Only CAMERA or TEST source is supported")

    def _on_sample(self, appsink):
        """Appsink callback for both CAMERA (BGRx) and TEST (BGR) sources."""
        t_wall_start = time.perf_counter()

        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        s = sample.get_caps().get_structure(0)
        h = s.get_value("height")
        w = s.get_value("width")
        fmt = s.get_value("format")

        t0 = time.perf_counter()
        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            if fmt == "BGRx":
                bgrx = np.frombuffer(map_info.data, dtype=np.uint8).reshape(h, w, 4)
                bgr = bgrx[:, :, :3].copy()
            else:
                bgr = np.ndarray((h, w, 3), dtype=np.uint8, buffer=map_info.data).copy()
        finally:
            buf.unmap(map_info)
        t_capture = (time.perf_counter() - t0) * 1000.0

        if self.threaded_pipeline:
            _put_drop_oldest(self._pipe_pre_q, (bgr, t_capture))
        else:
            result, timings = self.engine.infer_frame_timed(bgr)
            self.state.update_result(self.engine.task, result)
            total_ms = (time.perf_counter() - t_wall_start) * 1000.0
            self._publish(bgr, result, timings, t_capture, total_ms=total_ms)

        return Gst.FlowReturn.OK

    def _publish(self, bgr, result, timings, t_capture, total_ms=None):
        """Draw, JPEG-encode and publish one finished frame, then record stats.

        total_ms: measured wall time for the frame; when None it is derived
        from the FPS, which is what the pipelined path reports (its stages
        overlap, so per-frame wall time is not the sum of the stages).
        """
        t0 = time.perf_counter()
        corrected = correct_exposure(bgr)
        fps, _ = self.state.tick_fps()
        annotated = self.engine.task.draw(corrected, result, fps=fps,
                                          backend=self.backend)
        t_draw = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ok, jpeg_buf = cv2.imencode(".jpg", annotated,
                                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        t_encode = (time.perf_counter() - t0) * 1000.0
        if ok:
            self.state.push_frame(jpeg_buf.tobytes())

        if total_ms is None:
            total_ms = 1000.0 / fps if fps > 0 else 0.0

        with self.state.lock:
            self.state.fps = fps
            self.state.invoke_ms = timings.get("infer_ms", 0.0)
            self.state.frame_count += 1

        merged = dict(timings)
        merged.update({"capture_ms": t_capture, "draw_ms": t_draw,
                       "encode_ms": t_encode, "total_ms": total_ms})
        self.state.update_timings(merged)

    # ------------------------------------------------------------------
    # 3-stage threaded pipeline
    # ------------------------------------------------------------------

    def _pipe_preprocess_worker(self):
        """Stage 1: BGR frame -> engine-native prepared input."""
        while not self._pipe_stop.is_set():
            try:
                bgr, t_capture = self._pipe_pre_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                prepared, pre_t = self.engine.preprocess(bgr)
            except Exception as exc:
                logger.warning("preprocess stage failed: %s", exc)
                continue
            _put_drop_oldest(self._pipe_inf_q, (prepared, bgr, t_capture, pre_t))

    def _pipe_infer_worker(self):
        """Stage 2: prepared input -> raw outputs.

        The only thread that calls engine.infer(), so the hardware needs no
        locking.
        """
        while not self._pipe_stop.is_set():
            try:
                prepared, bgr, t_capture, pre_t = self._pipe_inf_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                raw, inf_t = self.engine.infer(prepared)
            except Exception as exc:
                logger.warning("infer stage failed: %s", exc)
                continue
            timings = dict(pre_t)
            timings.update(inf_t)
            _put_drop_oldest(self._pipe_post_q, (raw, bgr, t_capture, timings))

    def _pipe_postprocess_worker(self):
        """Stage 3: raw outputs -> Result, then draw, encode and publish."""
        while not self._pipe_stop.is_set():
            try:
                raw, bgr, t_capture, timings = self._pipe_post_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result, post_t = self.engine.postprocess(raw, bgr.shape[:2])
            except Exception as exc:
                logger.warning("postprocess stage failed: %s", exc)
                continue
            timings.update(post_t)
            self.state.update_result(self.engine.task, result)
            self._publish(bgr, result, timings, t_capture)

    def _start_pipeline_threads(self):
        self._pipe_stop.clear()
        self._pipe_threads = [
            threading.Thread(target=self._pipe_preprocess_worker,
                             name="vision-preprocess", daemon=True),
            threading.Thread(target=self._pipe_infer_worker,
                             name="vision-infer", daemon=True),
            threading.Thread(target=self._pipe_postprocess_worker,
                             name="vision-postprocess", daemon=True),
        ]
        for t in self._pipe_threads:
            t.start()
        logger.info("3-stage pipeline started (preprocess/infer/postprocess)")

    def _stop_pipeline_threads(self):
        self._pipe_stop.set()
        for t in self._pipe_threads:
            t.join(timeout=2.0)
        self._pipe_threads = []

    def _drain_pipeline_queues(self):
        """Drop in-flight frames. Only race-free while the workers are stopped."""
        for q in (self._pipe_pre_q, self._pipe_inf_q, self._pipe_post_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def switch_model(self, model_name: str):
        """Hot-swap the engine to a different model.

        Parks the worker threads, drains in-flight tensors, reloads, then
        resumes. Capture and the MJPEG stream keep running, so only the
        inference results freeze during the reload. Raises on failure; the
        engine's reload() is expected to roll back.
        """
        with self._model_switch_lock:
            if model_name == getattr(self.engine, "model_name", None):
                return

            with self.state.lock:
                self.state.switch_state = "switching"
                self.state.switch_error = None

            pipeline_was_running = bool(self._pipe_threads)
            if pipeline_was_running:
                self._stop_pipeline_threads()
            self._drain_pipeline_queues()

            try:
                self.engine.reload(model_name)
                with self.state.lock:
                    # Read back from the engine: it may normalise the requested
                    # name (e.g. resolving a per-SoC variant) to what it loaded.
                    self.state.active_model = getattr(
                        self.engine, "model_name", model_name)

                    self.state.result = None
                    self.state.switch_state = "idle"
                    self.state.switch_error = None
                logger.info("model switched to: %s", model_name)
            except Exception as exc:
                with self.state.lock:
                    self.state.switch_state = "error"
                    self.state.switch_error = str(exc)
                logger.error("model switch to '%s' failed: %s", model_name, exc)
                raise
            finally:
                if pipeline_was_running:
                    self._start_pipeline_threads()

    def switch_camera(self, camera_id: str):
        """Point the capture pipeline at a different attached camera.

        Raises RuntimeError if the camera is unknown, the source is not
        CAMERA, or the new pipeline fails to start (in which case the
        previous camera is restored so the viewer is not left dead).
        """
        if self.source != "CAMERA":
            raise RuntimeError(
                f"camera switching is only available for the CAMERA source "
                f"(current source: {self.source})"
            )

        with self._camera_switch_lock:
            if camera_id == self.camera_id:
                return  # already streaming this camera, nothing to do

            # Resolve before touching the pipeline: a bad request should be
            # rejected with the current stream still running.
            profile, resolved_id, camera_index = cameras.resolve_camera(
                camera_id=camera_id)

            with self.state.lock:
                self.state.camera_switch_state = "switching"
                self.state.camera_switch_error = None

            if self.mainloop is None or not self.mainloop.is_running():
                # No main loop yet (run() has not been reached): just record
                # the selection so the first _start_capture() picks it up.
                self._apply_camera(profile, resolved_id, camera_index)
                with self.state.lock:
                    self.state.camera_switch_state = "idle"
                return

            done = threading.Event()
            outcome = {}

            def on_main_loop():
                try:
                    self._switch_camera_on_main_loop(
                        profile, resolved_id, camera_index)
                except Exception as exc:  # noqa: BLE001 - reported to caller
                    outcome["error"] = exc
                finally:
                    done.set()
                return GLib.SOURCE_REMOVE

            GLib.idle_add(on_main_loop)

            if not done.wait(timeout=10.0):
                msg = "camera switch timed out waiting for the capture pipeline"
                with self.state.lock:
                    self.state.camera_switch_state = "error"
                    self.state.camera_switch_error = msg
                raise RuntimeError(msg)

            if "error" in outcome:
                exc = outcome["error"]
                with self.state.lock:
                    self.state.camera_switch_state = "error"
                    self.state.camera_switch_error = str(exc)
                logger.error("camera switch to '%s' failed: %s",
                             cameras.short_id(resolved_id), exc)
                raise RuntimeError(str(exc)) from exc

            with self.state.lock:
                self.state.camera_switch_state = "idle"
                self.state.camera_switch_error = None

    def _apply_camera(self, profile, camera_id, camera_index):
        """Record the selected camera on self and in the published state."""
        self.camera_profile = profile
        self.camera_type = profile.name
        self.camera_id = camera_id
        self.camera_index = camera_index
        self.state.set_camera(profile.name, camera_id, camera_index)

    def _switch_camera_on_main_loop(self, profile, camera_id, camera_index):
        """Rebuild the capture pipeline for a new camera. Main-loop thread only.

        Restores the previous camera if the new pipeline cannot be brought
        up, so a failed switch degrades to "still streaming the old sensor"
        rather than a dead viewer.
        """
        previous = (self.camera_profile, self.camera_id, self.camera_index)
        logger.info("switching camera: %s -> %s (%s)",
                    cameras.short_id(self.camera_id or ""),
                    cameras.short_id(camera_id), profile.name)

        workers_were_running = bool(self._pipe_threads)
        if workers_were_running:
            self._stop_pipeline_threads()
        self._stop_capture()
        self._drain_pipeline_queues()
        self._drain_frame_queue()

        self._apply_camera(profile, camera_id, camera_index)
        try:
            self._start_capture()
        except Exception as exc:
            logger.error("failed to start capture for %s: %s",
                         cameras.short_id(camera_id), exc)
            self._apply_camera(*previous)
            try:
                self._start_capture()
                logger.info("restored previous camera %s",
                            cameras.short_id(previous[1] or ""))
            except Exception as rollback_exc:
                logger.error("could not restore previous camera: %s",
                             rollback_exc)
            raise
        finally:
            if workers_were_running:
                self._start_pipeline_threads()

        logger.info("camera switched to: %s (%s)",
                    cameras.short_id(camera_id), profile.name)

    def _drain_frame_queue(self):
        """Drop already-encoded JPEGs so the viewer does not show stale frames
        from the previous camera after a switch.
        """
        while True:
            try:
                self.state.frame_queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Run / stop
    # ------------------------------------------------------------------

    def _start_capture(self):
        """Build, wire up and start the GStreamer capture pipeline.

        Shared by run() and the camera switch so both paths build the
        pipeline, connect the appsink callback and install the bus watch in
        exactly the same way. Raises RuntimeError if the pipeline refuses to
        reach PLAYING (e.g. caps that the attached sensor cannot negotiate).
        """
        pipe_str = self._build_pipeline_string()
        logger.info("GStreamer pipeline: %s", pipe_str)
        self.pipeline = Gst.parse_launch(pipe_str)
        self._pipeline_generation += 1
        generation = self._pipeline_generation

        self.pipeline.get_by_name("appsink").connect("new-sample", self._on_sample)

        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message, generation)
        self._bus = bus

        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._stop_capture()
            raise RuntimeError(
                "GStreamer pipeline failed to start "
                f"(camera: {cameras.short_id(self.camera_id or '')})"
            )

        self.running = True
        logger.info("Pipeline playing")

    def _stop_capture(self):
        """Tear the current capture pipeline down. Main-loop thread only.

        Removes the bus watch BEFORE going to NULL: a pipeline being shut
        down can post an ERROR, and the bus handler quits the main loop on
        ERROR, which during a camera switch would take the whole app down.
        """
        if self.pipeline is None:
            return
        if self._bus is not None:
            self._bus.remove_signal_watch()
            self._bus = None
        self.pipeline.set_state(Gst.State.NULL)
        self.pipeline = None
        self._appsink = None
        self.running = False

    def run(self):
        os.environ.setdefault("LIBCAMERA_PIPELINES_MATCH_LIST",
                              cameras.LIBCAMERA_PIPELINES_MATCH_LIST)

        Gst.init([])

        self._start_capture()

        if self.threaded_pipeline:
            self._start_pipeline_threads()

        self.mainloop = GLib.MainLoop()
        self.mainloop.run()

        self._stop_capture()

    def stop(self):
        self.running = False
        if self.mainloop:
            self.mainloop.quit()
        if self.threaded_pipeline:
            self._stop_pipeline_threads()
        self.engine.close()

    def _on_bus_message(self, bus, message, generation=None):
        if generation is not None and generation != self._pipeline_generation:
            return

        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("EOS received")
            self.mainloop.quit()
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            logger.error("GStreamer error: %s [%s]", err.message, dbg)
            self.mainloop.quit()
        elif t == Gst.MessageType.WARNING:
            err, dbg = message.parse_warning()
            logger.warning("GStreamer warning: %s [%s]", err.message, dbg)
