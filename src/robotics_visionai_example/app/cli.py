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

import argparse
import logging
import os
import signal
import threading

import uvicorn

from robotics_visionai_example.app import server as server_mod
from robotics_visionai_example.app.cameras import CAMERA_TYPES
from robotics_visionai_example.common.docs import readme_hint
from robotics_visionai_example.engines import create_engine, engine_class
from robotics_visionai_example.engines import tflite_engine
from robotics_visionai_example.tasks import create_task



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("robotics_visionai_example.app.cli")

NEUTRON_CNN_DIR = os.environ.get("NEUTRON_CNN_DIR", "/usr/share/cnn/neutron")

PREFERRED_MODELS = ("yolov8n", "yolov8", "movenet")


def _default_tflite_model(models_dir, task_name, backend):
    """Pick a default TFLite model for this task from what is installed.

    Nothing about the model set is assumed: the registry is scanned and, of the
    models this backend can actually run, PREFERRED_MODELS decides which to
    favour. Returns a "<category>/<name>" the engine can resolve, and raises
    SystemExit with the installed models listed when there is no candidate.
    """
    engine_cls = engine_class(backend)
    available = engine_cls.list_models(models_dir=models_dir, tasks=(task_name,))


    if not available:
        all_for_task = tflite_engine.list_models(models_dir, (task_name,))
        logger.error("No '%s' model available for the %s backend under %s",
                     task_name, backend, models_dir)
        if all_for_task:
            # Models exist but this backend cannot load them, which for a
            # delegate-less backend means they are all accelerator-compiled.
            logger.error(
                "Installed '%s' models: %s -- these are compiled for an "
                "accelerator (SoC variant tag in the filename), so the %s "
                "backend cannot run them. Try --backend NEUTRON, or pass a "
                "portable model with --model_path <file.tflite>.",
                task_name, ", ".join(all_for_task), backend,
            )
        else:
            logger.error(
                "Models are compiled on a host (e.g. from "
                "nxp-nnstreamer-examples) and copied to the board as "
                "<category>/<name>.tflite under %s. Override the base "
                "directory with NEUTRON_CNN_DIR / --neutron-models-dir.",
                models_dir,
            )
        logger.error("%s", readme_hint("2. Set up the models"))
        raise SystemExit(1)

    for preference in PREFERRED_MODELS:
        for model in available:
            if preference in model.rpartition("/")[2].lower():
                return model
    return available[0]


def resolve_task_name(args):
    """Default --task to match the backend's own default model when omitted:
    "detection" for ARA (--ara-model defaults to YOLOv8), "pose" otherwise.
    """
    if args.task is not None:
        return args.task
    return "detection" if args.backend == "ARA" else "pose"


def build_engine(args):
    """Construct the TaskProcessor + InferenceEngine pair from parsed args."""
    args.task = resolve_task_name(args)
    task = create_task(args.task)

    if args.backend == "ARA":

        return create_engine(
            "ARA", task, model=args.ara_model, models_dir=args.models_dir,
        )

    model_path = args.model_path
    if not model_path:
        model_path = _default_tflite_model(
            args.neutron_models_dir, args.task, args.backend
        )
        logger.info("No --model_path given; using discovered model: %s", model_path)

    engine = create_engine(args.backend, task, model_path=model_path,
                           models_dir=args.neutron_models_dir)
    try:
        engine.init()
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1)
    return engine




def parse_args():
    parser = argparse.ArgumentParser(description="Vision inference -> web stream")
    parser.add_argument("--source", default=os.getenv("SOURCE", "CAMERA"),
                        choices=["CAMERA", "TEST"],
                        help="Input source: CAMERA (v4l2) or TEST (synthetic videotestsrc)")
    parser.add_argument("--backend", default=os.getenv("BACKEND", "CPU"),
                        choices=["CPU", "GPU", "NEUTRON", "ARA"])
    parser.add_argument("--task", default=os.getenv("TASK", None),
                        choices=["pose", "detection"],
                        help="Which TaskProcessor to decode engine outputs with. "
                             "Defaults to 'detection' for --backend ARA and 'pose' "
                             "for every other backend.")
    parser.add_argument("--ara-model", default=os.getenv("ARA_MODEL", "detection/yolov8s"),
                        help="Model to run for the ARA backend, as installed under "
                             "/usr/share/cnn on the board (e.g. detection/yolov8s). "
                             "Override the models base dir with ARA_CNN_DIR / --models-dir.")

    parser.add_argument("--models-dir", default=os.getenv("ARA_CNN_DIR", None),
                        help="Base directory for ARA's <task>/<name>/model.dvm tree "
                             "(default: /usr/share/cnn, or $ARA_CNN_DIR)")
    parser.add_argument("--neutron-models-dir", default=NEUTRON_CNN_DIR,
                        help="Base directory for the NEUTRON backend's "
                             "<task>/<name>.tflite tree (default: /usr/share/cnn/neutron, "
                             "or $NEUTRON_CNN_DIR)")

    parser.add_argument("--camera-type", "-t",
                        default=os.getenv("CAMERA_TYPE", "auto"),
                        choices=list(CAMERA_TYPES) + ["auto"],
                        help="Camera sensor to build the GStreamer pipeline for. Each "
                             "sensor negotiates different libcamerasrc caps. 'auto' "
                             "(default) asks `cam -l` which camera is attached and "
                             "fails with a clear error if it cannot tell.")
    parser.add_argument("--camera-index", type=int,
                        default=(int(os.environ["CAMERA_INDEX"])
                                 if os.getenv("CAMERA_INDEX") else None),
                        help="Which camera to open when several are attached, using the "
                             "1-based index from `cam -l`. Omitted: let libcamerasrc "
                             "pick its default (the first camera).")
    parser.add_argument("--model_path", "-p", default="",
                        help="Model for the NEUTRON/GPU/CPU backends: a .tflite "
                             "path, a '<category>/<name>' under the TFLite models "
                             "tree (e.g. object-detection/yolov8n-det-int8), or a "
                             "bare model name. The per-SoC variant suffix may be "
                             "omitted -- the right one for this board is selected "
                             "automatically. Empty: pick one of the installed "
                             "models for --task.")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--port", type=int, default=5010)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--doctor", action="store_true",
                        help="Check the runtime environment (Python deps, "
                             "GStreamer plugins, cameras, ARA-240 SDK, installed "
                             "models), print how to fix anything missing, and "
                             "exit without starting the stream.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.doctor:
        from robotics_visionai_example.app import doctor
        raise SystemExit(doctor.main(args.models_dir, args.neutron_models_dir))

    # Before importing app.stream, so a missing model (the more common
    # failure) reports its actionable error instead of an ImportError.
    engine = build_engine(args)

    try:
        from robotics_visionai_example.app.stream import VisionStream
    except ImportError as exc:
        logger.error("Cannot load the capture pipeline: %s", exc)
        logger.error(
            "This usually means the virtualenv was created without access to "
            "the system packages that provide PyGObject and the GStreamer "
            "bindings. Rebuild it with 'uv venv --system-site-packages', or "
            "on the board re-run:  sudo /usr/share/robotics-visionai-example/install.sh"
        )
        logger.error("Run '%s --doctor' for a full environment report.",
                     "robotics-visionai-example")
        raise SystemExit(1)

    stream = VisionStream(
        engine, source=args.source,
        width=args.width, height=args.height,
        camera_type=args.camera_type, camera_index=args.camera_index,
    )
    # VisionStream resolves the sensor profile and libcamera id, so the real
    # capture source can only be logged once it exists.
    if args.source == "CAMERA":

        logger.info("Camera: %s #%s (%s)", stream.camera_profile.name,
                    stream.camera_index, stream.camera_id)

    server_mod.set_stream_instance(stream)

    # uvicorn runs in a daemon thread so the GLib MainLoop driving the
    # GStreamer pipeline keeps the main thread, where the signal handlers
    # below must live.
    uvicorn_config = uvicorn.Config(
        server_mod.app, host="0.0.0.0", port=args.port,
        log_level="warning", access_log=False,
    )
    uvicorn_thread = threading.Thread(
        target=uvicorn.Server(uvicorn_config).run,
        daemon=True,
        name="uvicorn",
    )
    uvicorn_thread.start()
    logger.info("Web viewer: http://0.0.0.0:%s", args.port)

    signal.signal(signal.SIGINT, lambda *_: stream.stop())
    signal.signal(signal.SIGTERM, lambda *_: stream.stop())

    stream.run()
    logger.info("Stopped.")


if __name__ == "__main__":
    main()
