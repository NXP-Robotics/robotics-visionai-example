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

import importlib.util
import os
import shutil
import subprocess

from robotics_visionai_example.common.docs import readme_hint, readme_path

OK = "OK"
WARN = "WARN"
FAIL = "FAIL"


def _check(name, status, detail="", fix=""):
    return {"name": name, "status": status, "detail": detail, "fix": fix}


def _module_available(module_name):
    """True if a module can be imported, checked without importing it.

    find_spec avoids the side effects of a real import, so --doctor stays
    safe to run while another instance holds the camera or accelerator.
    """
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def check_python_deps():
    """Check the imports that pip cannot satisfy inside the venv.

    PyGObject (gi) and the TFLite runtime come from the board's system
    Python, so they are only visible when the venv was created with
    --system-site-packages.
    """
    results = []


    if _module_available("gi"):
        results.append(_check("PyGObject (gi)", OK))
    else:
        results.append(_check(
            "PyGObject (gi)", FAIL,
            "cannot import 'gi' -- the GStreamer capture pipeline needs it",
            "gi comes from the system python3-gi package, not from pip. Recreate "
            "the venv with system packages visible:  uv venv --system-site-packages",
        ))

    if _module_available("tflite_runtime") or _module_available("ai_edge_litert"):
        results.append(_check("TFLite runtime", OK))
    else:
        results.append(_check(
            "TFLite runtime", WARN,
            "no tflite_runtime / ai_edge_litert -- needed by the NEUTRON, GPU "
            "and CPU backends (the ARA backend does not use it)",
            "Installed system-wide on the board image; make sure the venv was "
            "created with  uv venv --system-site-packages",
        ))

    for module_name, label in (("cv2", "OpenCV (cv2)"),
                               ("numpy", "NumPy"),
                               ("fastapi", "FastAPI"),
                               ("uvicorn", "uvicorn")):
        if _module_available(module_name):
            results.append(_check(label, OK))
        else:
            results.append(_check(
                label, FAIL, "cannot import '{}'".format(module_name),
                "Reinstall the application venv:  "
                "sudo /usr/share/robotics-visionai-example/install.sh",
            ))

    return results


def check_gstreamer():
    """Check for the gst plugins the CAMERA source depends on."""
    results = []

    if shutil.which("gst-inspect-1.0") is None:
        return [_check(
            "GStreamer plugins", WARN,
            "gst-inspect-1.0 not found, cannot verify plugins",
            "Install gstreamer1.0-tools",
        )]

    for plugin, why in (("libcamerasrc", "camera capture source"),
                        ("jpegenc", "MJPEG encoding for the web stream")):
        try:
            proc = subprocess.run(
                ["gst-inspect-1.0", plugin],
                capture_output=True, text=True, timeout=20,
            )
            available = proc.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            available = False

        if available:
            results.append(_check("gst plugin '{}'".format(plugin), OK))
        else:
            results.append(_check(
                "gst plugin '{}'".format(plugin), FAIL,
                "not found -- required for {}".format(why),
                "Check the image's GStreamer packages "
                "(packagegroup-fsl-gstreamer1.0-full, libcamera-gst)",
            ))

    return results


def check_cameras():
    """Enumerate cameras the same way the capture pipeline will."""
    if shutil.which("cam") is None:
        return [_check(
            "Camera detection", WARN,
            "the `cam` tool is not installed, so --camera-type auto cannot work",
            "Install libcamera-tools, or always pass --camera-type explicitly",
        )]

    # Imported here, not at module scope, so the other checks still report
    # even if this import is broken.
    from robotics_visionai_example.app import cameras as cameras_mod


    found = cameras_mod.list_cameras()
    if not found:
        return [_check(
            "Camera detection", FAIL,
            "`cam -l` reports no cameras",
            "Check the CSI cable orientation, then run `cam -l` by hand. "
            "LIBCAMERA_PIPELINES_MATCH_LIST must include 'nxp/neo' (the app "
            "sets this itself unless it is already set in the environment).",
        )]

    results = []
    for index, sensor, camera_id in found:
        if sensor in cameras_mod.CAMERA_PROFILES:
            results.append(_check(
                "Camera #{} '{}'".format(index, sensor), OK, camera_id,
            ))
        else:
            results.append(_check(
                "Camera #{} '{}'".format(index, sensor), WARN,
                "detected but not one of the supported profiles ({})".format(
                    ", ".join(cameras_mod.CAMERA_TYPES)),
                "Pass --camera-type explicitly if its caps match a known sensor",
            ))
    return results


def check_ara():
    """Check the ARA-240 SDK, its systemd service and its proxy socket."""
    from robotics_visionai_example.engines import ara_engine

    results = []

    try:
        ara_engine._get_dvapi()
        results.append(_check("ARA-240 SDK (dvapi)", OK))
        sdk_ok = True
    except ImportError as exc:
        results.append(_check(
            "ARA-240 SDK (dvapi)", WARN,
            str(exc).split(". ")[0],
            "Only needed for --backend ARA. Install/initialise the SDK:  "
            "sudo /usr/share/rt-sdk-ara240/install.sh",
        ))
        sdk_ok = False

    # Only probe the service and socket when the SDK is present: on a board
    # with no ARA-240 fitted these would just be noise.
    if sdk_ok:

        if shutil.which("systemctl") is not None:
            try:
                proc = subprocess.run(
                    ["systemctl", "is-active", "rt-sdk-ara2.service"],
                    capture_output=True, text=True, timeout=15,
                )
                state = (proc.stdout or "").strip() or "unknown"
            except (OSError, subprocess.TimeoutExpired):
                state = "unknown"

            if state == "active":
                results.append(_check("rt-sdk-ara2.service", OK, state))
            else:
                results.append(_check(
                    "rt-sdk-ara2.service", WARN, state,
                    "sudo systemctl start rt-sdk-ara2.service  (then check "
                    "'sudo chip_info.sh' reports firmware >= 131072)",
                ))

        sock = ara_engine.ARA_PROXY_SOCK
        if os.path.exists(sock):
            # Existence is not enough: the app has to be able to open it.
            if os.access(sock, os.R_OK | os.W_OK):

                results.append(_check("ARA proxy socket", OK, sock))
            else:
                results.append(_check(
                    "ARA proxy socket", WARN,
                    "{} exists but is not readable/writable by this user".format(sock),
                    "Run the app with sudo, or fix the socket permissions",
                ))
        else:
            results.append(_check(
                "ARA proxy socket", WARN, "{} does not exist".format(sock),
                "Started by rt-sdk-ara2.service; override the path with "
                "ARA_PROXY_SOCK",
            ))

    return results


def check_models(models_dir=None, neutron_models_dir=None):
    """Check both model trees: ARA .dvm models and TFLite .tflite models."""
    from robotics_visionai_example.engines import ara_engine

    results = []

    models_dir = models_dir or ara_engine.ARA_CNN_DIR
    ara_models = ara_engine.list_models(models_dir)
    if ara_models:
        results.append(_check(
            "ARA models (.dvm)", OK,
            "{} found in {}: {}".format(len(ara_models), models_dir,
                                        ", ".join(ara_models)),
        ))
    else:
        results.append(_check(
            "ARA models (.dvm)", WARN,
            "none found under {}".format(models_dir),
            "Check README for steps on obtaining them.",
        ))

    from robotics_visionai_example.engines import tflite_engine

    neutron_models_dir = neutron_models_dir or tflite_engine.NEUTRON_CNN_DIR

    for task_name in sorted(set(tflite_engine.TFLITE_CATEGORY_TASKS.values())):
        label = "TFLite {} models (.tflite)".format(task_name)
        models = tflite_engine.list_models(neutron_models_dir, (task_name,))
        if not models:
            results.append(_check(
                label, WARN,
                "none found under {}".format(neutron_models_dir),
                "Models are compiled on a host (e.g. from "
                "nxp-nnstreamer-examples) and copied to the board as "
                "<category>/<name>.tflite; keep the filenames unchanged so "
                "the variant for this SoC is auto-selected",
            ))
            continue

        portable = tflite_engine.list_models(
            neutron_models_dir, (task_name,), require_portable=True
        )
        accelerated_only = [m for m in models if m not in portable]

        detail = "{} found in {}: {}".format(
            len(models), neutron_models_dir, ", ".join(models))
        if accelerated_only:
            detail += " (needs a delegate, e.g. --backend NEUTRON: {})".format(
                ", ".join(accelerated_only))
        results.append(_check(label, OK, detail))

    return results



def run_checks(models_dir=None, neutron_models_dir=None):
    """Run every check and return a flat list of result dicts."""
    results = []
    results.extend(check_python_deps())
    results.extend(check_gstreamer())
    results.extend(check_cameras())
    results.extend(check_ara())
    results.extend(check_models(models_dir, neutron_models_dir))
    return results


def main(models_dir=None, neutron_models_dir=None):
    """Print a preflight report and return a process exit code.

    Only a FAIL sets exit code 1: a board with no ARA-240 fitted and no
    models downloaded is a valid CPU-backend setup, so WARNs do not fail.
    """
    results = run_checks(models_dir, neutron_models_dir)

    width = max(len(r["name"]) for r in results)
    failures = 0
    warnings = 0

    print("robotics-visionai-example preflight check")
    print("=" * (width + 40))

    for result in results:
        status = result["status"]
        if status == FAIL:
            failures += 1
        elif status == WARN:
            warnings += 1

        line = "[{:4}] {:<{width}}".format(status, result["name"], width=width)
        if result["detail"]:
            line += "  {}".format(result["detail"])
        print(line)
        if result["fix"] and status != OK:
            print("{}   -> fix: {}".format(" " * 7, result["fix"]))

    print("=" * (width + 40))
    print("{} checks: {} ok, {} warning(s), {} failure(s)".format(
        len(results), len(results) - failures - warnings, warnings, failures))

    if failures or warnings:
        print("")
        print(readme_hint())
        if readme_path() is None:
            print("(README not found on this system -- see the project repository)")

    return 1 if failures else 0
