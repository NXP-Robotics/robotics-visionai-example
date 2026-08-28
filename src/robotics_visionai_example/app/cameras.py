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

import dataclasses
import logging
import os
import re
import subprocess

logger = logging.getLogger("robotics_visionai_example.app.cameras")

# Must be set before invoking libcamerasrc or cam: otherwise libcamera may
# pick the imx8-isi pipeline handler instead of nxp/neo, which then fails to
# negotiate at all.
LIBCAMERA_PIPELINES_MATCH_LIST = "nxp/neo,imx8-isi,uvcvideo"


@dataclasses.dataclass(frozen=True)
class CameraProfile:
    name: str                    # e.g. "imx708", also the `cam -l` sensor token
    sensor_caps: str             # caps string placed right after libcamerasrc


CAMERA_PROFILES = {
    "imx219": CameraProfile(
        name="imx219",
        sensor_caps="video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1",
    ),
    "imx708": CameraProfile(
        name="imx708",
        sensor_caps="video/x-raw,format=NV12",
    ),
    "os08a20": CameraProfile(
        name="os08a20",
        sensor_caps="video/x-raw,format=NV12",
    ),
}

CAMERA_TYPES = tuple(CAMERA_PROFILES.keys())

# Matches lines like:
#   1: External camera 'imx708' (/base/soc/.../imx708@1a)
_CAM_LIST_RE = re.compile(r"^\s*(\d+):\s*.*?'([^']+)'\s*\(([^)]+)\)\s*$")


def list_cameras():
    """Run `cam -l` and return [(index, sensor_name, camera_id), ...].

    The order matches libcamera's, which is also the order libcamerasrc
    picks from when no camera-name is given. Returns an empty list if `cam`
    is missing, fails, times out or reports nothing parseable; callers decide
    how to treat "nothing found".
    """
    env = dict(os.environ)
    env.setdefault("LIBCAMERA_PIPELINES_MATCH_LIST", LIBCAMERA_PIPELINES_MATCH_LIST)

    try:
        result = subprocess.run(
            ["cam", "-l"],
            capture_output=True, text=True, timeout=10, env=env,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("`cam -l` unavailable: %s", exc)
        return []

    if result.returncode != 0:
        logger.warning("`cam -l` exited %s: %s", result.returncode,
                        (result.stderr or result.stdout).strip())
        return []

    cameras = []
    for line in result.stdout.splitlines():
        m = _CAM_LIST_RE.match(line)
        if m:
            index, sensor_name, camera_id = m.groups()
            cameras.append((int(index), sensor_name, camera_id))
    return cameras


def short_id(camera_id):
    """Shorten a libcamera camera id to its sensor@address tail.

    libcamera ids are full device-tree paths (~60 chars), e.g.
    "/base/soc/i2c@42530000/i2c-mux.../imx708@1a" -> "imx708@1a". That tail
    is what distinguishes one attached sensor from another.
    """
    if not camera_id:
        return ""
    return camera_id.rstrip("/").rsplit("/", 1)[-1]


def describe_cameras():
    """List the attached, supported cameras as UI/API-friendly dicts.

    Returns [{index, camera_type, camera_id, label}, ...] for every `cam -l`
    entry whose sensor has a CameraProfile. Sensors without a profile are
    skipped.
    """
    described = []
    for index, sensor_name, camera_id in list_cameras():
        if sensor_name not in CAMERA_PROFILES:
            logger.debug("ignoring unsupported camera '%s' (%s)",
                         sensor_name, camera_id)
            continue
        described.append({
            "index": index,
            "camera_type": sensor_name,
            "camera_id": camera_id,
            "label": f"{sensor_name} ({short_id(camera_id)})",
        })
    return described


def resolve_camera(camera_id=None, index=None):
    """Resolve a (CameraProfile, camera_id, camera_index) triple from `cam -l`.

    At most one selector is used, in this precedence:
      camera_id: the libcamera camera id, i.e. the exact value given to
                 libcamerasrc's camera-name=. The runtime camera selector
                 sends this, since it names one specific sensor.
      index:     1-based camera index from `cam -l`.
      neither:   use the first entry found.

    Raises RuntimeError if nothing is reported, the requested camera does not
    exist, or the sensor is not a known profile. There is deliberately no
    fallback: guessing wrong fails downstream instead of at startup.
    """
    cameras = list_cameras()
    if not cameras:
        raise RuntimeError(
            "No cameras detected via `cam -l`. Pass --camera-type explicitly "
            f"(one of: {', '.join(CAMERA_TYPES)}) or check that a camera is "
            "attached and LIBCAMERA_PIPELINES_MATCH_LIST is set."
        )

    if camera_id is not None:
        matches = [c for c in cameras if c[2] == camera_id]
        if not matches:
            available = ", ".join(short_id(c[2]) for c in cameras)
            raise RuntimeError(
                f"camera '{short_id(camera_id)}' not found in `cam -l` output "
                f"(available: {available})."
            )
        chosen = matches[0]
    elif index is not None:
        matches = [c for c in cameras if c[0] == index]
        if not matches:
            available = ", ".join(str(c[0]) for c in cameras)
            raise RuntimeError(
                f"--camera-index {index} not found in `cam -l` output "
                f"(available indices: {available})."
            )
        chosen = matches[0]
    else:
        chosen = cameras[0]

    camera_index, sensor_name, resolved_id = chosen
    profile = CAMERA_PROFILES.get(sensor_name)
    if profile is None:
        raise RuntimeError(
            f"Detected camera '{sensor_name}' is not a supported camera type "
            f"(known: {', '.join(CAMERA_TYPES)}). Pass --camera-type explicitly."
        )
    return profile, resolved_id, camera_index


def detect_camera_type(index=None):
    """Startup auto-detection: resolve by `cam -l` index, or take the first.

    Thin wrapper kept for the --camera-type auto / --camera-index startup
    path. resolve_camera() is the general form, also used by the runtime
    camera switch (which selects by camera id instead).
    """
    return resolve_camera(index=index)


def get_profile(name):
    """Look up a CameraProfile by explicit --camera-type name."""
    profile = CAMERA_PROFILES.get(name)
    if profile is None:
        raise ValueError(
            f"Unknown camera type '{name}' (known: {', '.join(CAMERA_TYPES)})"
        )
    return profile


def find_camera_by_type(name):
    """Return the (index, camera_id) of the first `cam -l` entry matching a
    sensor name, or (None, None) if there is no match.

    Non-fatal, unlike detect_camera_type(): an explicit --camera-type already
    determines how to build the pipeline, so an unreadable `cam -l` just
    leaves libcamerasrc to pick its own default camera.
    """
    for index, sensor_name, camera_id in list_cameras():
        if sensor_name == name:
            return index, camera_id
    return None, None
