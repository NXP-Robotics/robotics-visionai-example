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

import importlib.resources
import os

SYSTEM_DOC_PATHS = (
    "/usr/share/doc/robotics-visionai-example/README.md",
    "/usr/share/robotics-visionai-example/README.md",
)


def _packaged_readme():
    """Return the README shipped inside the installed package, or None.

    """
    try:
        resource = importlib.resources.files("robotics_visionai_example").joinpath(
            "README.md"
        )
        if resource.is_file():
            return str(resource)
    except Exception:
        pass
    return None


def _checkout_readme():
    """Return the README at the git checkout root (three levels up), or None."""
    here = os.path.dirname(os.path.abspath(__file__))

    candidate = os.path.join(here, "..", "..", "..", "README.md")
    candidate = os.path.normpath(candidate)
    if os.path.isfile(candidate):
        return candidate
    return None


def readme_path():
    """Return a path to the README, or None if it cannot be found.

    Resolution order is packaged resource -> installed system doc -> source
    checkout: the packaged copy is guaranteed to match the running code.
    """
    packaged = _packaged_readme()

    if packaged:
        return packaged

    for path in SYSTEM_DOC_PATHS:
        if os.path.isfile(path):
            return path

    return _checkout_readme()


def readme_hint(section=None):
    """Build a one-line "see the docs" string suitable for logging.

    section: optional section title to name, e.g. "2. Set up the models".
    Always returns a non-empty string, so callers can append it to an error
    message unconditionally.
    """
    path = readme_path()

    if path and section:
        return "See {} -- section '{}'".format(path, section)
    if path:
        return "See {}".format(path)
    if section:
        return (
            "See the project README (section '{}'); it is installed at "
            "{} on the board.".format(section, SYSTEM_DOC_PATHS[0])
        )
    return "See the project README, installed at {} on the board.".format(
        SYSTEM_DOC_PATHS[0]
    )
