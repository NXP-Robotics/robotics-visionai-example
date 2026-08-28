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

ENGINE_NAMES = ("ARA", "NEUTRON", "GPU", "CPU")


def _engine_class(name):
    if name == "ARA":
        from robotics_visionai_example.engines.ara_engine import AraEngine
        return AraEngine
    if name in ("NEUTRON", "GPU", "CPU"):
        from robotics_visionai_example.engines import tflite_engine
        return {
            "NEUTRON": tflite_engine.NeutronEngine,
            "GPU": tflite_engine.GpuEngine,
            "CPU": tflite_engine.CpuEngine,
        }[name]
    raise ValueError(
        "unknown engine '{}'. Available: {}".format(name, ", ".join(ENGINE_NAMES))
    )


def engine_class(name):
    """Return the InferenceEngine subclass registered under `name`.

    Useful for querying a backend's capabilities (SUPPORTED_TASKS,
    list_models) before an instance exists.
    """
    return _engine_class(name)


def create_engine(name, task, **kwargs):
    """Instantiate the InferenceEngine registered under `name`.

    kwargs are forwarded to the engine constructor (model=... for AraEngine,
    model_path=... for the TFLite engines).

    Raises ValueError if `task` is not in the engine's SUPPORTED_TASKS, so a
    mismatch fails here instead of deep inside a worker thread.
    """
    cls = _engine_class(name)
    if task.NAME not in cls.SUPPORTED_TASKS:
        raise ValueError(
            "engine '{}' does not support task '{}'. Supported tasks: {}".format(
                name, task.NAME, ", ".join(cls.SUPPORTED_TASKS) or "(none)"
            )
        )
    return cls(task, **kwargs)


__all__ = ["ENGINE_NAMES", "engine_class", "create_engine"]

