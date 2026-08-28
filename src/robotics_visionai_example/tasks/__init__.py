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

from robotics_visionai_example.tasks.base import Result, TaskProcessor
from robotics_visionai_example.tasks.detection import DetectionTask, Detections
from robotics_visionai_example.tasks.pose import PoseTask, Pose

TASK_REGISTRY = {
    DetectionTask.NAME: DetectionTask,
    PoseTask.NAME: PoseTask,
}


def create_task(name, **kwargs):
    """Instantiate the TaskProcessor registered under `name`.

    kwargs are forwarded to the TaskProcessor constructor (e.g. iou_threshold
    for DetectionTask).
    """
    try:
        cls = TASK_REGISTRY[name]
    except KeyError:
        raise ValueError(
            "unknown task '{}'. Available: {}".format(
                name, ", ".join(sorted(TASK_REGISTRY))
            )
        )
    return cls(**kwargs)


__all__ = [
    "Result", "TaskProcessor",
    "DetectionTask", "Detections",
    "PoseTask", "Pose",
    "TASK_REGISTRY", "create_task",
]
