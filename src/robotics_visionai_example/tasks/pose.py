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

import cv2
import numpy

from robotics_visionai_example.common.labels import KEYPOINTS_DEF
from robotics_visionai_example.tasks.base import PreprocessPolicy, Result, TaskProcessor


MODEL_KEYPOINT_SIZE = 17
MODEL_SCORE_THRESHOLD = 0.4


class Pose(Result):
    """keypoints: (K, 3) array of [y, x, valid] in source-frame pixel coords.

    valid is encoded as 1.0/0.0 (score >= the task's score_threshold).
    """
    __slots__ = ("keypoints",)

    def __init__(self, keypoints):
        self.keypoints = keypoints


class PoseTask(TaskProcessor):
    """MoveNet 17-keypoint decode + skeleton draw + JSON."""
    NAME = "pose"

    def __init__(self, score_threshold=MODEL_SCORE_THRESHOLD):
        self.score_threshold = score_threshold

    def empty_result(self):
        return Pose(None)

    def preprocess_policy(self):
        # MoveNet uses a plain stretch resize (no letterbox padding), RGB input.
        return PreprocessPolicy(letterbox=False, bgr_to_rgb=True)

    def decode(self, tensors, letterbox, src_hw):
        """tensors[0]: raw model output, reshapeable to (K, 3) [y, x, score]
        normalized to the model input tensor size.
        """
        model_h, model_w = letterbox.model_hw
        gain_y, gain_x = letterbox.gain
        pad_top, pad_left = letterbox.pad

        kpts = numpy.asarray(tensors[0]).reshape(MODEL_KEYPOINT_SIZE, 3).astype(numpy.float32).copy()

        # normalized -> model-input pixel coords -> source-frame pixel coords
        model_px_y = kpts[:, 0] * model_h
        model_px_x = kpts[:, 1] * model_w
        kpts[:, 0] = (model_px_y - pad_top) / gain_y
        kpts[:, 1] = (model_px_x - pad_left) / gain_x

        valid = (kpts[:, 2] >= self.score_threshold).astype(numpy.float32)
        kpts[:, 2] = valid

        return Pose(kpts)

    def draw(self, bgr_frame, result, fps=None, backend=""):
        """Draw skeleton and keypoints on a BGR frame (returns a new copy)."""
        out = bgr_frame.copy()
        kpts = result.keypoints if result is not None else None

        if kpts is not None:
            # connections first, so the dots draw on top of them
            for i, kd in enumerate(KEYPOINTS_DEF):

                if not kpts[i, 2]:
                    continue
                x1 = int(kpts[i, 1])
                y1 = int(kpts[i, 0])
                for j in kd['connections']:
                    if not kpts[j, 2]:
                        continue
                    x2 = int(kpts[j, 1])
                    y2 = int(kpts[j, 0])
                    cv2.line(out, (x1, y1), (x2, y2), (0, 220, 0), 2, cv2.LINE_AA)

            for i, kd in enumerate(KEYPOINTS_DEF):

                if not kpts[i, 2]:
                    continue
                x = int(kpts[i, 1])
                y = int(kpts[i, 0])
                cv2.circle(out, (x, y), 4, (0, 0, 220), -1, cv2.LINE_AA)
                cv2.circle(out, (x, y), 4, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(out, kd['label'], (x + 5, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 220), 1, cv2.LINE_AA)

        if fps is not None:
            cv2.putText(out, f'FPS ({backend}): {fps:.1f}', (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 0, 255), 2, cv2.LINE_AA)
        return out

    def to_json(self, result):
        kpts = result.keypoints if result is not None else None
        kpt_list = []
        if kpts is not None:
            for i, kd in enumerate(KEYPOINTS_DEF):
                y, x, valid = float(kpts[i, 0]), float(kpts[i, 1]), bool(kpts[i, 2])
                kpt_list.append({'label': kd['label'], 'x': round(x, 1),
                                 'y': round(y, 1), 'valid': valid})
        return kpt_list
