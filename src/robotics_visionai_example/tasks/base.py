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

from abc import ABC, abstractmethod


class PreprocessPolicy:
    """How a TFLite engine should map a source BGR frame onto a model input.

    letterbox:  True for aspect-ratio-preserving resize+pad (YOLOv8-style),
                False for plain per-axis stretch resize (MoveNet-style).
    bgr_to_rgb: whether the model expects RGB channel order.
    """
    __slots__ = ("letterbox", "bgr_to_rgb")

    def __init__(self, letterbox=True, bgr_to_rgb=False):
        self.letterbox = letterbox
        self.bgr_to_rgb = bgr_to_rgb


class Result:
    """Base class for a single frame's decoded task output.

    Subclasses hold whatever numpy arrays make sense for that task, so
    app/state.py and app/server.py can pass a Result around without knowing
    its concrete type.
    """
    pass


class TaskProcessor(ABC):
    """Decodes raw model tensors into a Result, and can draw/serialize it.

    NAME: short identifier, e.g. "detection" or "pose". Matches the model
    registry <category> folder name and the tasks/ module name.
    """
    NAME = None

    @abstractmethod
    def preprocess_policy(self):
        """Return the PreprocessPolicy for this task's models.

        Only used by the TFLite engines: the ARA engine takes its preprocess
        geometry from the loaded model.dvm's own metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, tensors, letterbox, src_hw):
        """Decode float32 model output tensors into a Result.

        letterbox: the LetterboxInfo describing the preprocessing geometry
                   used for these tensors, for mapping coordinates back.
        src_hw: (h, w) of the original frame.
        """
        raise NotImplementedError

    @abstractmethod
    def draw(self, bgr_frame, result, fps=None, backend=""):
        """Return a new annotated BGR frame (bgr_frame is not modified)."""
        raise NotImplementedError

    @abstractmethod
    def to_json(self, result):
        """Return a plain-dict/list JSON-serializable summary of result."""
        raise NotImplementedError

    def empty_result(self):
        """Convenience Result to use before the first frame is available."""
        raise NotImplementedError
