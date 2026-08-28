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


class RawOutputs:
    """An engine's raw inference output.

    tensors: float32 numpy arrays in the layout the task's decode() expects.
    output_shapes: optional per-tensor (nch, height) metadata, used by engines
        whose dequantized outputs come back flat and need reshaping.
    """
    __slots__ = ("tensors", "output_shapes")

    def __init__(self, tensors, output_shapes=None):
        self.tensors = tensors
        self.output_shapes = output_shapes


class InferenceEngine(ABC):
    """Base class for a backend-specific inference engine.

    NAME: short backend identifier, e.g. "ARA", "NEUTRON", "CPU".
    SUPPORTED_TASKS: task names (see tasks.TASK_REGISTRY) this engine has a
        working preprocess/decode path for.
    """
    NAME = None
    SUPPORTED_TASKS = ()

    def __init__(self, task, model, models_dir=None):
        self.task = task              # TaskProcessor instance
        self.model_name = model       # "<task>/<name>", e.g. "detection/yolov8s"
        self.models_dir = models_dir

    @abstractmethod
    def init(self):
        """Load self.model_name and get ready for inference."""
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """Release any hardware/interpreter resources."""
        raise NotImplementedError

    @abstractmethod
    def reload(self, model):
        """Hot-swap to a different model without recreating this engine."""
        raise NotImplementedError

    # -- 3-stage pipeline API (drives app/stream.py's threaded pipeline) ---

    @abstractmethod
    def preprocess(self, bgr_frame):
        """bgr_frame -> (backend-native input tensor, timings dict)."""
        raise NotImplementedError

    @abstractmethod
    def infer(self, prepared_input):
        """backend-native input tensor -> (RawOutputs, timings dict)."""
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, raw_outputs, src_hw):
        """RawOutputs, (h, w) of the source frame -> (Result, timings dict)."""
        raise NotImplementedError

    def infer_frame_timed(self, bgr_frame):
        """Run the full pipeline, returning (Result, merged timings dict)."""
        src_hw = bgr_frame.shape[:2]
        prepared, t_pre = self.preprocess(bgr_frame)
        raw, t_inf = self.infer(prepared)
        result, t_post = self.postprocess(raw, src_hw)
        timings = {}
        timings.update(t_pre)
        timings.update(t_inf)
        timings.update(t_post)
        return result, timings
