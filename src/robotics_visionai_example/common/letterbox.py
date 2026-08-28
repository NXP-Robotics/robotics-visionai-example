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


class LetterboxInfo:
    """Describes how a source frame was mapped onto a model's input tensor.

    Engines build this during preprocess() and pass it to a TaskProcessor's
    decode(), so coordinates can be mapped back to the original frame
    regardless of which engine produced them.

    model_hw: (h, w) of the model input tensor.
    src_hw:   (h, w) of the original frame.
    gain:     (gain_y, gain_x) scale factors applied to the source image.
              Equal for letterboxed inputs, independent per-axis for a plain
              stretch resize.
    pad:      (pad_top, pad_left) in model-input pixels, (0, 0) when not
              letterboxed.
    """
    __slots__ = ("model_hw", "src_hw", "gain", "pad")

    def __init__(self, model_hw, src_hw, gain, pad=(0.0, 0.0)):
        self.model_hw = model_hw
        self.src_hw = src_hw
        self.gain = gain
        self.pad = pad

    @classmethod
    def letterboxed(cls, model_hw, src_hw):
        """Aspect-ratio-preserving resize+pad (YOLOv8-style)."""
        mh, mw = model_hw
        sh, sw = src_hw
        gain = min(mh / sh, mw / sw)
        resized_h = int(round(sh * gain))
        resized_w = int(round(sw * gain))
        pad_top = int(round((mh - resized_h) / 2 - 0.1))
        pad_left = int(round((mw - resized_w) / 2 - 0.1))
        return cls(model_hw, src_hw, (gain, gain), (pad_top, pad_left))

    @classmethod
    def stretched(cls, model_hw, src_hw):
        """Plain per-axis stretch resize, no padding (MoveNet-style)."""
        mh, mw = model_hw
        sh, sw = src_hw
        return cls(model_hw, src_hw, (mh / sh, mw / sw), (0.0, 0.0))


def preprocess_frame(bgr_frame, preprocess_cfg):
    """Preprocess an in-memory BGR numpy frame for a YOLOv8-style model.

    Steps: BGR->RGB (optional) -> aspect-ratio resize+pad -> scale -> HWC->CHW
    Returns a float32 numpy array of shape (C, H, W).
    """
    c = preprocess_cfg["input_shape"]["c"]
    h = preprocess_cfg["input_shape"]["h"]
    w = preprocess_cfg["input_shape"]["w"]
    pad_val = preprocess_cfg.get("padding", {}).get("color_val", 114)
    scale = preprocess_cfg.get("scale", {})
    bgr_to_rgb = preprocess_cfg.get("bgr_to_rgb", False)

    # work on uint8 for channel conversions, convert to float32 only once
    img_u8 = bgr_frame if bgr_frame.dtype == numpy.uint8 else bgr_frame.astype(numpy.uint8)

    # handle channel conversion on uint8 (no extra float32 copies)
    if img_u8.shape[2] == 3 and c == 1:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)
    elif img_u8.shape[2] == 1 and c == 3:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    elif img_u8.shape[2] == 4 and c == 3:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_BGRA2BGR)

    if bgr_to_rgb:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB)

    # resize on uint8 (faster), then single cast to float32
    img_h, img_w = img_u8.shape[:2]
    ratio = min(h / img_h, w / img_w)
    resized_h = int(round(img_h * ratio))
    resized_w = int(round(img_w * ratio))
    img_u8 = cv2.resize(img_u8, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    image = img_u8.astype(numpy.float32)

    # pad once into a pre-allocated buffer
    padded = numpy.full((h, w, 3 if c >= 3 else 1), float(pad_val), dtype=numpy.float32)
    dh = (h - resized_h) / 2
    dw = (w - resized_w) / 2
    top = int(round(dh - 0.1))
    left = int(round(dw - 0.1))
    padded[top:resized_h + top, left:resized_w + left] = image

    # fused per-channel scale (e.g. 1/255)
    if scale and scale.get("r") and scale.get("g") and scale.get("b"):
        scale_arr = numpy.array([scale["r"], scale["g"], scale["b"]], dtype=numpy.float32)
        padded *= scale_arr  # broadcasts over HxW

    # HWC -> CHW
    return numpy.ascontiguousarray(padded.transpose(2, 0, 1))


def xywh2xyxy(x):
    y = numpy.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def xyxy2xywh(x):
    y = numpy.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def numpy_nms(dets, scores, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(kind="mergesort")[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = numpy.maximum(x1[i], x1[order[1:]])
        yy1 = numpy.maximum(y1[i], y1[order[1:]])
        xx2 = numpy.minimum(x2[i], x2[order[1:]])
        yy2 = numpy.minimum(y2[i], y2[order[1:]])
        w = numpy.maximum(0.0, xx2 - xx1 + 1)
        h = numpy.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = numpy.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def scale_coords(model_input_shape, coords, img_shape, ratio_pad=None):
    """Rescale (x1, y1, x2, y2) coords from model input space to img_shape space.

    model_input_shape: [h, w] of the model input tensor.
    img_shape: [h, w] of the original image.
    """
    if ratio_pad is None:
        gain = min(model_input_shape[0] / img_shape[0], model_input_shape[1] / img_shape[1])
        pad = (
            (model_input_shape[1] - img_shape[1] * gain) / 2,
            (model_input_shape[0] - img_shape[0] * gain) / 2,
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0] = numpy.clip(coords[:, 0], 0, img_shape[1])
    coords[:, 2] = numpy.clip(coords[:, 2], 0, img_shape[1])
    coords[:, 1] = numpy.clip(coords[:, 1], 0, img_shape[0])
    coords[:, 3] = numpy.clip(coords[:, 3], 0, img_shape[0])
    return coords
