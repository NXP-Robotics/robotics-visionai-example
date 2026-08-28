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

"""Output-layout detection and normalisation for YOLOv8 detection models.

The three layouts:

  SPLIT     2 tensors, (1, 4, N) + (1, nc, N).
            The ARA-240 .dvm convention

  CONCAT    1 tensor, (1, 4+nc, N) or its transpose (1, N, 4+nc).
            Stock Ultralytics: what `yolo export format=onnx|tflite` produces
            and what the eIQ Neutron flow (onnx2tflite -> tflite-quantizer ->
            neutron-converter) carries through to a .tflite.

  RAW_HEAD  3 tensors, (1, H, W, 4*reg_max+nc) for strides 8/16/32.
            The detection head truncated before its DFL decode tail. That tail
            is Softmax/Transpose/Split/Reshape, none of which Neutron-S can
            execute, so cutting it is what lets the whole graph run on the NPU
            instead of being partitioned. The decode moves here, into
            decode_raw_head().
"""

import numpy


LAYOUT_SPLIT = "split"
LAYOUT_CONCAT = "concat"
LAYOUT_RAW_HEAD = "raw_head"

REG_MAX = 16
NUM_SIDES = 4

STRIDES = (8, 16, 32)

_NORMALIZED_MAX = 2.0


def detect_layout(shapes):
    """Classify a model's output shapes as one of the LAYOUT_* constants.

    shapes: sequence of per-output shape tuples, in the interpreter's order.
    Returns (layout, num_classes), or (None, None) when nothing matches.
    """
    shapes = [tuple(int(d) for d in s) for s in shapes]

    if len(shapes) == 2 and all(len(s) == 3 for s in shapes):
        box = next((s for s in shapes if s[1] == NUM_SIDES), None)
        score = next((s for s in shapes if s[1] != NUM_SIDES), None)
        if box is not None and score is not None and box[2] == score[2]:
            return LAYOUT_SPLIT, score[1]

    if len(shapes) == 1 and len(shapes[0]) == 3:
        _, a, b = shapes[0]
        channels = min(a, b)
        if channels > NUM_SIDES:
            return LAYOUT_CONCAT, channels - NUM_SIDES

    if len(shapes) == 3 and all(len(s) == 4 for s in shapes):
        channels = {s[3] for s in shapes}
        if len(channels) == 1:
            c = channels.pop()
            num_classes = c - NUM_SIDES * REG_MAX
            if num_classes > 0:
                return LAYOUT_RAW_HEAD, num_classes

    return None, None


def _sigmoid(x):
    return 1.0 / (1.0 + numpy.exp(-x))


def _softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = numpy.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def make_anchors(model_hw):
    """Cell-centre anchor points and per-anchor strides over all three scales.

    Returns (points (N, 2) in cell units, strides (N, 1)), matching
    ultralytics.utils.tal.make_anchors with grid_cell_offset=0.5.
    """
    model_h, model_w = model_hw
    points, strides = [], []
    for stride in STRIDES:
        nh, nw = model_h // stride, model_w // stride
        xs = numpy.arange(nw, dtype=numpy.float32) + 0.5
        ys = numpy.arange(nh, dtype=numpy.float32) + 0.5
        grid_y, grid_x = numpy.meshgrid(ys, xs, indexing="ij")
        points.append(numpy.stack((grid_x.ravel(), grid_y.ravel()), axis=1))
        strides.append(numpy.full((nh * nw, 1), stride, dtype=numpy.float32))
    return numpy.concatenate(points), numpy.concatenate(strides)


def decode_raw_head(tensors, model_hw, num_classes):
    """DFL-decode the three raw head tensors into (boxes, scores).

    Each tensor is (1, H, W, 4*REG_MAX + num_classes) NHWC, channels ordered
    [64 box distance logits | num_classes class logits] because the head
    concatenates cv2 (box) before cv3 (class).

    Returns boxes (1, 4, N) as cxcywh in model-input pixels and scores
    (1, num_classes, N) sigmoid-activated, i.e. exactly the SPLIT layout.
    """
    expected_c = NUM_SIDES * REG_MAX + num_classes

    ordered = sorted(tensors, key=lambda t: -t.shape[1])

    box_parts, cls_parts = [], []
    for arr in ordered:
        if arr.ndim != 4 or arr.shape[3] != expected_c:
            raise ValueError(
                "expected a (1, H, W, {}) raw head tensor, got {}".format(
                    expected_c, tuple(arr.shape)
                )
            )
        flat = arr.reshape(-1, expected_c)
        box_parts.append(flat[:, :NUM_SIDES * REG_MAX])
        cls_parts.append(flat[:, NUM_SIDES * REG_MAX:])

    box_logits = numpy.concatenate(box_parts, axis=0)
    cls_logits = numpy.concatenate(cls_parts, axis=0)

    n = box_logits.shape[0]
    points, strides = make_anchors(model_hw)
    if points.shape[0] != n:
        raise ValueError(
            "anchor count {} does not match prediction count {}; the model's "
            "input size and its head resolutions disagree".format(
                points.shape[0], n
            )
        )

    bins = box_logits.reshape(n, NUM_SIDES, REG_MAX)
    dist = (_softmax(bins, axis=2)
            * numpy.arange(REG_MAX, dtype=numpy.float32)).sum(axis=2)

    d = dist * strides
    centres = points * strides
    x1 = centres[:, 0] - d[:, 0]
    y1 = centres[:, 1] - d[:, 1]
    x2 = centres[:, 0] + d[:, 2]
    y2 = centres[:, 1] + d[:, 3]

    boxes = numpy.stack(
        ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1), axis=0
    )[numpy.newaxis].astype(numpy.float32)
    scores = _sigmoid(cls_logits).T[numpy.newaxis].astype(numpy.float32)
    return boxes, scores


def split_concat_head(tensor, num_classes):
    """Split a single (1, 4+nc, N) or (1, N, 4+nc) tensor into (boxes, scores)."""
    arr = tensor[0] if tensor.ndim == 3 else tensor
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T                              # -> (4+nc, N)

    boxes = arr[:NUM_SIDES][numpy.newaxis]
    scores = arr[NUM_SIDES:NUM_SIDES + num_classes][numpy.newaxis]
    return boxes, scores


def to_pixels(boxes, model_hw):
    """Scale normalised 0..1 box coordinates up to model-input pixels.

    A no-op when the values are already in pixels, so both the pixel-space
    (ONNX/imx) and normalised (tflite) Ultralytics heads decode correctly.
    """
    if not boxes.size or float(numpy.max(numpy.abs(boxes))) > _NORMALIZED_MAX:
        return boxes

    model_h, model_w = model_hw
    return boxes * numpy.array(
        [[[model_w], [model_h], [model_w], [model_h]]], dtype=numpy.float32
    )


def normalize_outputs(tensors, layout, num_classes, model_hw):
    """Convert any supported layout into [boxes (1,4,N), scores (1,nc,N)].

    Boxes come out as cxcywh in model-input pixels, scores sigmoid-activated.
    """
    if layout == LAYOUT_RAW_HEAD:
        boxes, scores = decode_raw_head(tensors, model_hw, num_classes)
    elif layout == LAYOUT_CONCAT:
        boxes, scores = split_concat_head(tensors[0], num_classes)
    elif layout == LAYOUT_SPLIT:
        box_index = next(
            (i for i, t in enumerate(tensors)
             if t.ndim == 3 and t.shape[1] == NUM_SIDES),
            0,
        )
        boxes = tensors[box_index]
        scores = tensors[1 - box_index]
    else:
        raise ValueError("unknown YOLOv8 output layout: {}".format(layout))

    return [to_pixels(boxes, model_hw), scores]
