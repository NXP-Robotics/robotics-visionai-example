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

from robotics_visionai_example.common.labels import COCO_CLASSES, COLORS
from robotics_visionai_example.common.letterbox import (
    numpy_nms, scale_coords, xywh2xyxy, xyxy2xywh,
)
from robotics_visionai_example.tasks.base import PreprocessPolicy, Result, TaskProcessor

DEFAULT_IOU_THRESHOLD = 0.45

DEFAULT_NMS_SCORE_THRESHOLD = 0.25
DEFAULT_TOP_K = 300


class Detections(Result):
    """boxes: (N, 4) [xmin, ymin, w, h] in source-frame pixel coords.
    scores: (N,) confidence in [0, 1].
    classes: (N,) integer COCO class ids.
    """
    __slots__ = ("boxes", "scores", "classes")

    def __init__(self, boxes, scores, classes):
        self.boxes = boxes
        self.scores = scores
        self.classes = classes

    def __len__(self):
        return len(self.boxes)


def _nms_yolov8(pred, nms_score_threshold=0.25, iou_threshold=0.45, max_det=300,
               is_agnostic_nms=False, multi_label=False, labels=()):
    """Threshold + class-wise NMS on a (batch, 4+nc, N) prediction tensor.

    Returns, per batch element, an (M, 6) array of
    [x1, y1, x2, y2, score, class_id] in the same coordinate space as the input.
    """
    assert 0 <= nms_score_threshold <= 1
    assert 0 <= iou_threshold <= 1

    bs = pred.shape[0]
    nc = pred.shape[1] - 4
    mi = 4 + nc
    xc = numpy.amax(pred[:, 4:mi], 1) > nms_score_threshold

    max_wh = 7680
    max_nms = 30000
    multi_label = multi_label and nc > 1

    pred = pred.transpose(0, -1, -2)
    output = [numpy.zeros((0, 6))] * bs

    for xi, x in enumerate(pred):
        x = x[xc[xi]]
        if not len(x):
            continue
        box = xywh2xyxy(x[:, :4])
        cls = x[:, 4:mi]

        if multi_label:
            i, j = (cls > nms_score_threshold).nonzero()
            x = numpy.concatenate(
                (box[i], cls[i, j, None], j[:, None].astype(numpy.float32)), 1
            )
        else:
            j = cls.argmax(axis=1)
            conf = cls[numpy.arange(cls.shape[0]), j]
            x = numpy.concatenate(
                (box, conf[:, None], j[:, None].astype(numpy.float32)), 1
            )
            x = x[x[:, 4] > nms_score_threshold]

        if not x.shape[0]:
            continue
        x = x[(-x[:, 4]).argsort(kind="mergesort")[:max_nms]]

        c = x[:, 5:6] * (0 if is_agnostic_nms else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = numpy.array(numpy_nms(boxes, scores, iou_threshold))
        if i.shape[0] > max_det:
            i = i[:max_det]
        output[xi] = x[i]

    return output



class DetectionTask(TaskProcessor):
    """YOLOv8 detection decode + draw + JSON, parameterised by NMS tunables."""
    NAME = "detection"

    def __init__(self, iou_threshold=DEFAULT_IOU_THRESHOLD,
                 nms_score_threshold=DEFAULT_NMS_SCORE_THRESHOLD,
                 top_k=DEFAULT_TOP_K):
        self.iou_threshold = iou_threshold
        self.nms_score_threshold = nms_score_threshold
        self.top_k = top_k

    def empty_result(self):
        return Detections(numpy.zeros((0, 4)), numpy.zeros((0,)), numpy.zeros((0,)))

    def preprocess_policy(self):
        # YOLOv8 uses aspect-ratio-preserving letterbox resize + RGB input.
        return PreprocessPolicy(letterbox=True, bgr_to_rgb=True)

    def decode(self, tensors, letterbox, src_hw):
        """tensors: [boxes (1, 4, N), scores (1, num_classes, N)].

        Boxes are cxcywh in model-input pixels and scores are already
        sigmoid-activated; the engine has normalised whichever output layout
        its model uses into this pair (see common/yolov8_layout.py), so this
        task is layout-agnostic. Concatenating on axis=1 gives the
        (1, 4+num_classes, N) tensor _nms_yolov8 expects.
        """
        boxes = numpy.concatenate([tensors[0], tensors[1]], axis=1)


        nms_out = _nms_yolov8(boxes, self.nms_score_threshold, self.iou_threshold,
                              self.top_k, is_agnostic_nms=False, multi_label=True)
        nms_out = nms_out[0]

        if len(nms_out) == 0:
            return self.empty_result()

        box_coords = nms_out[:, :4]
        scores = nms_out[:, 4]
        classes = nms_out[:, 5]
        box_coords = scale_coords(
            [letterbox.model_hw[0], letterbox.model_hw[1]], box_coords, list(src_hw)
        )
        box_coords = xyxy2xywh(box_coords)
        box_coords[:, :2] -= box_coords[:, 2:] / 2

        return Detections(box_coords, scores, classes)

    def draw(self, bgr_frame, result, fps=None, backend=""):
        """Draw detection boxes on a BGR frame and return an annotated copy."""
        out = bgr_frame.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        boxes, scores, classes = result.boxes, result.scores, result.classes

        for i in range(len(boxes)):
            box = boxes[i]
            cls_id = int(classes[i])
            score = float(scores[i])
            color_idx = cls_id % len(COLORS)
            color = (COLORS[color_idx] * 255).astype(numpy.uint8).tolist()
            label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
            text = "{}:{:.1f}%".format(label, score * 100)
            txt_color = (0, 0, 0) if numpy.mean(COLORS[color_idx]) > 0.5 else (255, 255, 255)
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.rectangle(
                out,
                (int(box[0]), int(box[1])),
                (int(box[0] + box[2]), int(box[1] + box[3])),
                color,
                2,
            )
            cv2.putText(out, text, (int(box[0]), int(box[1]) + txt_size[1]),
                        font, 0.4, txt_color, thickness=1)

        if fps is not None:
            cv2.putText(out, f'FPS ({backend}): {fps:.1f}', (10, 22),
                        font, 0.65, (200, 0, 255), 2, cv2.LINE_AA)

        return out

    def to_json(self, result):
        det_list = []
        for i in range(len(result.boxes)):
            cls_id = int(result.classes[i])
            label = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else str(cls_id)
            det_list.append({
                "label": label,
                "score": round(float(result.scores[i]), 3),
                "box": [round(float(v), 1) for v in result.boxes[i]],
            })
        return det_list
