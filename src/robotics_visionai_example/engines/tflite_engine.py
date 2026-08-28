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

import logging
import os
import re
import time

import cv2
import numpy

from robotics_visionai_example.common import yolov8_layout
from robotics_visionai_example.common.docs import readme_hint
from robotics_visionai_example.common.letterbox import LetterboxInfo
from robotics_visionai_example.engines.base import InferenceEngine, RawOutputs


logger = logging.getLogger(__name__)

TFLITE_CATEGORY_TASKS = {
    "object-detection": "detection",
    "pose-estimation": "pose",
}

TASK_MODEL_FAMILIES = {
    "detection": ("yolov8",),
}


NEUTRON_CNN_DIR = os.environ.get("NEUTRON_CNN_DIR", "/usr/share/cnn/neutron")

SOC_ID_PATH = "/sys/devices/soc0/soc_id"

_VARIANT_RE = re.compile(r"[._](imx\d+)$", re.IGNORECASE)


def _load_tflite_runtime():
    """Prefer the lightweight tflite_runtime package, fall back to tensorflow.lite."""
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
    return tflite


def read_soc_id():
    """Read the board's SoC identifier from sysfs (e.g. "i.MX95").

    Returns None when the path does not exist, i.e. when running off-board.
    """
    try:
        with open(SOC_ID_PATH) as f:
            return f.read().strip()
    except OSError:
        return None


def split_variant(stem):
    """Split a model filename stem into (base, soc_variant).

    "movenet_quant_imx95"        -> ("movenet_quant", "imx95")
    "yolov8n-det-int8.imx95"     -> ("yolov8n-det-int8", "imx95")
    "movenet_quant"              -> ("movenet_quant", None)
    """
    match = _VARIANT_RE.search(stem)
    if match is None:
        return stem, None
    return stem[:match.start()], match.group(1).lower()


def needs_delegate(model_path):
    """True if this model was compiled for an accelerator delegate.

    Inferred from the SoC variant tag in the filename: such a model contains
    a vendor custom op (e.g. NeutronGraph) that plain TFLite cannot resolve.
    """
    stem = os.path.splitext(os.path.basename(model_path))[0]
    return split_variant(stem)[1] is not None


def _soc_variant_tags(portable_only=False):
    """SoC variant tags to prefer for this board, most specific first.

    A None entry means "the file with no variant tag at all", which is the
    portable model every board and every backend can run.

    portable_only: the caller has no accelerator delegate, so only the untagged
        file is usable even when a variant matching this board exists. Without
        it a backend such as CPU would resolve yolov8n-det-int8 to the .imx95
        file sitting next to the portable one and then reject it at load time.
    """
    if portable_only:
        return [None]
    soc_id = read_soc_id()
    if soc_id is None:
        return [None]
    # "i.MX95" -> "imx95"
    tag = soc_id.replace(".", "").replace("-", "").lower()
    return [tag, None]


def list_models(models_dir=None, tasks=None, require_portable=False):
    """List "<category>/<name>" TFLite models installed under models_dir.

    One entry per distinct model, using the base name with the SoC variant tag
    stripped: the tag is re-applied at resolve time for whichever board this
    is, so the same name works everywhere.

    Only models whose family the task has a decode path for are listed (see
    TASK_MODEL_FAMILIES), so the caller never gets a model that would decode
    into nonsense.

    tasks: restrict to categories mapping to these task names (default: all
        categories that map to a known task).
    require_portable: only list models runnable without an accelerator
        delegate, i.e. those with an untagged file present.
    """
    models_dir = models_dir or NEUTRON_CNN_DIR


    found = []
    if not os.path.isdir(models_dir):
        return found

    for category in sorted(os.listdir(models_dir)):
        task_name = TFLITE_CATEGORY_TASKS.get(category)
        if task_name is None:
            continue
        if tasks is not None and task_name not in tasks:
            continue

        category_dir = os.path.join(models_dir, category)
        if not os.path.isdir(category_dir):
            continue

        # Group the per-SoC files of one model under its shared base name.
        variants = {}
        for filename in os.listdir(category_dir):
            if not filename.endswith(".tflite"):
                continue
            base, variant = split_variant(os.path.splitext(filename)[0])
            variants.setdefault(base, set()).add(variant)

        families = TASK_MODEL_FAMILIES.get(task_name)
        for base in sorted(variants):
            if require_portable and None not in variants[base]:
                continue
            if families is not None and not any(f in base.lower() for f in families):
                continue
            found.append("{}/{}".format(category, base))

    return found



def _resolve_variant_file(category_dir, base, portable_only=False):
    """Return the best <base>[.<soc>].tflite in category_dir for this board.

    portable_only restricts the preference to the untagged (portable) file,
    for a backend that cannot load an accelerator-compiled graph.
    """
    if not os.path.isdir(category_dir):
        return None

    candidates = {}
    for filename in os.listdir(category_dir):
        if not filename.endswith(".tflite"):
            continue
        stem = os.path.splitext(filename)[0]
        file_base, variant = split_variant(stem)
        if file_base == base:
            candidates[variant] = os.path.join(category_dir, filename)

    if not candidates:
        return None

    for tag in _soc_variant_tags(portable_only):
        if tag in candidates:
            return candidates[tag]

    # No usable variant: either nothing matches this board, or a delegate-less
    # backend asked for a model that only ships accelerator-compiled. Fall back
    # to any variant rather than reporting the model as missing, so an explicit
    # request still gets a clear error from _load_model() or the delegate
    # instead of a misleading "not found".
    return candidates[sorted(candidates, key=lambda v: (v is None, v or ""))[0]]


def resolve_model_path(model, models_dir=None, tasks=None, portable_only=False):
    """Resolve a .tflite path, accepting (in order) a direct .tflite path,
    "<category>/<name>" under models_dir, or a bare name searched across the
    known category directories.

    The SoC variant suffix may be omitted: the best variant for this board is
    selected from the files actually present. portable_only asks for the
    untagged variant, for a backend with no accelerator delegate.
    """
    models_dir = models_dir or NEUTRON_CNN_DIR

    if model.endswith(".tflite") and os.path.isfile(model):
        return os.path.abspath(model)

    categories = [c for c, t in TFLITE_CATEGORY_TASKS.items()
                  if tasks is None or t in tasks]

    if "/" in model:
        category, _, base = model.rpartition("/")
        # An absolute/relative directory prefix is honoured as-is, so a model
        # outside the registry tree can still be named without its variant.
        if os.path.isdir(category):
            path = _resolve_variant_file(category, base, portable_only)
        else:
            path = _resolve_variant_file(
                os.path.join(models_dir, category), base, portable_only
            )
        if path is not None:
            return os.path.abspath(path)
    else:
        for category in categories:
            path = _resolve_variant_file(
                os.path.join(models_dir, category), model, portable_only
            )
            if path is not None:
                return os.path.abspath(path)

    available = list_models(models_dir, tasks)
    if available:
        raise FileNotFoundError(
            "TFLite model '{}' not found under {}. Installed models: {}. {}".format(
                model, models_dir, ", ".join(available),
                readme_hint("2. Set up the models"),
            )
        )

    raise FileNotFoundError(
        "TFLite model '{}' not found and no models are installed under {}. "
        "Models are compiled on a host (e.g. from nxp-nnstreamer-examples) and "
        "copied to the board as <category>/<name>.tflite. Override the base "
        "directory with NEUTRON_CNN_DIR / --neutron-models-dir. {}".format(
            model, models_dir, readme_hint("2. Set up the models")
        )
    )



# YOLOv8 letterbox padding value, matching the ARA engine's DEFAULT_PAD_VALUE.
DEFAULT_PAD_VALUE = 114


class TFLiteEngine(InferenceEngine):
    """TFLite inference engine, optionally accelerated by an external delegate.

    Subclasses set NAME and DELEGATE_LIB. DELEGATE_LIB of None means plain
    CPU execution via TFLite's built-in XNNPACK backend.

    RUNS_COMPILED_MODELS says whether this backend can execute a model
    compiled for the Neutron NPU (a SoC-tagged file, carrying the NeutronGraph
    custom op). It is the single switch behind all three places that care:
    which models are listed, which variant of a model is resolved, and whether
    a resolved file is rejected at load time.
    """
    NAME = "TFLITE"
    DELEGATE_LIB = None
    RUNS_COMPILED_MODELS = False
    SUPPORTED_TASKS = ("pose", "detection")

    def __init__(self, task, model_path, models_dir=None, num_threads=None,
                 pad_value=DEFAULT_PAD_VALUE):
        super().__init__(task, model_path, models_dir=models_dir or NEUTRON_CNN_DIR)
        self.model_path = model_path
        self.num_threads = num_threads or (os.cpu_count() or 2)
        self._pad_value = pad_value

        self._interp = None
        self._input_detail = None
        self._output_details = None
        self._model_hw = None  # (h, w)
        self._input_quant = None    # (scale, zero_point) of the input tensor
        self._output_quant = None   # per-output (scale, zero_point)
        self._layout = None         # yolov8_layout.LAYOUT_* for a detection model
        self._num_classes = None


    @classmethod
    def list_models(cls, models_dir=None, tasks=None):
        """List the models this backend can actually run.

        A backend that cannot execute compiled graphs only sees models with a
        portable (untagged) file present; models shipping solely as .imx95 are
        left out instead of being offered and then refused.
        """
        return list_models(
            models_dir,
            tasks if tasks is not None else cls.SUPPORTED_TASKS,
            require_portable=not cls.RUNS_COMPILED_MODELS,
        )

    def _load_delegates(self, tflite):
        """Return the experimental_delegates list for this backend."""
        if self.DELEGATE_LIB is None:
            return []
        try:
            delegate = tflite.load_delegate(self.DELEGATE_LIB)
            logger.info("loaded delegate: %s", self.DELEGATE_LIB)
            return [delegate]
        except Exception as exc:
            logger.warning("could not load %s (%s) -- falling back to CPU/XNNPACK",
                           self.DELEGATE_LIB, exc)
            return []

    def _load_model(self):
        """Open an interpreter for self.model_path and cache its tensor details."""
        if not self.RUNS_COMPILED_MODELS and needs_delegate(self.model_path):
            raise RuntimeError(
                "model '{}' is compiled for an accelerator (its filename carries "
                "a SoC variant tag), so the {} backend cannot run it: it contains "
                "a custom op only the delegate implements. Run it with a backend "
                "that provides one (e.g. --backend NEUTRON), or pass a portable "
                "model with --model_path.".format(
                    os.path.basename(self.model_path), self.NAME
                )
            )

        tflite = _load_tflite_runtime()

        self._interp = tflite.Interpreter(
            model_path=self.model_path,
            experimental_delegates=self._load_delegates(tflite),
            num_threads=self.num_threads,
        )
        self._interp.allocate_tensors()
        self._input_detail = self._interp.get_input_details()[0]
        self._output_details = self._interp.get_output_details()
        _, h, w, _ = self._input_detail["shape"]
        self._model_hw = (int(h), int(w))

        self._input_quant = tuple(self._input_detail["quantization"])
        self._output_quant = [tuple(d["quantization"]) for d in self._output_details]

        self._check_output_layout()

        logger.info("%s engine ready: model=%s input=%s in_quant=%s out_quant=%s "
                    "threads=%s", self.NAME, self.model_path, self._model_hw,
                    self._input_quant, self._output_quant, self.num_threads)

    def _check_output_layout(self):
        """Identify which YOLOv8 output layout this model uses, or reject it.

        All three layouts a documented YOLOv8 export flow can produce are
        accepted (see common/yolov8_layout.py); the decode path is chosen from
        the output shapes. A model that matches none of them is rejected here
        rather than at the first frame, because --model_path accepts any file
        and silently decoding e.g. an SSDLite head with the YOLOv8 decoder
        would draw plausible-looking but meaningless boxes.
        """
        if self.task.NAME != "detection":
            return

        shapes = [tuple(d["shape"]) for d in self._output_details]
        self._layout, self._num_classes = yolov8_layout.detect_layout(shapes)

        if self._layout is None:
            raise RuntimeError(
                "model '{}' does not have a recognised YOLOv8 detection output "
                "layout (got {}). Supported layouts are: one (1, 4+nc, N) "
                "tensor (stock Ultralytics export), three (1, H, W, "
                "4*16+nc) raw head tensors (decode-in-host export), or a "
                "(1, 4, N) + (1, nc, N) pair. Only YOLOv8 detection models are "
                "supported today.".format(
                    os.path.basename(self.model_path), shapes
                )
            )

        logger.info("detected YOLOv8 output layout: %s (%d classes)",
                    self._layout, self._num_classes)



    def init(self):
        """Resolve the requested model and open an interpreter for it.

        Idempotent: callers may init() an engine that is already loaded (the
        CLI does so to surface model errors early, before VisionStream is
        built) without paying for a second model load.
        """
        if self._interp is not None:
            return

        self.model_path = resolve_model_path(
            self.model_path, self.models_dir, (self.task.NAME,),
            portable_only=not self.RUNS_COMPILED_MODELS,
        )
        self.model_name = self._display_name(self.model_path)
        self._load_model()


    def _display_name(self, model_path):
        """Registry-style "<category>/<name>" label for a resolved model path.

        Falls back to the bare filename for a model outside the registry tree,
        so /status always has something meaningful to show.
        """
        stem = os.path.splitext(os.path.basename(model_path))[0]
        base = split_variant(stem)[0]
        category = os.path.basename(os.path.dirname(model_path))
        if category in TFLITE_CATEGORY_TASKS:
            return "{}/{}".format(category, base)
        return base

    def reload(self, model):
        """Hot-swap to a different model, rolling back on failure.

        Not thread-safe: callers running the threaded pipeline in
        app/stream.py must park the worker threads around this call.
        """
        previous_path = self.model_path
        previous_name = self.model_name

        self.close()
        try:
            self.model_path = resolve_model_path(
                model, self.models_dir, (self.task.NAME,),
                portable_only=not self.RUNS_COMPILED_MODELS,
            )
            self.model_name = self._display_name(self.model_path)
            self._load_model()
        except Exception:
            logger.error("reload to model '%s' failed; rolling back to '%s'",
                         model, previous_name)
            self.model_path = previous_path
            self.model_name = previous_name
            try:
                self._load_model()
            except Exception as rollback_exc:
                logger.error("could not restore previous model: %s", rollback_exc)
            raise

    def close(self):
        self._interp = None

    def _letterbox_for(self, src_hw):
        if self.task.preprocess_policy().letterbox:
            return LetterboxInfo.letterboxed(self._model_hw, src_hw)
        return LetterboxInfo.stretched(self._model_hw, src_hw)

    def _resize_to_input(self, bgr_frame, letterbox):
        """Map a BGR frame onto the model input size per the task's policy.

        Letterboxed models get an aspect-ratio-preserving resize plus padding,
        using the very same geometry LetterboxInfo computed, so the coordinates
        decode() maps back are the ones actually fed to the model. Non-
        letterboxed (MoveNet-style) models get a plain stretch resize.
        """
        model_h, model_w = self._model_hw

        if not self.task.preprocess_policy().letterbox:
            return cv2.resize(bgr_frame, (model_w, model_h),
                              interpolation=cv2.INTER_LINEAR)

        src_h, src_w = bgr_frame.shape[:2]
        gain = letterbox.gain[0]
        resized_h = int(round(src_h * gain))
        resized_w = int(round(src_w * gain))
        pad_top, pad_left = letterbox.pad

        resized = cv2.resize(bgr_frame, (resized_w, resized_h),
                             interpolation=cv2.INTER_LINEAR)

        padded = numpy.full((model_h, model_w, bgr_frame.shape[2]),
                            self._pad_value, dtype=numpy.uint8)
        padded[int(pad_top):resized_h + int(pad_top),
               int(pad_left):resized_w + int(pad_left)] = resized
        return padded

    def _quantize_input(self, rgb_or_bgr_u8):
        """Scale a uint8 image to the model input's dtype and quantization.

        The float path normalizes to [0, 1]; the integer path applies the
        tensor's own (scale, zero_point). For the common uint8 input with
        scale=1/255, zero_point=0 this is the identity on the raw pixels.
        """
        dtype = self._input_detail["dtype"]

        if dtype == numpy.float32:
            return (rgb_or_bgr_u8.astype(numpy.float32) / 255.0)[numpy.newaxis]

        scale, zero_point = self._input_quant
        if not scale:
            # Not quantized despite an integer dtype: pass the pixels through.
            return rgb_or_bgr_u8.astype(dtype)[numpy.newaxis]

        normalized = rgb_or_bgr_u8.astype(numpy.float32) / 255.0
        quantized = numpy.round(normalized / scale) + zero_point
        info = numpy.iinfo(dtype)
        return numpy.clip(quantized, info.min, info.max).astype(dtype)[numpy.newaxis]

    def preprocess(self, bgr_frame):
        t0 = time.perf_counter()
        policy = self.task.preprocess_policy()

        image = self._resize_to_input(bgr_frame, self._letterbox_for(bgr_frame.shape[:2]))
        if policy.bgr_to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor = self._quantize_input(image)

        t1 = time.perf_counter()
        return tensor, {"preprocess_ms": (t1 - t0) * 1000.0}

    def infer(self, prepared_input):
        t0 = time.perf_counter()
        self._interp.set_tensor(self._input_detail["index"], prepared_input)
        self._interp.invoke()
        raw_tensors = [
            self._interp.get_tensor(d["index"]) for d in self._output_details
        ]
        t1 = time.perf_counter()

        tensors = []
        for tensor, (scale, zero_point) in zip(raw_tensors, self._output_quant):
            values = tensor.astype(numpy.float32)
            if scale:
                values = (values - zero_point) * scale
            tensors.append(values)
        t2 = time.perf_counter()

        return RawOutputs(tensors), {"infer_ms": (t1 - t0) * 1000.0,
                                     "dequantize_ms": (t2 - t1) * 1000.0}

    def postprocess(self, raw_outputs, src_hw):
        t0 = time.perf_counter()
        tensors = raw_outputs.tensors
        if self.task.NAME == "detection":
            tensors = yolov8_layout.normalize_outputs(
                tensors, self._layout, self._num_classes, self._model_hw
            )
        result = self.task.decode(tensors, self._letterbox_for(src_hw), src_hw)
        t1 = time.perf_counter()
        return result, {"nms_ms": (t1 - t0) * 1000.0}




class CpuEngine(TFLiteEngine):
    """Plain CPU inference (TFLite + XNNPACK, multi-threaded)."""
    NAME = "CPU"
    DELEGATE_LIB = None


class GpuEngine(TFLiteEngine):
    """i.MX Vivante GPU inference (TFLite + vx_delegate).

    vx_delegate accelerates portable models but cannot execute a graph
    compiled for the Neutron NPU, so it stays on the portable variants.
    """
    NAME = "GPU"
    DELEGATE_LIB = "libvx_delegate.so"


class NeutronEngine(TFLiteEngine):
    """i.MX Neutron NPU inference (TFLite + external delegate).

    The only backend implementing the NeutronGraph custom op, so it is also
    the only one that prefers the SoC-tagged variant of a model.
    """
    NAME = "NEUTRON"
    DELEGATE_LIB = "libneutron_delegate.so"
    RUNS_COMPILED_MODELS = True
