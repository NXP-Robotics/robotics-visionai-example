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
import sys
import time

import cv2
import numpy

from robotics_visionai_example.common import yolov8_layout
from robotics_visionai_example.common.docs import readme_hint
from robotics_visionai_example.common.letterbox import LetterboxInfo, preprocess_frame

from robotics_visionai_example.engines.base import InferenceEngine, RawOutputs

logger = logging.getLogger(__name__)

# dvapi ships with the on-device ARA-240 SDK, so it is imported lazily via
# _get_dvapi() to keep this module importable on a host without the SDK.
_dvapi = None


def _get_dvapi():
    global _dvapi
    if _dvapi is not None:
        return _dvapi

    sdk_include = os.environ.get(
        "RT_SDK_INCLUDE", "/usr/share/rt-sdk-ara240_2.1.1/include"
    )
    if sdk_include and sdk_include not in sys.path:
        sys.path.insert(0, sdk_include)

    import dvapi as _dvapi_mod
    _dvapi = _dvapi_mod
    return _dvapi


ARA_CNN_DIR = os.environ.get("ARA_CNN_DIR", "/usr/share/cnn")
ARA_CNN_CATEGORIES = ("detection", "pose")

# YOLOv8 letterbox padding value.
DEFAULT_PAD_VALUE = 114


def resolve_model_path(model, models_dir=None):
    """Resolve a model.dvm path, accepting (in order) a direct .dvm path, a
    directory containing model.dvm, "<task>/<name>" under models_dir, or a
    bare model name searched across ARA_CNN_CATEGORIES.
    """
    models_dir = models_dir or ARA_CNN_DIR

    if model.endswith(".dvm") and os.path.isfile(model):
        return os.path.abspath(model)

    for candidate in (os.path.join(model, "model.dvm"),
                      os.path.join(models_dir, model, "model.dvm")):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    for category in ARA_CNN_CATEGORIES:
        candidate = os.path.join(models_dir, category, model, "model.dvm")
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    available = list_models(models_dir)
    if available:
        raise FileNotFoundError(
            "ARA model '{}' not found under {}. Installed models: {}. {}".format(
                model, models_dir, ", ".join(available),
                readme_hint("2. Set up the models"),
            )
        )

    raise FileNotFoundError(
        "ARA model '{}' not found and no models are installed under {}. {}".format(
            model, models_dir, readme_hint("2. Set up the models")
        )
    )


def list_models(models_dir=None, tasks=None):
    """List "<task>/<name>" models installed under models_dir, restricted to
    the `tasks` category subfolders (default: all ARA_CNN_CATEGORIES).
    """
    models_dir = models_dir or ARA_CNN_DIR
    tasks = tasks if tasks is not None else ARA_CNN_CATEGORIES

    found = []
    if not os.path.isdir(models_dir):
        return found
    for category in tasks:
        category_dir = os.path.join(models_dir, category)
        if not os.path.isdir(category_dir):
            continue
        for name in sorted(os.listdir(category_dir)):
            if os.path.isfile(os.path.join(category_dir, name, "model.dvm")):
                found.append("{}/{}".format(category, name))
    return found


ARA_PROXY_SOCK = os.environ.get("ARA_PROXY_SOCK", "/var/run/proxy.sock")

# Shared-memory buffers let the app and the out-of-process ARA proxy daemon
# exchange tensors via mmap'd pages instead of copying them over the socket.
ARA_SHM_FILE = os.environ.get(
    "ARA_SHM_FILE", "/dev/shm/robotics-visionai-example-ara.shm"
)
ARA_SHM_SIZE = 256 * 1024 * 1024  # large enough for any YOLOv8 tensor


class NNApplication:
    """A dvapi session with one loaded model.dvm and its input/output tensors.

    All model parameters (shapes, quantization) are read from the loaded
    model.dvm itself, so there is no per-model config file.
    """
    def __init__(self, model_path, sock=None, shm_file=None):
        self.model_path = model_path
        self.sock = sock or ARA_PROXY_SOCK
        self.shm_file = ARA_SHM_FILE if shm_file is None else shm_file
        self.session = None
        self.model = None
        self.endpoint = None
        self.shmfile = None
        self.shmfile_size = ARA_SHM_SIZE
        self.dv_shm_desc = None
        self.input_list = []
        self.output_list = []

    def _get_aligned_size(self, unaligned_size, pagesize):
        mod = unaligned_size % pagesize
        if mod == 0:
            return unaligned_size
        return unaligned_size + pagesize - mod

    def init(self, exit_on_error=True):
        """Open the hardware session and load self.model_path.

        exit_on_error: exit the process on unrecoverable dvapi errors when
        True. reload() passes False so a bad model swap raises RuntimeError
        instead of killing a long-running server.
        """
        dvapi = _get_dvapi()
        logger.info("loading model: %s", self.model_path)

        # Drop any tensors from a previous load so a reload never mixes
        # buffers sized for the old model with the new one.
        self.input_list = []
        self.output_list = []

        def _fail(message):
            logger.error(message)
            if exit_on_error:
                sys.exit(1)
            raise RuntimeError(message)

        if ":" in self.sock:
            args = self.sock.split(":")
            ret, self.session = dvapi.DVSession.create_via_tcp_ipv4_socket(args[0], int(args[1]))
        else:
            ret, self.session = dvapi.DVSession.create_via_unix_socket(self.sock)

        if self.shm_file:
            try:
                self.shmfile = open(self.shm_file, "w+")
                self.shmfile.truncate(self.shmfile_size)
                self.shmfile.seek(0, 0)
                status, self.dv_shm_desc = self.session.register_shmfd(
                    self.shmfile.fileno(), self.shmfile_size, 0
                )
                if status != dvapi.dv_status_code.DV_SUCCESS:
                    logger.warning(
                        "failed to register shared memory file %s (status %s); "
                        "falling back to local (non-shared) buffers",
                        self.shm_file, status,
                    )
                    self.shmfile.close()
                    self.shmfile = None
                    self.dv_shm_desc = None
            except Exception as exc:
                logger.warning(
                    "shared memory setup failed for %s (%s); "
                    "falling back to local (non-shared) buffers",
                    self.shm_file, exc,
                )
                if self.shmfile is not None:
                    try:
                        self.shmfile.close()
                    except OSError:
                        pass
                self.shmfile = None
                self.dv_shm_desc = None

        ret, endpoints = self.session.get_endpoint_list()

        if ret != dvapi.dv_status_code.DV_SUCCESS or len(endpoints) == 0:
            _fail("could not list endpoints or no endpoint is connected to server")

        self.endpoint = endpoints[0]

        ret, self.model = self.session.load_model_from_file(self.endpoint, self.model_path)
        if ret != dvapi.dv_status_code.DV_SUCCESS:
            _fail("error in loading model: {}".format(ret))

        offset = 0
        pagesize = os.sysconf("SC_PAGE_SIZE")

        for i in range(self.model.num_inputs):
            if self.dv_shm_desc is not None:
                shared_ptr = self.dv_shm_desc.at_offset_with_size(offset, self.model.input_param[i].size)
                self.input_list.append(dvapi.DVTensor(shared_ptr, self.model.input_param[i], self.dv_shm_desc, offset))
                offset += self._get_aligned_size(self.model.input_param[i].size, pagesize)
            else:
                input_buf = numpy.zeros(self.model.input_param[i].size, dtype=numpy.int8)
                self.input_list.append(dvapi.DVTensor(input_buf, self.model.input_param[i]))

        for i in range(self.model.num_outputs):
            if self.dv_shm_desc is not None:
                shared_ptr = self.dv_shm_desc.at_offset_with_size(offset, self.model.output_param[i].size)
                self.output_list.append(dvapi.DVTensor(shared_ptr, self.model.output_param[i], self.dv_shm_desc, offset))
                offset += self._get_aligned_size(self.model.output_param[i].size, pagesize)
            else:
                output_buf = numpy.zeros(self.model.output_param[i].size, dtype=numpy.int8)
                self.output_list.append(dvapi.DVTensor(output_buf, self.model.output_param[i]))

    def infer(self, image_data):
        dvapi = _get_dvapi()

        self.input_list[0].numpy_data[:] = image_data
        status, resp = self.model.infer_async(
            input_tensors=self.input_list, output_tensors=self.output_list
        )
        if status != dvapi.dv_status_code.DV_SUCCESS:
            logger.error("could not initiate inference - error %s", status)
            sys.exit(1)

        status = resp.wait_for_completion(timeout=3600000)
        if status != dvapi.dv_status_code.DV_SUCCESS:
            logger.error("wait for request completion failed, %s", status)
            os._exit(1)

        return self.output_list

    def read_input_params(self, index=0):
        return self.model.input_param[index]

    def close(self):
        self.model.unload()
        if self.dv_shm_desc is not None:
            self.dv_shm_desc.unregister()
            self.shmfile.close()
        self.session.close()


def preprocess_cfg_from_input_param(input_param, pad_value=DEFAULT_PAD_VALUE):
    """Build a preprocess_frame() config dict from a loaded model's dvapi
    DVModelInputParam.
    """
    pm = input_param.preprocess_param
    scale = list(pm.scale) if pm.scale else []
    # replicate to 3 channels so preprocess_frame's r/g/b lookups work
    while len(scale) < 3:
        scale.append(scale[-1] if scale else 1.0)

    return {
        "input_shape": {
            "c": input_param.nch,
            "h": input_param.height,
            "w": input_param.width,
        },
        "padding": {"color_val": pad_value},
        "scale": {"r": scale[0], "g": scale[1], "b": scale[2]},
        "bgr_to_rgb": bool(pm.bgr_to_rgb),
    }


def quantize(image, ip_param, qmode=0):
    """Quantize a float32 CHW image to the model's integer input dtype."""
    pm = ip_param.preprocess_param
    flat = image.reshape(-1)   # a view, unlike flatten()

    if qmode == 9:
        scaled = numpy.floor(flat * (1.0 / pm.qn) + pm.offset + 0.5)
    else:
        scaled = numpy.floor(flat * (float(pm.qn) * float(pm.output_scale)) + 0.5)

    if pm.bpp == 1:
        lo, hi, dtype = (-128, 127, numpy.int8) if pm.is_signed else (0, 255, numpy.int8)
    else:
        lo, hi, dtype = ((-32768, 32767, numpy.int16) if pm.is_signed
                         else (0, 65535, numpy.int16))
    return scaled.clip(lo, hi).astype(dtype)


_DEQUANT_DTYPES = {
    (1, True): numpy.int8,   (1, False): numpy.uint8,
    (2, True): numpy.int16,  (2, False): numpy.uint16,
    (4, True): numpy.int32,  (4, False): numpy.uint32,
}


def dequantize(output_list, qmode=0):
    """Convert the model's integer output tensors to float32 arrays.

    Returns independent copies (via numpy.frombuffer + astype), so the shared
    output buffers can be safely overwritten by the next inference.
    """
    dequantized_outputs = []
    for tensor in output_list:
        raw_nd = tensor.numpy_data
        out_param = tensor.params
        pp = out_param.postprocess_param

        if pp.is_struct_format or pp.is_float:
            dequantized_outputs.append(numpy.asarray(raw_nd))
            continue

        intype = _DEQUANT_DTYPES.get((out_param.bpp, bool(pp.is_signed)), numpy.int8)
        arr = numpy.frombuffer(raw_nd, dtype=intype).astype(numpy.float32)
        if qmode == 9:
            arr -= pp.offset
            arr *= pp.qn
        else:
            arr += pp.offset
            arr /= (pp.qn * pp.output_scale)
        dequantized_outputs.append(arr)
    return dequantized_outputs


class AraEngine(InferenceEngine):
    """NXP ARA-240 accelerator inference engine.

    Only YOLOv8 detection models have a decode path today: tasks/pose.py
    decodes the MoveNet tensor layout, not the ARA YOLOv8-pose head.
    """
    NAME = "ARA"
    SUPPORTED_TASKS = ("detection",)

    def __init__(self, task, model="detection/yolov8s", models_dir=None,
                 sock=None, shm_file=None, pad_value=DEFAULT_PAD_VALUE):
        super().__init__(task, model, models_dir)
        self.sock = sock
        self.shm_file = shm_file
        self._pad_value = pad_value

        self._nnapp = None
        self._preprocess_cfg = None
        self._model_input_shape = None   # [c, h, w]
        self._qmode = None
        self._reset_fused()

    @classmethod
    def list_models(cls, models_dir=None, tasks=None):
        return list_models(models_dir, tasks if tasks is not None else cls.SUPPORTED_TASKS)

    def _reset_fused(self):
        """Clear all per-model fused fast-path state (see _setup_fused)."""
        self._fused_ready = False
        self._fused_combined = None   # scale * (1/qn) applied to uint8
        self._fused_offset = None
        self._fused_clip_lo = None
        self._fused_clip_hi = None
        self._fused_intype = None
        self._fused_bgr_to_rgb = None
        self._fused_pad_q = None
        self._fused_out_buf = None
        self._fused_out_region = None   # (top, rh, left, rw) last filled

    def _load_model(self, exit_on_error=True):
        """Open a session for self.model_name and derive its preprocess params."""
        model_path = resolve_model_path(self.model_name, self.models_dir)

        self._nnapp = NNApplication(model_path, sock=self.sock, shm_file=self.shm_file)
        self._nnapp.init(exit_on_error=exit_on_error)

        input_param = self._nnapp.read_input_params()
        self._qmode = input_param.preprocess_param.qmode
        self._preprocess_cfg = preprocess_cfg_from_input_param(input_param, self._pad_value)
        pp = self._preprocess_cfg["input_shape"]
        self._model_input_shape = [pp["c"], pp["h"], pp["w"]]
        self._setup_fused()

        logger.info("AraEngine model=%s path=%s qmode=%s input=%s fused=%s",
                    self.model_name, model_path, self._qmode,
                    self._model_input_shape, self._fused_ready)

    def init(self):
        self._load_model()

    def _setup_fused(self):
        """Precompute the constants for the fused preprocess+quantize path.

        For a zero-mean, uniform-scale, qmode==9, 8-bit model the separate
        scale and quantize passes collapse into one linear map on uint8:
            int8 = clip(floor(pixel_u8 * (scale/qn) + offset + 0.5))
        Any other model configuration keeps the generic float32 path.
        """
        cfg = self._preprocess_cfg
        pm = self._nnapp.read_input_params().preprocess_param
        scale = cfg.get("scale", {})
        sr, sg, sb = scale.get("r"), scale.get("g"), scale.get("b")
        mean = cfg.get("mean", {})
        mean_zero = all(float(mean.get(k, 0)) == 0 for k in ("r", "g", "b"))
        uniform = sr and sg and sb and sr == sg == sb
        if not (self._qmode == 9 and uniform and mean_zero and pm.bpp == 1):
            self._fused_ready = False
            return

        # float32 scalars, so the fused pass does not promote the uint8 image
        # to float64 and double its memory traffic
        self._fused_combined = numpy.float32(float(sr) * (1.0 / pm.qn))
        self._fused_offset = numpy.float32(float(pm.offset) + 0.5)
        if pm.is_signed:
            self._fused_clip_lo, self._fused_clip_hi = -128, 127
        else:
            self._fused_clip_lo, self._fused_clip_hi = 0, 255
        self._fused_intype = numpy.int8
        self._fused_bgr_to_rgb = bool(cfg.get("bgr_to_rgb", False))

        # pre-quantize the pad value so the letterbox borders can be filled
        # once instead of being recomputed per frame
        pad_val = float(cfg.get("padding", {}).get("color_val", DEFAULT_PAD_VALUE))
        pv = numpy.floor(pad_val * float(self._fused_combined)
                         + float(self._fused_offset))
        self._fused_pad_q = numpy.int8(
            int(numpy.clip(pv, self._fused_clip_lo, self._fused_clip_hi))
        )
        self._fused_out_buf = None
        self._fused_out_region = None
        self._fused_ready = True

    def _fused_preprocess_quantize(self, bgr_frame):
        """Fused BGR -> resize/pad -> RGB -> CHW -> scale -> int8 quantize.

        Returns a view of a reusable contiguous int8 (C*H*W,) buffer. Stays
        uint8 through resize/transpose and does a single arithmetic pass, and
        is numerically identical to preprocess_frame() + quantize() for the
        model configuration _setup_fused() accepts.
        """
        c, h, w = self._model_input_shape
        img_h, img_w = bgr_frame.shape[:2]

        ratio = min(h / img_h, w / img_w)
        rh = int(round(img_h * ratio))
        rw = int(round(img_w * ratio))
        if (rh, rw) != (img_h, img_w):
            img_u8 = cv2.resize(bgr_frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        else:
            img_u8 = bgr_frame

        top = int(round((h - rh) / 2 - 0.1))
        left = int(round((w - rw) / 2 - 0.1))

        # The letterbox borders are constant for a fixed input size, so fill
        # the whole buffer with the pre-quantized pad value only when the
        # content region changes, and overwrite just that region per frame.
        region = (top, rh, left, rw)
        if (self._fused_out_buf is None
                or self._fused_out_buf.shape != (c, h, w)
                or self._fused_out_region != region):
            self._fused_out_buf = numpy.empty((c, h, w), dtype=self._fused_intype)
            self._fused_out_buf[:] = self._fused_pad_q
            self._fused_out_region = region
        out_buf = self._fused_out_buf

        if self._fused_bgr_to_rgb:
            content_hwc = numpy.ascontiguousarray(img_u8[:, :, ::-1])
        else:
            content_hwc = numpy.ascontiguousarray(img_u8)
        content_chw = content_hwc.transpose(2, 0, 1)

        out = numpy.multiply(content_chw, self._fused_combined, dtype=numpy.float32)

        out += self._fused_offset
        numpy.floor(out, out=out)
        numpy.clip(out, self._fused_clip_lo, self._fused_clip_hi, out=out)
        out_buf[:, top:rh + top, left:rw + left] = out.astype(self._fused_intype)

        return out_buf.reshape(-1)

    def _letterbox_for(self, src_hw):
        model_hw = (self._model_input_shape[1], self._model_input_shape[2])
        return LetterboxInfo.letterboxed(model_hw, src_hw)

    # -- InferenceEngine 3-stage API ---------------------------------------

    def preprocess(self, bgr_frame):
        """Stage 1: BGR frame -> int8 CHW model input (preprocess + quantize).

        The returned array is always independent of any reusable internal
        buffer, so it stays valid while the next frame is being preprocessed.
        """
        t0 = time.perf_counter()
        if self._fused_ready:
            int8_input = self._fused_preprocess_quantize(bgr_frame).copy()
            t1 = time.perf_counter()
            return int8_input, {"preprocess_ms": (t1 - t0) * 1000.0, "quantize_ms": 0.0}

        fp32_input = preprocess_frame(bgr_frame, self._preprocess_cfg)
        t1 = time.perf_counter()
        int8_input = quantize(fp32_input, self._nnapp.read_input_params(), self._qmode)
        t2 = time.perf_counter()
        return int8_input, {"preprocess_ms": (t1 - t0) * 1000.0,
                            "quantize_ms": (t2 - t1) * 1000.0}

    def infer(self, prepared_input):
        """Stage 2: int8 input -> dequantized outputs.

        Dequantizes inside this stage so the shared output buffer is consumed
        before it can be overwritten, keeping all access to the shared tensors
        on the single inference thread.
        """
        t0 = time.perf_counter()
        raw_output = self._nnapp.infer(prepared_input)
        t1 = time.perf_counter()
        deq_output = dequantize(raw_output, self._qmode)
        t2 = time.perf_counter()
        output_shapes = [(t.params.nch, t.params.height) for t in raw_output]
        raw = RawOutputs([deq_output[1], deq_output[0]], output_shapes=output_shapes)
        return raw, {"infer_ms": (t1 - t0) * 1000.0, "dequantize_ms": (t2 - t1) * 1000.0}

    def postprocess(self, raw_outputs, src_hw):
        """Stage 3: dequantized outputs -> task Result."""
        t0 = time.perf_counter()
        (nch1, h1), (nch0, h0) = raw_outputs.output_shapes[1], raw_outputs.output_shapes[0]
        tensors = [
            raw_outputs.tensors[0].reshape(1, nch1, h1),
            raw_outputs.tensors[1].reshape(1, nch0, h0),
        ]
        model_hw = (self._model_input_shape[1], self._model_input_shape[2])
        tensors = yolov8_layout.normalize_outputs(
            tensors, yolov8_layout.LAYOUT_SPLIT, tensors[1].shape[1], model_hw
        )
        result = self.task.decode(tensors, self._letterbox_for(src_hw), src_hw)
        t1 = time.perf_counter()
        return result, {"nms_ms": (t1 - t0) * 1000.0}


    def reload(self, model):
        """Hot-swap to a different model, rolling back on failure.

        Not thread-safe: callers running the threaded pipeline in
        app/stream.py must park the worker threads around this call.
        """
        previous_model_name = self.model_name

        # Close the old session first so the proxy daemon frees its
        # shared-memory tensors before new ones are allocated.
        if self._nnapp is not None:
            try:
                self._nnapp.close()
            except Exception as exc:
                logger.warning("error closing previous ARA session: %s", exc)

        self._nnapp = None
        self.model_name = model
        self._preprocess_cfg = None
        self._model_input_shape = None
        self._qmode = None
        self._reset_fused()

        try:
            self._load_model(exit_on_error=False)
        except Exception:
            logger.error("reload to model '%s' failed; rolling back to '%s'",
                         model, previous_model_name)
            self.model_name = previous_model_name
            self._nnapp = None
            raise

    def close(self):
        if self._nnapp is not None:
            self._nnapp.close()
            self._nnapp = None
            logger.info("AraEngine closed")
