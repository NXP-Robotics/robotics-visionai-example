# robotics-visionai-example

Vision AI web viewer for the NXP MR-NavQ95 board: it captures from the
attached camera, runs inference, and streams the annotated video as MJPEG
plus JSON stats to a browser.

Backends (`--backend`) and model families (`--task`) are independent choices:

| Backend   | Hardware                                    | Available models today          |
|-----------|---------------------------------------------|---------------------------------|
| `ARA`     | NXP ARA-240 accelerator (via `dvapi`)       | YOLOv8 **detection** (`yolov8n/s/m/l/x`) |
| `NEUTRON` | i.MX Neutron NPU (TFLite external delegate) | YOLOv8 **detection**, MoveNet **pose** |
| `GPU`     | i.MX Vivante GPU (TFLite `vx_delegate`)     | whichever portable `.tflite` models are installed |
| `CPU`     | plain TFLite + XNNPACK                      | whichever portable `.tflite` models are installed |

Which models a TFLite backend offers is discovered from what is installed on
the board (see section 2), not hardcoded: `GPU`/`CPU` are limited to models
that do not need an accelerator delegate.


This README assumes the repository is already present on the board and all
commands below are run on the board, from the repository root.

## 1. Set up the virtualenv

```bash
uv venv --system-site-packages
uv sync
uv run robotics-visionai-example --help
```

`--system-site-packages` is required in both cases: the capture pipeline uses
PyGObject (`gi`) and the `imx*` GStreamer plugins.

### Checking the environment

If anything fails to start, run the built-in preflight check first. It reports
what is missing and the exact command to fix each item:

```bash
robotics-visionai-example --doctor
```

## 2. Set up the models

### ARA (`.dvm`) - on the board

The YOLOv8 detection models are downloaded directly on the board with
`fetch_models` (requires internet access):

```bash
sudo fetch_models --repo-id nxp/YOLOv8
```

If the models are not available or you want to use a different yolo model, use the
ARA compiler from the ARA SDK to compile your own `.dvm` files. See https://www.nxp.com/design/design-center/software/embedded-software/ara-software-development-kit:ARA-SDK for details.

This installs them under `/usr/share/cnn/<task>/<name>/model.dvm`.

### Neutron (`.tflite`) - compiled on a host

TFLite models have to be compiled for the NPU on a host machine:

1. On the host, clone the NXP NNStreamer examples repo:
   <https://github.com/nxp-imx/nxp-nnstreamer-examples>
2. Follow the instructions in that repo's `downloads/README.md` to download
   and compile the models.
3. Copy the resulting `.tflite` files to the board into
   `/usr/share/cnn/neutron/<category>/`, using `pose-estimation/` for pose
   models and `object-detection/` for detection models.

Keep the filenames unchanged. Models compiled for a specific SoC carry it in
the filename (e.g. `movenet_quant_imx95.tflite`,
`yolov8n-det-int8.imx95.tflite`), and the app picks the variant matching
`/sys/devices/soc0/soc_id`, so a model can be named without its suffix.

That suffix is also how the app knows a model needs an accelerator delegate:
such a file contains a custom op (`NeutronGraph`) that plain TFLite cannot
resolve, so it only runs with `--backend NEUTRON`. Untagged files are portable
and also run on `--backend CPU` / `GPU`.

For detection, only **YOLOv8** models are supported.

#### Accepted YOLOv8 output layouts

You do not have to match a particular export recipe. The app inspects the
model's output tensor shapes and picks the decode path itself, so any of the
three layouts a documented YOLOv8 export flow produces will work — including a
model you compiled yourself:

| Outputs | Shapes | Produced by |
| --- | --- | --- |
| 1 | `(1, 4+nc, N)` or `(1, N, 4+nc)` | Stock Ultralytics: `yolo export format=onnx\|tflite`, carried through the eIQ Neutron flow (`onnx2tflite` → `tflite-quantizer` → `neutron-converter`) |
| 3 | `(1, H, W, 4*16+nc)` for strides 8/16/32 | The head truncated before its DFL decode tail. The tail is `Softmax`/`Transpose`/`Split`, which Neutron-S cannot execute, so removing it is what lets the **whole** graph run on the NPU instead of being partitioned. The app does the DFL decode on the host. |
| 2 | `(1, 4, N)` + `(1, nc, N)` | The ARA-240 `.dvm` convention, and Ultralytics' `format="imx"` export |

`nc` is the class count, read from the tensor shape rather than assumed to be
80, so a model retrained on your own classes works unchanged. Box coordinates
may be either normalised to 0..1 (as the `format=tflite` head emits) or in
model-input pixels; both are handled.

> **Note on quantizing the single-tensor layout:** if you quantize a
> `(1, 4+nc, N)` model, export it with Ultralytics' **tflite** head so the box
> coordinates are normalised to 0..1. With pixel coordinates the one
> per-tensor output scale has to cover 0..640 for the 4 box rows while the
> score rows only span 0..1, which collapses every class score onto two
> representable levels and the model detects nothing. The three-tensor layout
> is immune, since boxes are still undecoded logits there.



### Expected layout on the board

```
/usr/share/cnn/
├── detection
│   ├── yolov8l
│   │   ├── LICENSE.txt
│   │   ├── model.dvm
│   │   └── README.md
│   ├── yolov8m
│   │   ├── LICENSE.txt
│   │   ├── model.dvm
│   │   └── README.md
│   ├── yolov8n
│   │   ├── LICENSE.txt
│   │   ├── model.dvm
│   │   └── README.md
│   ├── yolov8s
│   │   ├── LICENSE.txt
│   │   ├── model.dvm
│   │   └── README.md
│   └── yolov8x
│       ├── LICENSE.txt
│       ├── model.dvm
│       └── README.md
├── neutron
    ├── object-detection
    │   ├── yolov8n-det-int8.imx95.tflite
    │   ├── yolov8s-det-int8.imx95.tflite
    │   └── yolov8m-det-int8.imx95.tflite
    ├── pose-estimation
        ├── movenet_quant_imx952.tflite
        ├── movenet_quant_imx95.tflite
        └── movenet_quant.tflite
```


## 3. Run the application

When installed from the image the command is on `$PATH`:

```bash
# ARA backend, YOLOv8 object detection
robotics-visionai-example --backend ARA --task detection --ara-model detection/yolov8s

# Neutron NPU backend, MoveNet pose estimation
robotics-visionai-example --backend NEUTRON --task pose

# Neutron NPU backend, YOLOv8 object detection
robotics-visionai-example --backend NEUTRON --task detection

# ...or a specific YOLOv8 model (the per-SoC suffix may be omitted)
robotics-visionai-example --backend NEUTRON --task detection \
  --model_path object-detection/yolov8s-det-int8
```

With no `--model_path`, a model installed for `--task` is selected
automatically.


From a source checkout, prefix the same commands with `uv run`:

```bash
uv run robotics-visionai-example --backend NEUTRON --task pose
```

Every CLI flag also has an equivalent environment variable:

```bash
BACKEND=ARA TASK=detection ARA_MODEL=detection/yolov8s robotics-visionai-example
```

Useful flags:

| Flag                    | Purpose                                                          |
|-------------------------|------------------------------------------------------------------|
| `--backend`             | `ARA`, `NEUTRON`, `GPU` or `CPU` (default `CPU`)                 |
| `--task`                | `detection` or `pose` (defaults to `detection` for ARA, `pose` otherwise) |
| `--ara-model`           | ARA model as `<task>/<name>`, e.g. `detection/yolov8m`           |
| `--models-dir`          | ARA models base dir (default `/usr/share/cnn`)                   |
| `--model_path`          | Model for NEUTRON/GPU/CPU: a `.tflite` path, `<category>/<name>`, or a bare name (per-SoC suffix optional) |
| `--neutron-models-dir`  | TFLite models base dir (default `/usr/share/cnn/neutron`)        |
| `--camera-type` / `-t`  | Camera sensor, or `auto` (default) - see below                    |
| `--camera-index`        | Which camera to open when several are attached                   |
| `--width` / `--height`  | Capture resolution (default 640x480)                             |
| `--port`                | HTTP port (default 5010)                                         |
| `--verbose` / `-v`      | Debug logging                                                    |
| `--doctor`              | Check the environment and exit (see section 1)                   |

Run `robotics-visionai-example --help` for the full list.

## 4. Open the viewer in a browser

Browse to `http://<board-ip>:5010`.

The page shows the live annotated stream, FPS and latency stats, a per-stage
timing breakdown, and a model dropdown that hot-swaps the active model without
restarting the stream. The dropdown lists the models installed for the running
backend and task, so it works the same for `ARA` (`.dvm`) and for the TFLite
backends (`.tflite`); only capture keeps running while the model reloads.


## 5. HTTP endpoints

| Method | Path           | Description                                                                 |
|--------|----------------|-----------------------------------------------------------------------------|
| GET    | `/`            | Web viewer UI                                                               |
| GET    | `/stream`      | MJPEG stream of annotated frames                                            |
| GET    | `/status`      | JSON: fps, invoke time, backend, camera, active model, detections/keypoints, per-stage timings |
| GET    | `/api/options` | JSON: models for the UI selector (for the running backend and task) plus the detected cameras |
| POST   | `/api/model`   | Hot-swap the model, body `{"model": "<one of /api/options models>"}`        |
| POST   | `/api/camera`  | Switch the active camera, body `{"camera_id": "<id from /api/options>"}`    |
| GET    | `/docs`        | FastAPI interactive API docs                                                |

```bash
curl -s http://<board-ip>:5010/status | jq
curl -s -X POST http://<board-ip>:5010/api/model \
  -H 'Content-Type: application/json' -d '{"model":"detection/yolov8m"}'

# Switch camera: list the attached ones, then select one by its camera_id
curl -s http://<board-ip>:5010/api/options | jq '.detected_cameras'
curl -s -X POST http://<board-ip>:5010/api/camera \
  -H 'Content-Type: application/json' \
  -d '{"camera_id":"/base/soc/i2c@42530000/i2c-mux@77/i2c@0/imx708@1a"}'
```

## 6. Camera selection

Three camera sensors are supported, each needing its own `libcamerasrc` caps:

| `--camera-type` | Sensor caps negotiated after `libcamerasrc`                        |
|-----------------|-------------------------------------------------------------------|
| `imx219`        | `video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1`    |
| `imx708`        | `video/x-raw,format=NV12` (libcamera picks the native mode)        |
| `os08a20`       | `video/x-raw,format=NV12` (libcamera picks the native mode)        |

`--camera-type auto` (the default, or `CAMERA_TYPE=auto`) asks `cam -l` which
camera is attached and picks the matching profile. It does **not** guess: if
`cam -l` reports nothing, or a sensor that is not one of the three above,
startup fails with an error listing the valid values rather than silently
running a pipeline that cannot negotiate.

### Switching camera at runtime

When more than one camera is attached, the web UI shows a **camera dropdown**
next to the backend label, listing each attached camera as `<sensor> (<bus id>)`
-- for example `imx708 (imx708@1a)`. Selecting one rebuilds the capture
pipeline with **that sensor's caps**, so no restart and no `--camera-type` flag
is needed to move between cameras of different types. Cameras whose sensor is
not one of the three supported profiles are not listed, since there is no
pipeline that could be built for them.

The engine and the active model are untouched by a camera switch.

```bash
# Explicit sensor (skips detection entirely)
robotics-visionai-example --camera-type os08a20

# Two cameras attached: pick the second one from `cam -l`'s 1-based numbering
robotics-visionai-example --camera-index 2
```

Without `--camera-index`, `libcamerasrc` opens its default (the first)
camera. The app exports `LIBCAMERA_PIPELINES_MATCH_LIST='nxp/neo,imx8-isi,uvcvideo'`
itself (required for all three cameras), so no manual `export` is needed;
setting it in the environment still overrides the built-in default.
