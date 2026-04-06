# OpenVINO Vision

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Real-time AI inference library for classroom analytics using [Intel OpenVINO](https://docs.openvino.ai/). Detects student actions (sitting, standing, raising hand), recognizes faces, and tracks people across video frames.

Dual-language architecture: C++ provides the core inference engine (`libovi_vision`); Rust provides safe idiomatic bindings via CXX bridge with a builder-pattern API.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Building](#building)
- [Model Setup](#model-setup)
- [Face Gallery Setup](#face-gallery-setup)
- [Rust API](#rust-api)
- [Running Examples](#running-examples)
- [Testing](#testing)
- [Command-Line Reference](#command-line-reference)

## Prerequisites

- **OpenVINO** 2025.0.0+
- **OpenCV** 4.x (core, imgproc, highgui, videoio, imgcodecs)
- **CMake** 3.10+
- **C++17** compatible compiler
- **Rust** (stable toolchain) — for Rust crates and tools
- **gflags** — only required for the C++ example application

### macOS (Homebrew)

```bash
brew install opencv openvino gflags cmake
```

### Model Download Tools (openvino-dev)

Models are downloaded using Intel's [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) CLI tools, which are part of the `openvino-dev` Python package.

```bash
pip install openvino-dev[onnx,tensorflow]
```

This installs two key tools:

- **`omz_downloader`** — Downloads pre-trained models from Intel's Open Model Zoo catalog
- **`omz_converter`** — Converts downloaded models to OpenVINO IR format (`.xml` + `.bin`) for inference

**Download all models listed in `models.lst`:**

```bash
omz_downloader --list models.lst -o assets/
```

**Convert downloaded models to OpenVINO IR format:**

```bash
omz_converter --list models.lst -o assets/
```

You can also download individual models by name:

```bash
# Download a specific model
omz_downloader --name face-detection-adas-0001 -o assets/

# Convert it to IR format
omz_converter --name face-detection-adas-0001 -o assets/

# List all available models
omz_downloader --print_all
```

After downloading and converting, models will be available under `assets/intel/` (Intel models) and `assets/public/` (public models) in FP32, FP16, and FP16-INT8 precision variants.

## Building

### Rust (recommended)

The project uses a Cargo workspace. The `ovi-vision-sys` crate's `build.rs` invokes CMake internally to compile the C++ library and FFI bridge.

```bash
cargo build                              # Build default members (crates + tools)
cargo build -p ovi-vision-sys       # Build raw CXX FFI bindings
cargo build -p ovi-vision           # Build safe Rust wrapper
cargo build -p smart-classroom-rs        # Build Rust example
```

On macOS with Homebrew, OpenCV/OpenVINO paths are auto-detected via `brew --prefix`. On other systems, set environment variables or CMake flags as needed.

### C++ Library (standalone)

```bash
cd cpp && mkdir build && cd build && cmake .. && make -j$(nproc)
```

### C++ Example

```bash
cd examples/smart_classroom && mkdir build && cd build && cmake .. && make -j$(nproc)
```

On non-macOS systems, specify paths:

```bash
cmake .. -DOpenVINO_DIR=/path/to/openvino -DOpenCV_DIR=/path/to/opencv
```

## Model Setup

This project uses the following models (available in FP32, FP16, and FP16-INT8 precision variants under `assets/intel/`). See [Model Download Tools](#model-download-tools-openvino-dev) for how to download them.

| Model | Purpose |
|-------|---------|
| `face-detection-adas-0001` | Face detection |
| `face-reidentification-retail-0095` | Face re-identification (embeddings) |
| `landmarks-regression-retail-0009` | Facial landmarks regression |
| `person-detection-action-recognition-0005` | Person detection + 3 actions (sit/stand/raise hand) |
| `person-detection-action-recognition-0006` | Person detection + 6 actions |
| `person-detection-action-recognition-teacher-0002` | Teacher action detection |
| `person-detection-raisinghand-recognition-0001` | Raise hand detection |

## Face Gallery Setup

Face recognition requires a gallery JSON file mapping identities to reference images.


## Rust API

The `ovi-vision` crate provides a safe, ergonomic Rust API with builder pattern, `Frame`-based lifetime safety, and async inference support.

### Basic usage

```rust
use ovi_vision::{Core, FaceDetector, Video};

let core = Core::new()?;
let mut detector = FaceDetector::builder("model.xml")
    .confidence(0.6)
    .build(&core)?;

let mut video = Video::open("input.mp4")?;

// Frame borrows &Video — prevents video.read() while frame is alive
let frame = video.current_frame()?;
let faces = detector.detect_frame(&frame)?;
```

### Async inference (double-buffering)

Detectors support async `enqueue()` / `fetch_results()` for pipelined inference:

```rust
// Start inference on first frame
let frame = video.current_frame()?;
face_detector.enqueue(&frame)?;
action_detector.enqueue(&frame)?;
drop(frame);

loop {
    let faces = face_detector.fetch_results()?;    // wait for frame N
    let actions = action_detector.fetch_results()?;

    // ... process results ...

    if !video.read()? { break; }

    // Start inference on frame N+1 (runs while we process N's results)
    let next_frame = video.current_frame()?;
    face_detector.enqueue(&next_frame)?;
    action_detector.enqueue(&next_frame)?;
}
```

### Error handling

All FFI functions return `Result`. C++ exceptions are caught via try/catch and converted to `cxx::Exception`, which maps to `ovi_vision::Error::Cxx`.

### Modules

| Module | Key types |
|--------|-----------|
| `detector` | `FaceDetector`, `ActionDetector`, `Detection`, `ActionResult` |
| `tracker` | `Tracker`, `TrackerConfig`, `TrackedResult`, `ActiveTrack`, `TrackPoint` |
| `reid` | `FaceGallery`, `FaceGalleryBuilder` |
| `landmarks` | `LandmarksDetector`, `LandmarksDetectorBuilder`, `FaceLandmarks`, `LandmarkPoint` |
| `video` | `Video`, `Frame` |
| `output` | `AnnotatedFrame`, `VideoWriter`, `Color` |
| `model` | `Device` (Cpu, Gpu, Auto) |
| `error` | `Error`, `Result` |

## Running Examples

### Rust Example

```bash
cargo run -p smart-classroom-rs -- \
    --input assets/data/test1_small.mp4 \
    --m-fd assets/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml \
    --m-act assets/intel/person-detection-action-recognition-0005/FP32/person-detection-action-recognition-0005.xml \
    --m-lm assets/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \
    --m-reid assets/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml \
    --fg assets/data/faces_gallery.json
```

Key flags: `--input` (video path), `--m-fd` (face detection model), `--m-act` (action model), `--m-lm` (landmarks model), `--m-reid` (re-ID model), `--fg` (face gallery JSON), `--no-show` (headless mode), `--read-limit N` (frame limit).

### C++ Example

```bash
cd examples/smart_classroom/build

./smart_classroom_example \
    -i ../../../assets/data/test1_small.mp4 \
    -m_fd ../../../assets/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml \
    -m_act ../../../assets/intel/person-detection-action-recognition-0005/FP32/person-detection-action-recognition-0005.xml \
    -m_lm ../../../assets/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \
    -m_reid ../../../assets/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml \
    -fg ../../../assets/data/faces_gallery.json
```

## Testing

### Unit tests

Pure Rust tests that do not require OpenVINO runtime:

```bash
cargo test -p ovi-vision
```

Tests cover: `Device::as_str()` correctness, `Device::default()`, `TrackerConfig::default()` values, and `Error` display messages.

### Integration testing

End-to-end validation is done by running examples on test videos in `assets/data/`.

### Action event metrics

```bash
cargo run -p action_event_metrics -- -d detections.json -a annotation.xml
```

## Command-Line Reference

### C++ Example Flags

#### Input / Output

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-i` | string | *required* | Input: video file, image, or camera ID (e.g., `0`) |
| `-o` | string | | Output video file path |
| `-loop` | bool | false | Loop input source continuously |
| `-read_limit` | uint32 | 0 | Max frames to process (0 = unlimited) |
| `-no_show` | bool | false | Disable display window |

#### Model Paths

| Flag | Type | Description |
|------|------|-------------|
| `-m_act` | string | Person/Action Detection model (`.xml`) |
| `-m_fd` | string | Face Detection model (`.xml`) |
| `-m_lm` | string | Facial Landmarks Regression model (`.xml`) |
| `-m_reid` | string | Face Reidentification model (`.xml`) |

#### Device Selection

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-d_act` | string | CPU | Device for action detection |
| `-d_fd` | string | CPU | Device for face detection |
| `-d_lm` | string | CPU | Device for landmarks regression |
| `-d_reid` | string | CPU | Device for face reidentification |

#### Detection Thresholds

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-t_fd` | double | 0.6 | Face detection confidence threshold |
| `-t_ad` | double | 0.3 | Person/action detection threshold |
| `-t_ar` | double | 0.75 | Action recognition threshold |
| `-t_reid` | double | 0.7 | Face re-ID cosine distance threshold |

#### Face Gallery & Recognition

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-fg` | string | | Path to face gallery JSON file |
| `-teacher_id` | string | | Teacher identity label |
| `-crop_gallery` | bool | false | Crop and align gallery images |
| `-greedy_reid_matching` | bool | false | Use greedy matching |

### Rust Example Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--input` / `-i` | string | *required* | Input video file path |
| `--m-fd` | string | *required* | Face detection model |
| `--m-act` | string | *required* | Action detection model |
| `--m-lm` | string | *required* | Landmarks model |
| `--m-reid` | string | *required* | Re-identification model |
| `--fg` | string | `""` | Face gallery JSON path |
| `--t-fd` | f32 | 0.6 | Face detection threshold |
| `--t-ad` | f32 | 0.3 | Action detection threshold |
| `--t-ar` | f32 | 0.75 | Action recognition threshold |
| `--no-show` | bool | false | Headless mode |
| `--read-limit` | u32 | 0 | Max frames (0 = unlimited) |
