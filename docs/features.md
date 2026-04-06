# OpenVINO Vision - Features

A dual-language (C++ and Rust) real-time AI inference library focused on computer vision, designed for applications such as smart classroom monitoring.

---

## Core Inference Features

### Face Detection

- SSD-based face detection returning bounding boxes with confidence scores
- Configurable confidence threshold, input dimensions, and bounding box expansion ratio
- Supports both synchronous and asynchronous (double-buffered) inference

### Action Detection

- SSD-based human action detection that classifies actions such as sitting, standing, and raising hand
- Soft Non-Maximum Suppression for improved detection quality
- Auto-detection of old and new model formats
- Configurable detection and action confidence thresholds, with support for multiple action classes

### Multi-Object Tracking

- Hungarian algorithm (Kuhn-Munkres) based multi-object tracker
- Combines shape affinity and motion prediction for robust matching
- Configurable parameters including forget delay, minimum track duration, affinity threshold, bounding box aspect ratio and height range constraints
- Supports greedy matching as a fallback option
- Track lifecycle management: query active tracks with trajectory history, check track validity and forgotten status
- Programmatic track control: drop forgotten tracks and full tracker reset

### Face Re-Identification

- Loads a JSON-formatted face gallery database for identity matching
- Three-model pipeline: face detection -> landmark detection -> face alignment -> embedding extraction
- Cosine distance-based face matching with configurable re-identification threshold
- Supports both greedy matching and Hungarian algorithm matching
- Bulk gallery operations: retrieve all identity labels, check if a label exists

### Face Alignment

- 5-point landmark-based Procrustes alignment using SVD decomposition
- Produces a 2x3 affine transformation matrix for face cropping and normalization

### Standalone Landmarks Detection

- 5-point facial landmarks detection (left eye, right eye, nose, left mouth corner, right mouth corner)
- `LandmarksDetector` with builder pattern API (configurable model, device, max batch size)
- Can be used independently from the face gallery pipeline for custom face alignment workflows

---

## Rust Safe API

### Frame Lifetime Safety

- `Frame<'a>` type uses `PhantomData<&'a Video>` to bind frame lifetime to the video source
- Prevents dangling pointer bugs at compile time — the compiler rejects code that uses a stale frame after reading the next one

### Builder Pattern API

- `FaceDetectorBuilder`, `ActionDetectorBuilder`, and `FaceGalleryBuilder` provide fluent builder interfaces
- Sensible defaults for device (CPU), thresholds, and input dimensions
- Chainable configuration methods for device (CPU/GPU/AUTO), thresholds, input sizes, and more

### Asynchronous Inference

- All detectors expose `enqueue()` and `fetch_results()` for double-buffered async inference
- Enables overlapping inference with I/O operations for higher throughput

### Error Handling

- All FFI calls map C++ exceptions to Rust `Result<T>` types via the CXX bridge
- Typed error enum (`Error`) with variants for model loading, inference, video capture, gallery, and CXX bridge errors

### Camera Input

- Open camera devices by index via `Video::open_camera(device_index)` for live inference
- Built on the same OpenCV backend as file-based video input

### Video Output

- `AnnotatedFrame` for drawing bounding boxes and labels on cloned video frames
- `VideoWriter` for saving annotated frames to MP4 output files
- Predefined color constants (green, red, blue, yellow, cyan, white) for visualization

---

## Tools

### create_gallery

- Generates a face gallery JSON file from a directory structure
- Directory names serve as person labels; contained `.jpg` images are collected as face samples
- Produces stable, sorted output with absolute image paths

### action_event_metrics

- Evaluates action detection accuracy by computing Precision and Recall metrics
- Supports CVAT XML annotation format for ground truth and JSON format for detection results
- Full post-processing pipeline: event smoothing, filtering, extrapolation, interpolation, and merging
- Configurable minimum action length and smoothing window size

---

## Example Applications

### Smart Classroom (C++)

- Full-featured classroom monitoring application with visualization
- Supports multiple operating modes: Student, Teacher, and Top-K tracking
- Approximately 50 command-line flags via gflags for fine-grained configuration
- Includes output video writing, detection logging, and action event export

### Smart Classroom (Rust)

- Equivalent Rust implementation demonstrating the safe API
- Uses clap for CLI argument parsing
- Demonstrates the async double-buffered inference pipeline: detect -> identify -> track
- Compact implementation (~209 lines) compared to the C++ version (~848 lines)

### Face Detection (Rust)

- Minimal single-model example for face detection only
- Demonstrates `FaceDetector` builder pattern and synchronous inference

### Action Detection (Rust)

- Single-model example for action detection
- Shows per-frame action classification with confidence scores

### Tracking (Rust)

- Detection combined with multi-object tracking
- Demonstrates async double-buffered inference pattern (`enqueue` / `fetch_results`)
- Supports both face tracking and action tracking modes

### Face Re-Identification (Rust)

- Multi-model pipeline: face detection + landmarks + re-identification
- Demonstrates gallery-based identity matching with known/unknown classification

---

## Build System

### C++ Library

- CMake 3.10+ with C++17 standard
- Auto-detects macOS Homebrew paths for OpenCV and OpenVINO
- Builds as a static library (`libopenvino_vision`)

### Rust Workspace

- Cargo workspace with five member crates
- `build.rs` invokes CMake to compile the C++ library and FFI bridge
- CXX bridge connects C++ and Rust with automatic exception-to-Result mapping

### Model Management

- Uses Open Model Zoo tools (`omz_downloader`, `omz_converter`) for model downloading and conversion
- Pre-trained models stored in `assets/intel/` and `assets/public/`

---

## Dependencies

| Dependency | Version | Purpose |
|---|---|---|
| OpenVINO | 2025.0.0+ | Neural network inference runtime |
| OpenCV | 4.x | Image processing, video I/O, visualization |
| CXX | 1.0 | Rust-C++ FFI bridge |
| clap | 4.x | Rust CLI argument parsing |
| gflags | - | C++ CLI argument parsing (examples only) |
| thiserror | 2.x | Rust error type derivation |
| serde / serde_json | 1.x | JSON serialization (tools) |
| quick-xml | 0.37 | XML parsing (metrics tool) |

---

## Testing Guide

### Prerequisites

Ensure OpenVINO and OpenCV are installed:

```bash
# macOS (Homebrew)
brew install openvino opencv
```

Then verify the project compiles:

```bash
cargo build
```

### Step 1: Unit Tests (No OpenVINO Runtime Required)

Run pure Rust unit tests to verify basic types (`Device`, `TrackerConfig`, `Error`):

```bash
cargo test -p openvino-vision
```

### Step 2: Face Detection

Test face detection in isolation — only requires one model:

```bash
cargo run -p example-face-detection -- \
  --input assets/data/test1.mp4 \
  --m-fd assets/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml
```

Output shows per-frame face count with bounding box coordinates and confidence scores.

### Step 3: Action Detection

Test action detection in isolation — only requires one model:

```bash
cargo run -p example-action-detection -- \
  --input assets/data/test1.mp4 \
  --m-act assets/intel/person-detection-action-recognition-0006/FP32/person-detection-action-recognition-0006.xml
```

Output shows per-frame action count with bounding boxes, detection confidence, and action labels.

To test with different action models, replace the `--m-act` argument:

| Model | Description |
|---|---|
| `person-detection-action-recognition-0005` | 3 action classes |
| `person-detection-action-recognition-0006` | 6 action classes |
| `person-detection-raisinghand-recognition-0001` | Raising hand detection |
| `person-detection-action-recognition-teacher-0002` | Teacher action detection |

### Step 4: Detection + Tracking

Test detection combined with multi-object tracking. Choose face or action mode:

```bash
# Face tracking
cargo run -p example-tracking -- \
  --input assets/data/test1.mp4 \
  --mode face \
  --m-fd assets/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml

# Action tracking
cargo run -p example-tracking -- \
  --input assets/data/test1.mp4 \
  --mode action \
  --m-act assets/intel/person-detection-action-recognition-0006/FP32/person-detection-action-recognition-0006.xml
```

This example uses the async double-buffered inference pattern (`enqueue` / `fetch_results`). Output shows tracked object IDs that persist across frames.

### Step 5: Face Re-Identification

Test face detection + gallery-based identity matching. Requires face detection, landmarks, and re-identification models plus a gallery JSON:

```bash
cargo run -p example-face-reid -- \
  --input assets/data/test1.mp4 \
  --m-fd assets/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml \
  --m-lm assets/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \
  --m-reid assets/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml \
  --fg assets/data/faces_gallery.json
```

Output shows detected faces matched against the gallery identities or labeled as "Unknown".

### Step 6: Face Gallery Generation Tool

Generate a new gallery JSON from the face database directory:

```bash
cargo run -p create_gallery -- \
  --face-db assets/face_db/ \
  --output test_gallery.json
```

Compare the output with the existing `assets/data/faces_gallery.json` to verify correctness.

### Step 7: Full Pipeline (All Features Combined)

Run the integrated smart classroom example with all models and face gallery:

```bash
cargo run -p smart-classroom-rs -- \
  --input assets/data/test1.mp4 \
  --m-fd assets/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml \
  --m-act assets/intel/person-detection-action-recognition-0006/FP32/person-detection-action-recognition-0006.xml \
  --m-lm assets/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \
  --m-reid assets/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml \
  --fg assets/data/faces_gallery.json \
  --no-show
```

Remove `--no-show` to enable the real-time GUI display (requires a desktop environment).

### Step 8: Test with Different Videos

Multiple test videos are available in `assets/data/`:

| Video | Description |
|---|---|
| `test1.mp4` / `test1_small.mp4` | General test footage |
| `magnetic.mp4` / `magnetic_clip3.mp4` | Classroom scene clips |
| `classroom_source.mp4` | Alternative test source |

Replace `--input` with any of these files in any of the examples above.

### Step 9: C++ Example (Optional)

Build and run the C++ version for comparison:

```bash
cd examples/smart_classroom && mkdir -p build && cd build && cmake .. && make
./smart_classroom_example \
  -i ../../../assets/data/test1.mp4 \
  -m_fd ../../../assets/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml \
  -m_act ../../../assets/intel/person-detection-action-recognition-0006/FP32/person-detection-action-recognition-0006.xml \
  -m_lm ../../../assets/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml \
  -m_reid ../../../assets/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml \
  -fg ../../../assets/data/faces_gallery.json \
  -no_show
```

### Recommended Testing Order

1. Unit tests (Step 1)
2. Build verification (`cargo build`)
3. Face detection only (Step 2)
4. Action detection only (Step 3)
5. Detection + tracking (Step 4)
6. Face re-identification (Step 5)
7. Gallery generation tool (Step 6)
8. Full pipeline integration (Step 7)
9. Different videos and models (Step 8)
10. C++ example comparison (Step 9)
