# Architecture Plan: From `openvino-vision` to `ovi`

## Background

### Motivation

This project provides a dual-language (C++ and Rust) inference framework built on Intel OpenVINO. C++ provides the core inference engine; Rust provides safe idiomatic bindings. Intel's official `openvino-rs` crate is unmaintained and too slow to evolve, so we build our own. C++ users can also consume the libraries directly without the Rust layer.

### Goals

1. **Rust crate ecosystem** — publish crates that other Rust projects can depend on via Cargo
2. **C/C++ reusable** — the C++ libraries are standalone and can be consumed directly by C/C++ projects without the Rust layer
3. **Multimodal** — expand beyond computer vision into audio, LLM, and other modalities, all powered by OpenVINO inference

### Naming

- **`ovi`** — unified code-facing prefix across the entire ecosystem
  - Stands for **O**pen**V**INO **I**nterface
  - Short (3 chars), follows C++ ecosystem conventions (`ov`, `cv`, `tf`)
  - Language-neutral — works equally well for C++ includes/namespaces and Rust crate names
  - Used for: C++ include prefix, C++ namespaces, C++ library names, Rust crate names
- **`openvino-rust-cpp`** — GitHub repository name
  - Optimized for discoverability — clearly communicates dual-language C++ and Rust support
  - Only appears in URLs and git commands, not in code

## Current State

```
openvino-rust-cpp/                     # GitHub repo (was openvino-vision)
├── cpp/                              # Single C++ library (vision only)
│   ├── include/openvino_vision/      # All headers under one namespace
│   ├── src/
│   └── ffi/                          # CXX bridge layer
├── crates/
│   ├── openvino-vision-sys/          # FFI bindings
│   ├── openvino-vision/              # Safe Rust API (includes Core, Device, Error)
│   └── tui/                          # Terminal UI
├── examples/
└── tools/
```

### Problems with current structure

- `Core`, `Device`, `Error` live inside `openvino-vision` but are not vision-specific — they are OpenVINO fundamentals shared across all domains
- C++ headers all live under a single `openvino_vision/` prefix with no domain separation
- No clear separation point for adding audio/LLM modules

## Proposed Architecture

### C++ include layout

Headers share a single include root at `cpp/include/`. The `ovi/` prefix plus domain subdirectories (`core/`, `vision/`, ...) provide namespaced `#include` paths without redundant nesting inside each domain directory.

```
openvino-rust-cpp/                     # GitHub repo name (for discoverability)
├── cpp/
│   ├── include/ovi/                  # Single shared include root
│   │   ├── core/                     #   Shared OpenVINO infrastructure
│   │   │   ├── cnn.hpp              #     Base CNN classes (CnnDLSDKBase, VectorCNN)
│   │   │   ├── ocv_common.hpp       #     OpenCV helpers
│   │   │   ├── openvino_utils.hpp   #     OpenVINO utilities
│   │   │   ├── images_capture.hpp   #     Video/image input abstraction
│   │   │   ├── kuhn_munkres.hpp     #     Hungarian algorithm
│   │   │   ├── performance_metrics.hpp
│   │   │   └── slog.hpp            #     Structured logging
│   │   ├── vision/                   #   Computer vision domain
│   │   │   ├── detector.hpp         #     Face detection
│   │   │   ├── action_detector.hpp  #     Action detection
│   │   │   ├── tracker.hpp          #     Multi-object tracking
│   │   │   ├── face_reid.hpp        #     Face re-identification & gallery
│   │   │   ├── actions.hpp          #     Action data structures
│   │   │   └── logger.hpp           #     Detection logging
│   │   ├── audio/                    #   (future) Audio inference
│   │   └── llm/                      #   (future) LLM inference
│   │
│   ├── core/src/                     # Core implementations
│   │   ├── cnn.cpp
│   │   └── images_capture.cpp
│   │
│   ├── vision/                       # Vision implementations + FFI
│   │   ├── src/
│   │   │   ├── detector.cpp
│   │   │   ├── action_detector.cpp
│   │   │   ├── tracker.cpp
│   │   │   ├── reid_gallery.cpp
│   │   │   ├── align_transform.cpp
│   │   │   └── logger.cpp
│   │   └── ffi/
│   │       ├── bridge.hpp
│   │       └── bridge.cpp
│   │
│   ├── audio/                        # (future)
│   │   ├── src/
│   │   └── ffi/
│   │
│   └── llm/                          # (future)
│       ├── src/
│       └── ffi/
│
├── crates/
│   ├── ovi-core/                     # Shared Rust types & OpenVINO engine wrapper
│   │   └── src/
│   │       ├── lib.rs               #   Re-exports
│   │       ├── error.rs             #   Error, Result
│   │       ├── model.rs             #   Device enum
│   │       └── core.rs              #   Core (wraps ov::Core)
│   │
│   ├── ovi-vision-sys/               # Vision FFI bindings (CXX bridge)
│   │   ├── build.rs                 #   CMake: builds cpp/core + cpp/vision + FFI
│   │   └── src/lib.rs               #   #[cxx::bridge] definitions
│   │
│   ├── ovi-vision/                   # Vision safe Rust API
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── detector.rs          #   FaceDetector, ActionDetector
│   │       ├── tracker.rs           #   Tracker, TrackerConfig
│   │       ├── reid.rs              #   FaceGallery
│   │       ├── video.rs             #   Video, Frame<'a>
│   │       └── output.rs           #   AnnotatedFrame, VideoWriter
│   │
│   ├── ovi-audio-sys/                # (future)
│   ├── ovi-audio/                    # (future)
│   ├── ovi-llm-sys/                  # (future)
│   ├── ovi-llm/                      # (future)
│   │
│   ├── ovi/                          # Umbrella crate — re-exports all domains
│   │   └── src/lib.rs
│   │
│   └── tui/                          # Terminal UI application
│
├── examples/
├── tools/
└── Cargo.toml                        # Workspace
```

## Design Decisions

### 1. Extract `core` layer

`Core`, `Device`, `Error` are OpenVINO fundamentals, not vision-specific. Every domain (vision, audio, LLM) will need the same `ov::Core` to load models and the same `Device` enum to select hardware.

**Before:**
```rust
use openvino_vision::{Core, Device, FaceDetector};
```

**After:**
```rust
use ovi_core::{Core, Device};
use ovi_vision::{FaceDetector, ActionDetector};
```

### 2. C++ include path convention

Each domain gets its own subdirectory under the unified `ovi/` prefix:

```cpp
#include <ovi/core/cnn.hpp>
#include <ovi/vision/detector.hpp>
#include <ovi/audio/speech.hpp>       // future
#include <ovi/llm/generator.hpp>      // future
```

C++ namespaces mirror the include paths:

```cpp
namespace ovi::core { class CnnDLSDKBase { ... }; }
namespace ovi::vision { class FaceDetection { ... }; }
```

### 3. Independent `-sys` crates per domain

Each domain has its own `-sys` crate with its own `build.rs`. Benefits:

- **Minimal compilation** — users who only need vision don't compile audio C++ code
- **Independent versioning** — domains can evolve at different speeds
- **Isolated dependencies** — audio might need different system libs than vision

### 4. Umbrella crate with feature flags

```toml
# crates/ovi/Cargo.toml
[features]
default = ["vision"]
vision  = ["dep:ovi-vision"]
audio   = ["dep:ovi-audio"]
llm     = ["dep:ovi-llm"]
full    = ["vision", "audio", "llm"]
```

Consumer usage:

```toml
# Only vision
ovi = "0.1"

# Vision + audio
ovi = { version = "0.1", features = ["audio"] }

# Everything
ovi = { version = "0.1", features = ["full"] }
```

### 5. Umbrella crate re-export structure

```rust
// crates/ovi/src/lib.rs
pub use ovi_core as core;

#[cfg(feature = "vision")]
pub use ovi_vision as vision;

#[cfg(feature = "audio")]
pub use ovi_audio as audio;

#[cfg(feature = "llm")]
pub use ovi_llm as llm;
```

Consumer usage:

```rust
use ovi::core::{Core, Device};
use ovi::vision::{FaceDetector, Video, Frame};
```

## What belongs where

### `ovi-core` (shared across all domains)

| Item | Reason |
|------|--------|
| `Core` (wraps `ov::Core`) | All domains load models through the same engine |
| `Device` enum | Hardware selection is domain-agnostic |
| `Error`, `Result` | Shared error types |

### `ovi-vision` (vision-specific)

| Item | Reason |
|------|--------|
| `FaceDetector`, `ActionDetector` | Vision inference |
| `Tracker`, `TrackerConfig` | Object tracking |
| `FaceGallery` | Face re-identification |
| `Video`, `Frame<'a>` | Video I/O with lifetime safety |
| `AnnotatedFrame`, `VideoWriter` | Visual output |
| `Detection`, `ActionResult`, `TrackedResult` | Vision result types |

### `ovi-audio` (future)

| Item | Reason |
|------|--------|
| `SpeechRecognizer` | Speech-to-text inference |
| `AudioClassifier` | Audio event detection |
| `AudioStream`, `AudioFrame` | Audio I/O |

### `ovi-llm` (future)

| Item | Reason |
|------|--------|
| `TextGenerator` | Text generation / completion |
| `Embedder` | Text embedding extraction |
| `Tokenizer` | Tokenization utilities |

## C++ library CMake structure

Each C++ domain is a separate static library. All share one include root at `cpp/include/`. Vision and future domains link against core.

```cmake
# cpp/CMakeLists.txt (top-level)
set(OVI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# cpp/core/CMakeLists.txt
add_library(ovi_core STATIC ...)
target_include_directories(ovi_core PUBLIC ${OVI_INCLUDE_DIR})
target_link_libraries(ovi_core PUBLIC ${OpenCV_LIBS} openvino::runtime)

# cpp/vision/CMakeLists.txt
add_library(ovi_vision STATIC ...)
target_link_libraries(ovi_vision PUBLIC ovi_core)
# No need to set include dir — inherited from ovi_core (PUBLIC)

# cpp/vision/ffi/CMakeLists.txt
add_library(ovi_vision_ffi STATIC bridge.cpp ${CXX_GENERATED_SOURCE})
target_link_libraries(ovi_vision_ffi PUBLIC ovi_vision)
```

With `cpp/include/` as the single include root, `#include <ovi/core/cnn.hpp>` resolves to `cpp/include/ovi/core/cnn.hpp` and `#include <ovi/vision/detector.hpp>` resolves to `cpp/include/ovi/vision/detector.hpp`. CMake only needs one `target_include_directories` call on `ovi_core`, and all downstream targets inherit it via `PUBLIC` linkage.

## Migration order

Phase 1 — Foundation (do now):

1. Extract `Core`, `Device`, `Error` into `ovi-core`
2. Rename C++ include prefix from `openvino_vision/` to `ovi/vision/` (+ `ovi/core/`)
3. Rename Rust crates: `openvino-vision` → `ovi-vision`, `openvino-vision-sys` → `ovi-vision-sys`
4. Split C++ source into `cpp/core/` and `cpp/vision/`
5. Update `build.rs` to build both `ovi_core` and `ovi_vision`

Phase 2 — Umbrella (do after phase 1):

6. Create `ovi` umbrella crate with feature flags
7. Update examples and tools to use new crate names

Phase 3 — New domains (do when needed):

8. Add `cpp/audio/` + `ovi-audio-sys` + `ovi-audio` when starting audio work
9. Add `cpp/llm/` + `ovi-llm-sys` + `ovi-llm` when starting LLM work

## Repository naming

Rename the repository from `openvino-vision` to `openvino-rust-cpp`. This reflects the broader scope and dual-language nature. On crates.io:

- `ovi` — umbrella
- `ovi-core` — shared types
- `ovi-vision` — computer vision
- `ovi-audio` — audio (future)
- `ovi-llm` — LLM (future)
