# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-06

### Added

- Face detection with configurable confidence threshold and async double-buffered inference
- Action detection (SSD-based with Soft Non-Maximum Suppression)
- Multi-object tracking using Hungarian algorithm with configurable parameters
- Face re-identification with cosine distance-based gallery matching
- Standalone landmarks detection (5-point facial landmarks via `LandmarksDetector`)
- Tracker API: `active_tracks()`, `is_track_valid()`, `is_track_forgotten()`, `drop_forgotten_tracks()`, `reset()`
- Gallery API: `all_labels()`, `label_exists()`
- Camera input: `Video::open_camera(device_index)`
- Frame lifetime safety (`Frame<'a>` with `PhantomData` prevents use-after-read at compile time)
- Video output: `AnnotatedFrame` for drawing and `VideoWriter` for saving MP4
- Builder pattern API for all detectors and gallery
- Async inference pipeline: `enqueue()` / `fetch_results()` for double-buffered pipelining
- Safety: null-check guards on all frame-based FFI functions, `expect()` with descriptive messages in build scripts
- CXX bridge FFI with exception-safe wrappers (C++ exceptions mapped to `Result`)
- 5 Rust example applications: face detection, action detection, tracking, face re-ID, smart classroom
- 1 C++ example application: smart classroom with 50+ CLI flags
- Tools: `create_gallery` (face gallery JSON generator), `action_event_metrics` (precision/recall evaluation)
