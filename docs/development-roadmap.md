# Development Roadmap

> This document consolidates the current project status, known gaps, and planned next steps.
> Last updated: 2026-04-06

## Current Status

### What's done

- C++ inference pipeline is complete: face detection → landmarks → face alignment → re-ID → gallery matching → action recognition → multi-object tracking
- CXX FFI bridge with exception-safe wrappers (all C++ exceptions caught and converted to `Result`)
- Safe Rust API with builder pattern, `Frame<'a>` lifetime safety, async inference (double-buffering)
- TUI application (ratatui) with menu/config/dashboard screens and async task spawning
- 6 example applications: 1 C++ integrated, 4 Rust modular (face detection, action detection, tracking, face re-ID), 1 Rust integrated (smart_classroom_rs)
- 2 tools: `create_gallery` (face gallery JSON builder), `action_event_metrics` (precision/recall evaluation)
- 7 pre-trained Intel models downloaded with FP32/FP16/FP16-INT8 variants
- Test videos and face database (4 identities) in assets/
- Standalone landmarks detection (`LandmarksDetector` with builder pattern)
- Tracker API expansion: `active_tracks()`, `is_track_valid()`, `is_track_forgotten()`, `drop_forgotten_tracks()`, `reset()`
- Camera input: `Video::open_camera(device_index)`
- Gallery bulk operations: `all_labels()`, `label_exists()`
- Safety hardening: null-check guard (`require_frame`) on all frame-based FFI functions, `unwrap()` → `expect()` in build scripts, safe fallback in TUI config lookup

### What's working but minimal

- Unit tests: only 4 tests covering `Device::as_str()`, `Device::default()`, `TrackerConfig::default()`, `Error` display
- Rustdoc: present on public API but could be more thorough with examples

---

## Known Gaps

### A. Rust API gaps (C++ has it, Rust doesn't expose)

| Gap | Status | Description | Impact |
|-----|--------|-------------|--------|
| ~~Standalone landmarks detection~~ | **Done** | `LandmarksDetector` with builder pattern exposed as public API | ~~Users can't build custom pipelines~~ |
| ~~Tracker trajectory history~~ | **Done** | `active_tracks()`, `is_track_valid()`, `is_track_forgotten()`, `drop_forgotten_tracks()`, `reset()` | ~~Can't do behavior analysis over time~~ |
| ~~Gallery bulk operations~~ | **Done** | `all_labels()`, `label_exists()` exposed on `FaceGallery` | ~~Inconvenient for building UI~~ |
| ~~Camera input~~ | **Done** | `Video::open_camera(device_index)` | ~~No formal API~~ |
| Per-action-class confidence | Open | C++ `ActionDetection` computes confidence for each action class; Rust only receives top-1 label + confidence. Fix requires modifying C++ `GetDetections()` post-processing (data lost before FFI boundary) | Can't implement confidence-based filtering across all action classes |
| Batch inference | Open | C++ `VectorCNN::Compute()` supports batch processing; not exposed in FFI or Rust | Can't efficiently process multiple frames or images in a single inference call |
| Performance metrics | Open | C++ has `performance_metrics.hpp` for timing; not exposed | No built-in profiling from Rust |
| Logging/debugging | Open | C++ has `DetectionsLogger` (XML/CSV output) and `slog.hpp`; not exposed | No structured logging from Rust side |

### B. ~~Deprecated methods to clean up~~ — **Done**

Removed in the `ovi` migration (breaking change). The 4 deprecated `&Video` methods and `deprecated.rs` module have been deleted.

### C. Publishing readiness

| Item | Status |
|------|--------|
| LICENSE file | Not checked into repo (CLAUDE.md mentions Apache 2.0) |
| `license` field in Cargo.toml | Missing in both crates |
| `repository` field in Cargo.toml | Missing in both crates |
| `keywords` / `categories` in Cargo.toml | Missing |
| CI/CD pipeline (GitHub Actions) | None |
| Test coverage | 4 unit tests only, no integration tests |
| Benchmarks | None |
| CHANGELOG | None |

---

## Development Directions

### Direction 1: Deepen the Rust API

Fill the gaps between what C++ provides and what Rust exposes.

**Completed:**
- ~~Standalone landmarks detection~~ — `LandmarksDetector` with builder pattern
- ~~Tracker trajectory history~~ — 5 new methods on `Tracker`
- ~~Camera input~~ — `Video::open_camera(device_index)`
- ~~Gallery bulk operations~~ — `all_labels()`, `label_exists()`

**Remaining (priority order):**
1. **Per-action-class confidence** — expose full confidence vector per detection, not just top-1. Requires modifying C++ `ActionDetection::GetDetections()` to preserve per-class scores before FFI boundary. Higher risk — touches core inference logic.
2. **Batch inference** — expose `VectorCNN::Compute()` batch API through FFI
3. **Performance metrics** — expose `performance_metrics.hpp` timing from Rust
4. **Logging/debugging** — expose `DetectionsLogger` and `slog.hpp` from Rust

### Direction 2: Publishing foundation

Prepare the crate for public consumption:

1. Add `LICENSE` file (Apache 2.0) to repo root
2. Add `license`, `repository`, `keywords`, `categories` to all Cargo.toml files
3. Set up GitHub Actions CI (at minimum: `cargo test`, `cargo clippy`, `cargo doc`)
4. Expand unit test coverage (especially for builder defaults, error paths)
5. Add integration test infrastructure (optional: requires OpenVINO runtime)
6. Create CHANGELOG.md
7. Remove deprecated `&Video` methods (breaking change → bump to 0.2.0)

### Direction 3: `ovi` architecture migration — **Phase 1 Done**

Phase 1 (rename) is complete:
- C++ headers moved to `ovi/core/` and `ovi/vision/`
- Rust crates renamed to `ovi-vision-sys` and `ovi-vision`
- CMake targets renamed to `ovi_vision` and `ovi_vision_ffi`
- Deprecated `&Video` methods removed

Remaining phases (documented in [architecture-plan.md](architecture-plan.md)):
- Phase 2: Create `ovi` umbrella crate with feature flags
- Phase 3: Add new domains (`ovi-audio`, `ovi-llm`) when needed

---

## Recommended Next Steps

Direction 1 is mostly complete — the high-impact API gaps (landmarks, tracker, camera, gallery) are filled. The remaining gaps (per-action confidence, batch inference, metrics, logging) are lower priority and can be addressed incrementally.

Direction 2 (publishing foundation) is done — LICENSE, Cargo.toml metadata, deprecated methods removed.

**Next: Direction 3 Phase 2 (umbrella crate) or Direction 1 remaining gaps, depending on priorities.**

---

## Notes

- The `ovi` architecture plan is documented separately in [architecture-plan.md](architecture-plan.md)
- Feature documentation is in [features.md](features.md)
- Models are managed via Open Model Zoo tools (`omz_downloader` / `omz_converter`)
- Integration testing is done by running examples against test videos in `assets/data/`
