//! # ovi-vision
//!
//! Safe Rust wrapper for the ovi_vision C++ library.
//! Provides face detection, action detection, multi-object tracking,
//! and face re-identification using OpenVINO.

pub mod error;
pub mod model;
pub mod detector;
pub mod tracker;
pub mod reid;
pub mod landmarks;
pub mod embedder;
pub mod video;
pub mod output;
pub use error::{Error, Result};
pub use model::Device;
pub use detector::{FaceDetector, FaceDetectorBuilder, ActionDetector, ActionDetectorBuilder, Detection, ActionResult};
pub use tracker::{Tracker, TrackerConfig, TrackedResult, TrackPoint, ActiveTrack};
pub use reid::{FaceGallery, FaceGalleryBuilder};
pub use landmarks::{LandmarksDetector, LandmarksDetectorBuilder, LandmarkPoint, FaceLandmarks};
pub use embedder::{FaceEmbedder, FaceEmbedderBuilder, FaceEmbedding};
pub use video::{Video, Frame};
pub use output::{AnnotatedFrame, VideoWriter, Color};

/// OpenVINO Core — entry point for creating detectors.
pub struct Core {
    pub(crate) inner: cxx::UniquePtr<ovi_vision_sys::OvCore>,
}

impl Core {
    /// Create a new OpenVINO Core instance.
    pub fn new() -> Result<Self> {
        let inner = ovi_vision_sys::create_core()?;
        Ok(Self { inner })
    }
}
