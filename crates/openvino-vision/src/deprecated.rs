//! Deprecated methods that accept `&Video` instead of `&Frame`.
//!
//! These extension traits provide backwards compatibility for code using the
//! old `&Video`-based API. Prefer the `_frame()` variants on each type instead.
//!
//! To use: `use openvino_vision::deprecated::*;`
//!
//! This module can be removed entirely when ready to drop the old API.

use crate::detector::{ActionDetector, ActionResult, Detection, FaceDetector};
use crate::error::Result;
use crate::reid::FaceGallery;
use crate::tracker::{TrackedResult, Tracker};
use crate::video::Video;

/// Deprecated face detection method.
#[deprecated(note = "Use FaceDetector::detect_frame() with a Frame reference instead")]
pub trait FaceDetectorDeprecated {
    fn detect(&mut self, video: &Video) -> Result<Vec<Detection>>;
}

#[allow(deprecated)]
impl FaceDetectorDeprecated for FaceDetector {
    fn detect(&mut self, video: &Video) -> Result<Vec<Detection>> {
        Ok(openvino_vision_sys::detect_faces(self.inner.pin_mut(), &video.inner)?)
    }
}

/// Deprecated action detection method.
#[deprecated(note = "Use ActionDetector::detect_frame() with a Frame reference instead")]
pub trait ActionDetectorDeprecated {
    fn detect(&mut self, video: &Video) -> Result<Vec<ActionResult>>;
}

#[allow(deprecated)]
impl ActionDetectorDeprecated for ActionDetector {
    fn detect(&mut self, video: &Video) -> Result<Vec<ActionResult>> {
        Ok(openvino_vision_sys::detect_actions(self.inner.pin_mut(), &video.inner)?)
    }
}

/// Deprecated tracking method.
#[deprecated(note = "Use Tracker::process_frame() with a Frame reference instead")]
pub trait TrackerDeprecated {
    fn process(
        &mut self,
        detections: &Vec<Detection>,
        frame_idx: i32,
        video: &Video,
    ) -> Result<Vec<TrackedResult>>;
}

#[allow(deprecated)]
impl TrackerDeprecated for Tracker {
    fn process(
        &mut self,
        detections: &Vec<Detection>,
        frame_idx: i32,
        video: &Video,
    ) -> Result<Vec<TrackedResult>> {
        Ok(openvino_vision_sys::track(self.inner.pin_mut(), detections, frame_idx, &video.inner)?)
    }
}

/// Deprecated face identification method.
#[deprecated(note = "Use FaceGallery::identify_frame() with a Frame reference instead")]
pub trait FaceGalleryDeprecated {
    fn identify(&mut self, video: &Video, faces: &Vec<Detection>) -> Result<Vec<i32>>;
}

#[allow(deprecated)]
impl FaceGalleryDeprecated for FaceGallery {
    fn identify(&mut self, video: &Video, faces: &Vec<Detection>) -> Result<Vec<i32>> {
        Ok(openvino_vision_sys::identify_faces(self.inner.pin_mut(), &video.inner, faces)?)
    }
}
