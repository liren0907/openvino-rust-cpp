use crate::error::Result;
use crate::model::Device;
use crate::video::Frame;
use crate::Core;

pub use openvino_vision_sys::{Detection, FaceLandmarks, LandmarkPoint};

/// Builder for creating a LandmarksDetector.
pub struct LandmarksDetectorBuilder {
    model: String,
    device: Device,
    max_batch_size: i32,
}

impl LandmarksDetectorBuilder {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            device: Device::Cpu,
            max_batch_size: 16,
        }
    }

    pub fn device(mut self, d: Device) -> Self {
        self.device = d;
        self
    }

    pub fn max_batch_size(mut self, n: i32) -> Self {
        self.max_batch_size = n;
        self
    }

    pub fn build(self, core: &Core) -> Result<LandmarksDetector> {
        let inner = openvino_vision_sys::create_landmarks_detector(
            &core.inner,
            &self.model,
            self.device.as_str(),
            self.max_batch_size,
        )?;
        Ok(LandmarksDetector { inner })
    }
}

/// Standalone facial landmarks detector.
///
/// Computes 5-point facial landmarks (left eye, right eye, nose, left mouth,
/// right mouth) for detected faces in a frame.
pub struct LandmarksDetector {
    inner: cxx::UniquePtr<openvino_vision_sys::LandmarksDetectorWrapper>,
}

impl LandmarksDetector {
    pub fn builder(model: impl Into<String>) -> LandmarksDetectorBuilder {
        LandmarksDetectorBuilder::new(model)
    }

    /// Compute facial landmarks for detected faces in a frame.
    ///
    /// Returns one `FaceLandmarks` per input `Detection`, each containing
    /// 5 normalized (x, y) landmark points.
    pub fn compute(&mut self, frame: &Frame<'_>, faces: &Vec<Detection>) -> Result<Vec<FaceLandmarks>> {
        Ok(openvino_vision_sys::compute_landmarks(
            self.inner.pin_mut(),
            &frame.inner,
            faces,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landmarks_builder_defaults() {
        let b = LandmarksDetectorBuilder::new("model.xml");
        assert_eq!(b.model, "model.xml");
        assert_eq!(b.device, Device::Cpu);
        assert_eq!(b.max_batch_size, 16);
    }

    #[test]
    fn test_landmarks_builder_chaining() {
        let b = LandmarksDetectorBuilder::new("m.xml")
            .device(Device::Gpu)
            .max_batch_size(32);
        assert_eq!(b.device, Device::Gpu);
        assert_eq!(b.max_batch_size, 32);
    }
}
