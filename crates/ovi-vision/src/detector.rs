use crate::error::Result;
use crate::model::Device;
use crate::Core;
use crate::video::Frame;

pub use ovi_vision_sys::Detection;
pub use ovi_vision_sys::ActionResult;

/// Builder for a face detector.
pub struct FaceDetectorBuilder {
    model: String,
    device: Device,
    confidence: f32,
    input_h: i32,
    input_w: i32,
    expand_ratio: f32,
}

impl FaceDetectorBuilder {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            device: Device::Cpu,
            confidence: 0.6,
            input_h: 600,
            input_w: 600,
            expand_ratio: 1.15,
        }
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn confidence(mut self, threshold: f32) -> Self {
        self.confidence = threshold;
        self
    }

    pub fn input_size(mut self, h: i32, w: i32) -> Self {
        self.input_h = h;
        self.input_w = w;
        self
    }

    pub fn expand_ratio(mut self, ratio: f32) -> Self {
        self.expand_ratio = ratio;
        self
    }

    pub fn build(self, core: &Core) -> Result<FaceDetector> {
        let inner = ovi_vision_sys::create_face_detector(
            &core.inner,
            &self.model,
            self.device.as_str(),
            self.confidence,
            self.input_h,
            self.input_w,
            self.expand_ratio,
        )?;
        Ok(FaceDetector { inner })
    }
}

/// Face detector wrapping OpenVINO inference.
pub struct FaceDetector {
    pub(crate) inner: cxx::UniquePtr<ovi_vision_sys::FaceDetectorWrapper>,
}

impl FaceDetector {
    pub fn builder(model: impl Into<String>) -> FaceDetectorBuilder {
        FaceDetectorBuilder::new(model)
    }

    /// Synchronous detection on a Frame reference.
    pub fn detect_frame(&mut self, frame: &Frame<'_>) -> Result<Vec<Detection>> {
        Ok(ovi_vision_sys::detect_faces_frame(self.inner.pin_mut(), &frame.inner)?)
    }

    /// Submit a frame for async inference. Call `fetch_results()` to get results.
    pub fn enqueue(&mut self, frame: &Frame<'_>) -> Result<()> {
        Ok(ovi_vision_sys::enqueue_face_detection(self.inner.pin_mut(), &frame.inner)?)
    }

    /// Wait for and retrieve async inference results.
    pub fn fetch_results(&mut self) -> Result<Vec<Detection>> {
        Ok(ovi_vision_sys::fetch_face_results(self.inner.pin_mut())?)
    }
}

/// Builder for an action detector.
pub struct ActionDetectorBuilder {
    model: String,
    device: Device,
    det_threshold: f32,
    act_threshold: f32,
    num_actions: u32,
}

impl ActionDetectorBuilder {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            device: Device::Cpu,
            det_threshold: 0.3,
            act_threshold: 0.75,
            num_actions: 3,
        }
    }

    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn det_threshold(mut self, t: f32) -> Self {
        self.det_threshold = t;
        self
    }

    pub fn act_threshold(mut self, t: f32) -> Self {
        self.act_threshold = t;
        self
    }

    pub fn num_actions(mut self, n: u32) -> Self {
        self.num_actions = n;
        self
    }

    pub fn build(self, core: &Core) -> Result<ActionDetector> {
        let inner = ovi_vision_sys::create_action_detector(
            &core.inner,
            &self.model,
            self.device.as_str(),
            self.det_threshold,
            self.act_threshold,
            self.num_actions,
        )?;
        Ok(ActionDetector { inner })
    }
}

/// Action detector wrapping OpenVINO inference.
pub struct ActionDetector {
    pub(crate) inner: cxx::UniquePtr<ovi_vision_sys::ActionDetectorWrapper>,
}

impl ActionDetector {
    pub fn builder(model: impl Into<String>) -> ActionDetectorBuilder {
        ActionDetectorBuilder::new(model)
    }

    /// Synchronous detection on a Frame reference.
    pub fn detect_frame(&mut self, frame: &Frame<'_>) -> Result<Vec<ActionResult>> {
        Ok(ovi_vision_sys::detect_actions_frame(self.inner.pin_mut(), &frame.inner)?)
    }

    /// Submit a frame for async inference. Call `fetch_results()` to get results.
    pub fn enqueue(&mut self, frame: &Frame<'_>) -> Result<()> {
        Ok(ovi_vision_sys::enqueue_action_detection(self.inner.pin_mut(), &frame.inner)?)
    }

    /// Wait for and retrieve async inference results.
    pub fn fetch_results(&mut self) -> Result<Vec<ActionResult>> {
        Ok(ovi_vision_sys::fetch_action_results(self.inner.pin_mut())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_detector_builder_defaults() {
        let b = FaceDetectorBuilder::new("model.xml");
        assert_eq!(b.model, "model.xml");
        assert_eq!(b.device, Device::Cpu);
        assert!((b.confidence - 0.6).abs() < f32::EPSILON);
        assert_eq!(b.input_h, 600);
        assert_eq!(b.input_w, 600);
        assert!((b.expand_ratio - 1.15).abs() < f32::EPSILON);
    }

    #[test]
    fn test_face_detector_builder_chaining() {
        let b = FaceDetectorBuilder::new("m.xml")
            .device(Device::Gpu)
            .confidence(0.8)
            .input_size(300, 300)
            .expand_ratio(1.0);
        assert_eq!(b.device, Device::Gpu);
        assert!((b.confidence - 0.8).abs() < f32::EPSILON);
        assert_eq!(b.input_h, 300);
        assert_eq!(b.input_w, 300);
        assert!((b.expand_ratio - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_action_detector_builder_defaults() {
        let b = ActionDetectorBuilder::new("model.xml");
        assert_eq!(b.model, "model.xml");
        assert_eq!(b.device, Device::Cpu);
        assert!((b.det_threshold - 0.3).abs() < f32::EPSILON);
        assert!((b.act_threshold - 0.75).abs() < f32::EPSILON);
        assert_eq!(b.num_actions, 3);
    }

    #[test]
    fn test_action_detector_builder_chaining() {
        let b = ActionDetectorBuilder::new("m.xml")
            .device(Device::Auto)
            .det_threshold(0.5)
            .act_threshold(0.9)
            .num_actions(6);
        assert_eq!(b.device, Device::Auto);
        assert!((b.det_threshold - 0.5).abs() < f32::EPSILON);
        assert!((b.act_threshold - 0.9).abs() < f32::EPSILON);
        assert_eq!(b.num_actions, 6);
    }
}
