use crate::error::Result;
use crate::model::Device;
use crate::video::Frame;
use crate::Core;

pub use ovi_vision_sys::{Detection, FaceEmbedding};

/// Builder for creating a FaceEmbedder.
pub struct FaceEmbedderBuilder {
    model: String,
    device: Device,
    max_batch_size: i32,
}

impl FaceEmbedderBuilder {
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

    pub fn build(self, core: &Core) -> Result<FaceEmbedder> {
        let inner = ovi_vision_sys::create_face_embedder(
            &core.inner,
            &self.model,
            self.device.as_str(),
            self.max_batch_size,
        )?;
        Ok(FaceEmbedder { inner })
    }
}

/// Standalone face embedding extractor.
///
/// Computes embedding vectors for detected faces using a re-identification model.
/// Useful for building custom matching logic or storing embeddings in external databases.
pub struct FaceEmbedder {
    inner: cxx::UniquePtr<ovi_vision_sys::FaceReidentifierWrapper>,
}

impl FaceEmbedder {
    pub fn builder(model: impl Into<String>) -> FaceEmbedderBuilder {
        FaceEmbedderBuilder::new(model)
    }

    /// Compute face embeddings for detected faces in a frame.
    ///
    /// Returns one `FaceEmbedding` per input `Detection`, each containing
    /// a float vector representing the face's identity embedding.
    pub fn compute(&mut self, frame: &Frame<'_>, faces: &Vec<Detection>) -> Result<Vec<FaceEmbedding>> {
        Ok(ovi_vision_sys::compute_embeddings(
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
    fn test_embedder_builder_defaults() {
        let b = FaceEmbedderBuilder::new("model.xml");
        assert_eq!(b.model, "model.xml");
        assert_eq!(b.device, Device::Cpu);
        assert_eq!(b.max_batch_size, 16);
    }

    #[test]
    fn test_embedder_builder_chaining() {
        let b = FaceEmbedderBuilder::new("m.xml")
            .device(Device::Gpu)
            .max_batch_size(32);
        assert_eq!(b.device, Device::Gpu);
        assert_eq!(b.max_batch_size, 32);
    }
}
