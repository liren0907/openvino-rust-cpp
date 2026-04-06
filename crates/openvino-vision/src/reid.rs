use crate::error::Result;
use crate::model::Device;
use crate::video::Frame;
use crate::Core;

pub use openvino_vision_sys::Detection;

/// Face gallery for re-identification.
pub struct FaceGallery {
    pub(crate) inner: cxx::UniquePtr<openvino_vision_sys::FaceGalleryWrapper>,
}

/// Builder for creating a FaceGallery.
pub struct FaceGalleryBuilder {
    gallery_path: String,
    fd_model: String,
    fd_device: Device,
    fd_confidence: f32,
    expand_ratio: f32,
    lm_model: String,
    lm_device: Device,
    reid_model: String,
    reid_device: Device,
    reid_threshold: f64,
    min_size_fr: i32,
    crop_gallery: bool,
    greedy_matching: bool,
}

impl FaceGalleryBuilder {
    pub fn new(
        gallery_path: impl Into<String>,
        fd_model: impl Into<String>,
        lm_model: impl Into<String>,
        reid_model: impl Into<String>,
    ) -> Self {
        Self {
            gallery_path: gallery_path.into(),
            fd_model: fd_model.into(),
            fd_device: Device::Cpu,
            fd_confidence: 0.6,
            expand_ratio: 1.15,
            lm_model: lm_model.into(),
            lm_device: Device::Cpu,
            reid_model: reid_model.into(),
            reid_device: Device::Cpu,
            reid_threshold: 0.7,
            min_size_fr: 128,
            crop_gallery: false,
            greedy_matching: false,
        }
    }

    pub fn fd_device(mut self, d: Device) -> Self { self.fd_device = d; self }
    pub fn fd_confidence(mut self, c: f32) -> Self { self.fd_confidence = c; self }
    pub fn expand_ratio(mut self, r: f32) -> Self { self.expand_ratio = r; self }
    pub fn lm_device(mut self, d: Device) -> Self { self.lm_device = d; self }
    pub fn reid_device(mut self, d: Device) -> Self { self.reid_device = d; self }
    pub fn reid_threshold(mut self, t: f64) -> Self { self.reid_threshold = t; self }
    pub fn min_size_fr(mut self, s: i32) -> Self { self.min_size_fr = s; self }
    pub fn crop_gallery(mut self, c: bool) -> Self { self.crop_gallery = c; self }
    pub fn greedy_matching(mut self, g: bool) -> Self { self.greedy_matching = g; self }

    pub fn build(self, core: &Core) -> Result<FaceGallery> {
        let inner = openvino_vision_sys::create_gallery(
            &core.inner,
            &self.gallery_path,
            &self.fd_model,
            self.fd_device.as_str(),
            self.fd_confidence,
            self.expand_ratio,
            &self.lm_model,
            self.lm_device.as_str(),
            &self.reid_model,
            self.reid_device.as_str(),
            self.reid_threshold,
            self.min_size_fr,
            self.crop_gallery,
            self.greedy_matching,
        )?;
        Ok(FaceGallery { inner })
    }
}

impl FaceGallery {
    pub fn builder(
        gallery_path: impl Into<String>,
        fd_model: impl Into<String>,
        lm_model: impl Into<String>,
        reid_model: impl Into<String>,
    ) -> FaceGalleryBuilder {
        FaceGalleryBuilder::new(gallery_path, fd_model, lm_model, reid_model)
    }

    /// Identify faces using a Frame reference.
    pub fn identify_frame(&mut self, frame: &Frame<'_>, faces: &Vec<Detection>) -> Result<Vec<i32>> {
        Ok(openvino_vision_sys::identify_faces_frame(self.inner.pin_mut(), &frame.inner, faces)?)
    }

    /// Get the label string for a gallery ID.
    pub fn label(&self, id: i32) -> Result<String> {
        Ok(openvino_vision_sys::get_gallery_label(&self.inner, id)?)
    }

    /// Number of identities in the gallery.
    pub fn size(&self) -> i32 {
        openvino_vision_sys::gallery_size(&self.inner)
    }

    /// Get all label strings in the gallery.
    pub fn all_labels(&self) -> Result<Vec<String>> {
        Ok(openvino_vision_sys::gallery_get_all_labels(&self.inner)?)
    }

    /// Check if a label exists in the gallery.
    pub fn label_exists(&self, label: &str) -> Result<bool> {
        Ok(openvino_vision_sys::gallery_label_exists(&self.inner, label)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gallery_builder_defaults() {
        let b = FaceGalleryBuilder::new("gallery.json", "fd.xml", "lm.xml", "reid.xml");
        assert_eq!(b.gallery_path, "gallery.json");
        assert_eq!(b.fd_model, "fd.xml");
        assert_eq!(b.lm_model, "lm.xml");
        assert_eq!(b.reid_model, "reid.xml");
        assert_eq!(b.fd_device, Device::Cpu);
        assert!((b.fd_confidence - 0.6).abs() < f32::EPSILON);
        assert!((b.expand_ratio - 1.15).abs() < f32::EPSILON);
        assert_eq!(b.lm_device, Device::Cpu);
        assert_eq!(b.reid_device, Device::Cpu);
        assert!((b.reid_threshold - 0.7).abs() < f64::EPSILON);
        assert_eq!(b.min_size_fr, 128);
        assert!(!b.crop_gallery);
        assert!(!b.greedy_matching);
    }

    #[test]
    fn test_gallery_builder_chaining() {
        let b = FaceGalleryBuilder::new("g.json", "fd.xml", "lm.xml", "reid.xml")
            .fd_device(Device::Gpu)
            .fd_confidence(0.8)
            .reid_threshold(0.5)
            .min_size_fr(64)
            .crop_gallery(true)
            .greedy_matching(true);
        assert_eq!(b.fd_device, Device::Gpu);
        assert!((b.fd_confidence - 0.8).abs() < f32::EPSILON);
        assert!((b.reid_threshold - 0.5).abs() < f64::EPSILON);
        assert_eq!(b.min_size_fr, 64);
        assert!(b.crop_gallery);
        assert!(b.greedy_matching);
    }
}
