use crate::error::Result;
use crate::video::Frame;

/// A mutable clone of a video frame for drawing annotations.
pub struct AnnotatedFrame {
    pub(crate) inner: cxx::UniquePtr<ovi_vision_sys::AnnotatedFrame>,
}

/// Color for drawing operations (RGB).
#[derive(Debug, Clone, Copy)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const GREEN: Color = Color { r: 0, g: 255, b: 0 };
    pub const RED: Color = Color { r: 255, g: 0, b: 0 };
    pub const BLUE: Color = Color { r: 0, g: 0, b: 255 };
    pub const YELLOW: Color = Color { r: 255, g: 255, b: 0 };
    pub const CYAN: Color = Color { r: 0, g: 255, b: 255 };
    pub const WHITE: Color = Color { r: 255, g: 255, b: 255 };
}

impl AnnotatedFrame {
    /// Clone the current video frame for drawing annotations.
    pub fn from_frame(frame: &Frame<'_>) -> Result<Self> {
        let inner = ovi_vision_sys::clone_frame_for_drawing(&frame.inner)?;
        Ok(Self { inner })
    }

    /// Draw a bounding box with an optional text label.
    pub fn draw_detection(
        &mut self,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        label: &str,
        color: Color,
    ) -> Result<()> {
        ovi_vision_sys::draw_detection(
            self.inner.pin_mut(),
            x,
            y,
            width,
            height,
            label,
            color.r,
            color.g,
            color.b,
        )?;
        Ok(())
    }
}

/// Writes annotated frames to an output MP4 video file.
pub struct VideoWriter {
    inner: cxx::UniquePtr<ovi_vision_sys::VideoWriterWrapper>,
}

impl VideoWriter {
    /// Create a new video writer. The output file is created lazily on the first write.
    pub fn new(path: &str, fps: f64) -> Result<Self> {
        let inner = ovi_vision_sys::create_video_writer(path, fps)?;
        Ok(Self { inner })
    }

    /// Write an annotated frame to the output video.
    pub fn write(&mut self, frame: &AnnotatedFrame) -> Result<()> {
        ovi_vision_sys::write_frame(self.inner.pin_mut(), &frame.inner)?;
        Ok(())
    }
}
