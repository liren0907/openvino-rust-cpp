use std::marker::PhantomData;

use crate::error::Result;

/// Video capture wrapper.
pub struct Video {
    pub(crate) inner: cxx::UniquePtr<openvino_vision_sys::FrameDataWrapper>,
}

/// An opaque reference to the current video frame.
///
/// Borrows the internal frame buffer of a [`Video`]. The borrow checker
/// prevents calling [`Video::read()`] or [`Video::read_frame()`] while
/// a `Frame` is alive, so stale-pointer bugs are caught at compile time.
pub struct Frame<'a> {
    pub(crate) inner: cxx::UniquePtr<openvino_vision_sys::FrameRef>,
    _borrow: PhantomData<&'a Video>,
}

impl Video {
    /// Open a video file or image.
    pub fn open(path: &str) -> Result<Self> {
        let inner = openvino_vision_sys::open_video(path, false, 0)?;
        Ok(Self { inner })
    }

    /// Open a camera device by index (e.g., 0 for the default camera).
    pub fn open_camera(device_index: u32) -> Result<Self> {
        let path = device_index.to_string();
        let inner = openvino_vision_sys::open_video(&path, false, 0)?;
        Ok(Self { inner })
    }

    /// Open with loop and frame limit options.
    pub fn open_with_options(path: &str, loop_video: bool, read_limit: u32) -> Result<Self> {
        let inner = openvino_vision_sys::open_video(path, loop_video, read_limit)?;
        Ok(Self { inner })
    }

    /// Read the next frame. Returns false when the video ends.
    ///
    /// Cannot be called while a [`Frame`] obtained from this `Video` is alive.
    pub fn read(&mut self) -> Result<bool> {
        Ok(openvino_vision_sys::read_frame(self.inner.pin_mut())?)
    }

    /// Get a reference to the current frame.
    ///
    /// The returned `Frame` borrows this `Video`, preventing `read()` from
    /// being called until the `Frame` is dropped.
    pub fn current_frame(&self) -> Result<Frame<'_>> {
        let inner = openvino_vision_sys::current_frame(&self.inner)?;
        Ok(Frame { inner, _borrow: PhantomData })
    }

    /// Read the next frame and return it, or `None` if the video has ended.
    pub fn read_frame(&mut self) -> Result<Option<Frame<'_>>> {
        if self.read()? {
            let inner = openvino_vision_sys::current_frame(&self.inner)?;
            Ok(Some(Frame { inner, _borrow: PhantomData }))
        } else {
            Ok(None)
        }
    }

    /// Width of the current frame.
    pub fn width(&self) -> i32 {
        openvino_vision_sys::frame_width(&self.inner)
    }

    /// Height of the current frame.
    pub fn height(&self) -> i32 {
        openvino_vision_sys::frame_height(&self.inner)
    }

    /// Video FPS.
    pub fn fps(&self) -> f64 {
        openvino_vision_sys::video_fps(&self.inner)
    }
}
