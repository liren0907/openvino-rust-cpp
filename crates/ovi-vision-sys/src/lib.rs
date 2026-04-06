#[cxx::bridge]
mod ffi {
    /// A detected bounding box with confidence.
    #[derive(Debug, Clone)]
    struct Detection {
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        confidence: f32,
        label: i32,
    }

    /// Result from action detection.
    #[derive(Debug, Clone)]
    struct ActionResult {
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        detection_conf: f32,
        action_conf: f32,
        label: i32,
    }

    /// Result from tracking.
    #[derive(Debug, Clone)]
    struct TrackedResult {
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        confidence: f32,
        object_id: i32,
        label: i32,
    }

    /// A 2D point from a track's trajectory.
    #[derive(Debug, Clone)]
    struct TrackPoint {
        x: i32,
        y: i32,
    }

    /// An active track with its trajectory points.
    #[derive(Debug, Clone)]
    struct ActiveTrack {
        track_id: i64,
        points: Vec<TrackPoint>,
    }

    /// A single facial landmark point (normalized coordinates).
    #[derive(Debug, Clone)]
    struct LandmarkPoint {
        x: f32,
        y: f32,
    }

    /// Facial landmarks for a single detected face.
    #[derive(Debug, Clone)]
    struct FaceLandmarks {
        points: Vec<LandmarkPoint>,
    }

    /// Tracker configuration passed from Rust to C++.
    #[derive(Debug, Clone)]
    struct TrackerConfig {
        min_track_duration: u32,
        forget_delay: u32,
        affinity_thr: f32,
        shape_affinity_w: f32,
        motion_affinity_w: f32,
        min_det_conf: f32,
        bbox_heights_min: f32,
        bbox_heights_max: f32,
        drop_forgotten_tracks: bool,
        max_num_objects_in_track: i32,
        averaging_window_size_for_rects: i32,
        averaging_window_size_for_labels: i32,
    }

    unsafe extern "C++" {
        include!("ovi/vision/ffi/bridge.hpp");

        // Opaque types
        type OvCore;
        type FrameDataWrapper;
        type FaceDetectorWrapper;
        type ActionDetectorWrapper;
        type ObjectTrackerWrapper;
        type LandmarksDetectorWrapper;
        type FaceGalleryWrapper;
        type FrameRef;

        // Core
        fn create_core() -> Result<UniquePtr<OvCore>>;

        // Video I/O
        fn open_video(path: &str, loop_video: bool, read_limit: u32) -> Result<UniquePtr<FrameDataWrapper>>;
        fn read_frame(capture: Pin<&mut FrameDataWrapper>) -> Result<bool>;
        fn frame_width(capture: &FrameDataWrapper) -> i32;
        fn frame_height(capture: &FrameDataWrapper) -> i32;
        fn video_fps(capture: &FrameDataWrapper) -> f64;
        fn current_frame(capture: &FrameDataWrapper) -> Result<UniquePtr<FrameRef>>;

        // Face Detection
        fn create_face_detector(
            core: &OvCore, model: &str, device: &str,
            confidence: f32, input_h: i32, input_w: i32,
            expand_ratio: f32,
        ) -> Result<UniquePtr<FaceDetectorWrapper>>;
        fn detect_faces_frame(det: Pin<&mut FaceDetectorWrapper>, frame: &FrameRef) -> Result<Vec<Detection>>;

        // Action Detection
        fn create_action_detector(
            core: &OvCore, model: &str, device: &str,
            det_thresh: f32, act_thresh: f32, num_actions: u32,
        ) -> Result<UniquePtr<ActionDetectorWrapper>>;
        fn detect_actions_frame(det: Pin<&mut ActionDetectorWrapper>, frame: &FrameRef) -> Result<Vec<ActionResult>>;

        // Tracking
        fn create_tracker(config: &TrackerConfig) -> Result<UniquePtr<ObjectTrackerWrapper>>;
        fn track_frame(
            tracker: Pin<&mut ObjectTrackerWrapper>,
            detections: &Vec<Detection>,
            frame_idx: i32,
            frame: &FrameRef,
        ) -> Result<Vec<TrackedResult>>;

        // Tracker API expansion
        fn tracker_get_active_tracks(tracker: &ObjectTrackerWrapper) -> Result<Vec<ActiveTrack>>;
        fn tracker_is_track_valid(tracker: &ObjectTrackerWrapper, id: i64) -> Result<bool>;
        fn tracker_is_track_forgotten(tracker: &ObjectTrackerWrapper, id: i64) -> Result<bool>;
        fn tracker_drop_forgotten_tracks(tracker: Pin<&mut ObjectTrackerWrapper>) -> Result<()>;
        fn tracker_reset(tracker: Pin<&mut ObjectTrackerWrapper>) -> Result<()>;

        // Standalone landmarks detection
        fn create_landmarks_detector(
            core: &OvCore, model: &str, device: &str, max_batch_size: i32,
        ) -> Result<UniquePtr<LandmarksDetectorWrapper>>;
        fn compute_landmarks(
            det: Pin<&mut LandmarksDetectorWrapper>,
            frame: &FrameRef,
            faces: &Vec<Detection>,
        ) -> Result<Vec<FaceLandmarks>>;

        // Gallery (landmarks + reid + gallery combined)
        fn create_gallery(
            core: &OvCore,
            gallery_path: &str,
            fd_model: &str, fd_device: &str,
            fd_confidence: f32, expand_ratio: f32,
            lm_model: &str, lm_device: &str,
            reid_model: &str, reid_device: &str,
            reid_threshold: f64, min_size_fr: i32, crop_gallery: bool,
            greedy_matching: bool,
        ) -> Result<UniquePtr<FaceGalleryWrapper>>;
        fn identify_faces_frame(
            gallery: Pin<&mut FaceGalleryWrapper>,
            frame: &FrameRef,
            faces: &Vec<Detection>,
        ) -> Result<Vec<i32>>;
        fn get_gallery_label(gallery: &FaceGalleryWrapper, id: i32) -> Result<String>;
        fn gallery_size(gallery: &FaceGalleryWrapper) -> i32;
        fn gallery_get_all_labels(gallery: &FaceGalleryWrapper) -> Result<Vec<String>>;
        fn gallery_label_exists(gallery: &FaceGalleryWrapper, label: &str) -> Result<bool>;

        // Async pipeline
        fn enqueue_face_detection(det: Pin<&mut FaceDetectorWrapper>, frame: &FrameRef) -> Result<()>;
        fn fetch_face_results(det: Pin<&mut FaceDetectorWrapper>) -> Result<Vec<Detection>>;
        fn enqueue_action_detection(det: Pin<&mut ActionDetectorWrapper>, frame: &FrameRef) -> Result<()>;
        fn fetch_action_results(det: Pin<&mut ActionDetectorWrapper>) -> Result<Vec<ActionResult>>;

        // Video output — annotated frame drawing + writing
        type AnnotatedFrame;
        type VideoWriterWrapper;

        fn clone_frame_for_drawing(frame: &FrameRef) -> Result<UniquePtr<AnnotatedFrame>>;
        fn draw_detection(
            frame: Pin<&mut AnnotatedFrame>,
            x: i32, y: i32, w: i32, h: i32,
            label: &str,
            r: u8, g: u8, b: u8,
        ) -> Result<()>;
        fn create_video_writer(path: &str, fps: f64) -> Result<UniquePtr<VideoWriterWrapper>>;
        fn write_frame(writer: Pin<&mut VideoWriterWrapper>, frame: &AnnotatedFrame) -> Result<()>;
    }
}

// Re-export everything
pub use ffi::*;
