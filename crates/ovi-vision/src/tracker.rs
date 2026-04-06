use crate::error::Result;
use crate::video::Frame;

pub use ovi_vision_sys::{ActiveTrack, Detection, TrackPoint, TrackedResult};

/// Configuration for the object tracker.
#[derive(Debug, Clone)]
pub struct TrackerConfig {
    pub min_track_duration: u32,
    pub forget_delay: u32,
    pub affinity_thr: f32,
    pub shape_affinity_w: f32,
    pub motion_affinity_w: f32,
    pub min_det_conf: f32,
    pub bbox_heights_range: (f32, f32),
    pub drop_forgotten_tracks: bool,
    pub max_num_objects_in_track: i32,
    pub averaging_window_size_for_rects: i32,
    pub averaging_window_size_for_labels: i32,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            min_track_duration: 25,
            forget_delay: 150,
            affinity_thr: 0.85,
            shape_affinity_w: 0.5,
            motion_affinity_w: 0.2,
            min_det_conf: 0.0,
            bbox_heights_range: (1.0, 1280.0),
            drop_forgotten_tracks: true,
            max_num_objects_in_track: 300,
            averaging_window_size_for_rects: 1,
            averaging_window_size_for_labels: 1,
        }
    }
}

/// Multi-object tracker using Hungarian algorithm.
pub struct Tracker {
    pub(crate) inner: cxx::UniquePtr<ovi_vision_sys::ObjectTrackerWrapper>,
}

impl Tracker {
    pub fn new(config: &TrackerConfig) -> Result<Self> {
        let sys_config = ovi_vision_sys::TrackerConfig {
            min_track_duration: config.min_track_duration,
            forget_delay: config.forget_delay,
            affinity_thr: config.affinity_thr,
            shape_affinity_w: config.shape_affinity_w,
            motion_affinity_w: config.motion_affinity_w,
            min_det_conf: config.min_det_conf,
            bbox_heights_min: config.bbox_heights_range.0,
            bbox_heights_max: config.bbox_heights_range.1,
            drop_forgotten_tracks: config.drop_forgotten_tracks,
            max_num_objects_in_track: config.max_num_objects_in_track,
            averaging_window_size_for_rects: config.averaging_window_size_for_rects,
            averaging_window_size_for_labels: config.averaging_window_size_for_labels,
        };
        let inner = ovi_vision_sys::create_tracker(&sys_config)?;
        Ok(Self { inner })
    }

    /// Process detections on a Frame reference.
    pub fn process_frame(
        &mut self,
        detections: &Vec<Detection>,
        frame_idx: i32,
        frame: &Frame<'_>,
    ) -> Result<Vec<TrackedResult>> {
        Ok(ovi_vision_sys::track_frame(self.inner.pin_mut(), detections, frame_idx, &frame.inner)?)
    }

    /// Get active tracks with their trajectory points.
    pub fn active_tracks(&self) -> Result<Vec<ActiveTrack>> {
        Ok(ovi_vision_sys::tracker_get_active_tracks(&self.inner)?)
    }

    /// Check if a track has lasted long enough to be considered valid.
    pub fn is_track_valid(&self, id: i64) -> Result<bool> {
        Ok(ovi_vision_sys::tracker_is_track_valid(&self.inner, id)?)
    }

    /// Check if a track has been forgotten (lost for too many frames).
    pub fn is_track_forgotten(&self, id: i64) -> Result<bool> {
        Ok(ovi_vision_sys::tracker_is_track_forgotten(&self.inner, id)?)
    }

    /// Remove tracks that have been forgotten.
    pub fn drop_forgotten_tracks(&mut self) -> Result<()> {
        Ok(ovi_vision_sys::tracker_drop_forgotten_tracks(self.inner.pin_mut())?)
    }

    /// Reset the tracker, clearing all tracks.
    pub fn reset(&mut self) -> Result<()> {
        Ok(ovi_vision_sys::tracker_reset(self.inner.pin_mut())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_config_default() {
        let config = TrackerConfig::default();
        assert_eq!(config.min_track_duration, 25);
        assert_eq!(config.forget_delay, 150);
        assert!((config.affinity_thr - 0.85).abs() < f32::EPSILON);
        assert!((config.shape_affinity_w - 0.5).abs() < f32::EPSILON);
        assert!((config.motion_affinity_w - 0.2).abs() < f32::EPSILON);
        assert!((config.min_det_conf - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.bbox_heights_range, (1.0, 1280.0));
        assert!(config.drop_forgotten_tracks);
        assert_eq!(config.max_num_objects_in_track, 300);
        assert_eq!(config.averaging_window_size_for_rects, 1);
        assert_eq!(config.averaging_window_size_for_labels, 1);
    }
}
