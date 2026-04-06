// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// CXX bridge implementation for ovi-vision-sys crate.

#include "ovi/vision/ffi/bridge.hpp"

#include <iostream>

// ====== Helpers ======

static inline void require_frame(const FrameRef& frame, const char* func) {
    if (!frame.mat) {
        throw std::runtime_error(std::string(func) + ": frame reference is null");
    }
}

// ====== Factory implementations ======

std::unique_ptr<OvCore> create_core() {
    return std::make_unique<OvCore>();
}

std::unique_ptr<FrameDataWrapper> open_video(rust::Str path, bool loop_video, uint32_t read_limit) {
    std::string path_str(path.data(), path.size());
    auto wrapper = std::make_unique<FrameDataWrapper>();
    size_t limit = (read_limit == 0) ? ~size_t(0) : static_cast<size_t>(read_limit);
    wrapper->capture = openImagesCapture(path_str, loop_video, read_type::safe, 0, limit);
    wrapper->fps_val = wrapper->capture->fps();
    // Read first frame
    wrapper->current_frame = wrapper->capture->read();
    if (wrapper->current_frame.empty()) {
        throw std::runtime_error("Cannot read first frame from: " + path_str);
    }
    return wrapper;
}

bool read_frame(FrameDataWrapper& capture) {
    try {
        capture.current_frame = capture.capture->read();
        return !capture.current_frame.empty();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("read_frame failed: ") + e.what());
    }
}

int32_t frame_width(const FrameDataWrapper& capture) {
    return capture.current_frame.cols;
}

int32_t frame_height(const FrameDataWrapper& capture) {
    return capture.current_frame.rows;
}

double video_fps(const FrameDataWrapper& capture) {
    return capture.fps_val;
}

// ====== Face Detector ======

std::unique_ptr<FaceDetectorWrapper> create_face_detector(
    const OvCore& core, rust::Str model, rust::Str device,
    float confidence, int32_t input_h, int32_t input_w,
    float expand_ratio) {
    std::string model_str(model.data(), model.size());
    std::string device_str(device.data(), device.size());

    detection::DetectorConfig config(model_str);
    config.m_core = core.core;
    config.m_deviceName = device_str;
    config.confidence_threshold = confidence;
    config.input_h = input_h;
    config.input_w = input_w;
    config.increase_scale_x = expand_ratio;
    config.increase_scale_y = expand_ratio;
    config.is_async = true;

    return std::make_unique<FaceDetectorWrapper>(config);
}

// ====== Action Detector ======

std::unique_ptr<ActionDetectorWrapper> create_action_detector(
    const OvCore& core, rust::Str model, rust::Str device,
    float det_thresh, float act_thresh, uint32_t num_actions) {
    std::string model_str(model.data(), model.size());
    std::string device_str(device.data(), device.size());

    ActionDetectorConfig config(model_str, "Person/Action Detection");
    config.m_core = core.core;
    config.m_deviceName = device_str;
    config.detection_confidence_threshold = det_thresh;
    config.action_confidence_threshold = act_thresh;
    config.num_action_classes = num_actions;
    config.is_async = true;

    return std::make_unique<ActionDetectorWrapper>(config);
}

// ====== Tracker ======

std::unique_ptr<ObjectTrackerWrapper> create_tracker(const TrackerConfig& config) {
    try {
        TrackerParams params;
        params.min_track_duration = config.min_track_duration;
        params.forget_delay = config.forget_delay;
        params.affinity_thr = config.affinity_thr;
        params.shape_affinity_w = config.shape_affinity_w;
        params.motion_affinity_w = config.motion_affinity_w;
        params.min_det_conf = config.min_det_conf;
        params.bbox_heights_range = cv::Vec2f(config.bbox_heights_min, config.bbox_heights_max);
        params.drop_forgotten_tracks = config.drop_forgotten_tracks;
        params.max_num_objects_in_track = config.max_num_objects_in_track;
        params.averaging_window_size_for_rects = config.averaging_window_size_for_rects;
        params.averaging_window_size_for_labels = config.averaging_window_size_for_labels;

        return std::make_unique<ObjectTrackerWrapper>(params);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("create_tracker failed: ") + e.what());
    }
}

// ====== Gallery ======

std::unique_ptr<FaceGalleryWrapper> create_gallery(
    const OvCore& core,
    rust::Str gallery_path,
    rust::Str fd_model, rust::Str fd_device,
    float fd_confidence, float expand_ratio,
    rust::Str lm_model, rust::Str lm_device,
    rust::Str reid_model, rust::Str reid_device,
    double reid_threshold, int32_t min_size_fr, bool crop_gallery,
    bool greedy_matching) {

    auto wrapper = std::make_unique<FaceGalleryWrapper>();

    // Landmarks detector
    std::string lm_model_str(lm_model.data(), lm_model.size());
    std::string lm_device_str(lm_device.data(), lm_device.size());
    CnnConfig lm_config(lm_model_str, "Facial Landmarks");
    lm_config.m_core = core.core;
    lm_config.m_deviceName = lm_device_str;
    lm_config.m_max_batch_size = 16;
    wrapper->landmarks = std::make_unique<VectorCNN>(lm_config);

    // Re-ID detector
    std::string reid_model_str(reid_model.data(), reid_model.size());
    std::string reid_device_str(reid_device.data(), reid_device.size());
    CnnConfig reid_config(reid_model_str, "Face Re-ID");
    reid_config.m_core = core.core;
    reid_config.m_deviceName = reid_device_str;
    reid_config.m_max_batch_size = 16;
    wrapper->reid = std::make_unique<VectorCNN>(reid_config);

    // Face detector config for gallery registration
    std::string fd_model_str(fd_model.data(), fd_model.size());
    std::string fd_device_str(fd_device.data(), fd_device.size());
    detection::DetectorConfig fd_config(fd_model_str);
    fd_config.m_core = core.core;
    fd_config.m_deviceName = fd_device_str;
    fd_config.confidence_threshold = fd_confidence;
    fd_config.increase_scale_x = expand_ratio;
    fd_config.increase_scale_y = expand_ratio;
    fd_config.is_async = false;

    std::string gallery_str(gallery_path.data(), gallery_path.size());
    wrapper->gallery = std::make_unique<EmbeddingsGallery>(
        gallery_str, reid_threshold, min_size_fr, crop_gallery,
        fd_config, *wrapper->landmarks, *wrapper->reid, greedy_matching);

    return wrapper;
}

rust::String get_gallery_label(const FaceGalleryWrapper& gallery, int32_t id) {
    try {
        std::string label = gallery.gallery->GetLabelByID(id);
        return rust::String(label);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("get_gallery_label failed: ") + e.what());
    }
}

int32_t gallery_size(const FaceGalleryWrapper& gallery) {
    return static_cast<int32_t>(gallery.gallery->size());
}

// ====== Standalone landmarks detection ======

std::unique_ptr<LandmarksDetectorWrapper> create_landmarks_detector(
    const OvCore& core, rust::Str model, rust::Str device, int32_t max_batch_size) {
    try {
        std::string model_str(model.data(), model.size());
        std::string device_str(device.data(), device.size());
        CnnConfig config(model_str, "Facial Landmarks");
        config.m_core = core.core;
        config.m_deviceName = device_str;
        config.m_max_batch_size = max_batch_size;
        auto wrapper = std::make_unique<LandmarksDetectorWrapper>();
        wrapper->detector = std::make_unique<VectorCNN>(config);
        return wrapper;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("create_landmarks_detector failed: ") + e.what());
    }
}

rust::Vec<FaceLandmarks> compute_landmarks(
    LandmarksDetectorWrapper& det, const FrameRef& frame, const rust::Vec<Detection>& faces) {
    try {
        require_frame(frame, "compute_landmarks");
        std::vector<cv::Mat> face_rois;
        for (const auto& f : faces) {
            cv::Rect rect(f.x, f.y, f.width, f.height);
            cv::Rect clipped = rect & cv::Rect(0, 0, frame.mat->cols, frame.mat->rows);
            if (clipped.area() > 0) {
                face_rois.push_back((*frame.mat)(clipped));
            } else {
                face_rois.emplace_back();
            }
        }

        std::vector<cv::Mat> landmarks_raw;
        if (!face_rois.empty()) {
            det.detector->Compute(face_rois, &landmarks_raw, cv::Size(2, 5));
        }

        rust::Vec<FaceLandmarks> out;
        for (const auto& lm : landmarks_raw) {
            FaceLandmarks fl;
            rust::Vec<LandmarkPoint> pts;
            for (int i = 0; i < 5; ++i) {
                LandmarkPoint p;
                p.x = lm.at<float>(0, i);
                p.y = lm.at<float>(1, i);
                pts.push_back(p);
            }
            fl.points = std::move(pts);
            out.push_back(std::move(fl));
        }
        return out;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("compute_landmarks failed: ") + e.what());
    }
}

// ====== Gallery API expansion ======

rust::Vec<rust::String> gallery_get_all_labels(const FaceGalleryWrapper& gallery) {
    try {
        auto labels = gallery.gallery->GetIDToLabelMap();
        rust::Vec<rust::String> out;
        for (const auto& label : labels) {
            out.push_back(rust::String(label));
        }
        return out;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("gallery_get_all_labels failed: ") + e.what());
    }
}

bool gallery_label_exists(const FaceGalleryWrapper& gallery, rust::Str label) {
    try {
        std::string label_str(label.data(), label.size());
        return gallery.gallery->LabelExists(label_str);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("gallery_label_exists failed: ") + e.what());
    }
}

// ====== Frame-based API ======

std::unique_ptr<FrameRef> current_frame(const FrameDataWrapper& capture) {
    auto frame_ref = std::make_unique<FrameRef>();
    frame_ref->mat = &capture.current_frame;
    return frame_ref;
}

rust::Vec<Detection> detect_faces_frame(FaceDetectorWrapper& det, const FrameRef& frame) {
    try {
        require_frame(frame, "detect_faces_frame");
        det.detector->enqueue(*frame.mat);
        det.detector->submitRequest();
        det.detector->wait();
        auto results = det.detector->fetchResults();

        rust::Vec<Detection> out;
        for (const auto& r : results) {
            Detection d;
            d.x = r.rect.x;
            d.y = r.rect.y;
            d.width = r.rect.width;
            d.height = r.rect.height;
            d.confidence = r.confidence;
            d.label = -1;
            out.push_back(d);
        }
        return out;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("detect_faces_frame failed: ") + e.what());
    }
}

rust::Vec<ActionResult> detect_actions_frame(ActionDetectorWrapper& det, const FrameRef& frame) {
    try {
        require_frame(frame, "detect_actions_frame");
        det.detector->enqueue(*frame.mat);
        det.detector->submitRequest();
        det.detector->wait();
        auto results = det.detector->fetchResults();

        rust::Vec<ActionResult> out;
        for (const auto& r : results) {
            ActionResult a;
            a.x = r.rect.x;
            a.y = r.rect.y;
            a.width = r.rect.width;
            a.height = r.rect.height;
            a.detection_conf = r.detection_conf;
            a.action_conf = r.action_conf;
            a.label = r.label;
            out.push_back(a);
        }
        return out;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("detect_actions_frame failed: ") + e.what());
    }
}

rust::Vec<TrackedResult> track_frame(ObjectTrackerWrapper& tracker,
                                     const rust::Vec<Detection>& detections,
                                     int32_t frame_idx,
                                     const FrameRef& frame) {
    try {
        require_frame(frame, "track_frame");
        TrackedObjects objs;
        for (const auto& d : detections) {
            TrackedObject obj(cv::Rect(d.x, d.y, d.width, d.height), d.confidence, d.label);
            objs.push_back(obj);
        }

        tracker.tracker.Process(*frame.mat, objs, frame_idx);
        auto tracked = tracker.tracker.TrackedDetectionsWithLabels();

        rust::Vec<TrackedResult> out;
        for (const auto& t : tracked) {
            TrackedResult r;
            r.x = t.rect.x;
            r.y = t.rect.y;
            r.width = t.rect.width;
            r.height = t.rect.height;
            r.confidence = t.confidence;
            r.object_id = t.object_id;
            r.label = t.label;
            out.push_back(r);
        }
        return out;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("track_frame failed: ") + e.what());
    }
}

rust::Vec<int32_t> identify_faces_frame(
    FaceGalleryWrapper& gallery,
    const FrameRef& frame,
    const rust::Vec<Detection>& faces) {
    try {
        require_frame(frame, "identify_faces_frame");
        std::vector<cv::Mat> face_rois;
        for (const auto& f : faces) {
            cv::Rect rect(f.x, f.y, f.width, f.height);
            cv::Rect clipped = rect & cv::Rect(0, 0, frame.mat->cols, frame.mat->rows);
            if (clipped.area() > 0) {
                face_rois.push_back((*frame.mat)(clipped));
            } else {
                face_rois.emplace_back();
            }
        }

        std::vector<cv::Mat> landmarks, embeddings;
        if (!face_rois.empty()) {
            gallery.landmarks->Compute(face_rois, &landmarks, cv::Size(2, 5));
            AlignFaces(&face_rois, &landmarks);
            gallery.reid->Compute(face_rois, &embeddings);
        }

        auto ids = gallery.gallery->GetIDsByEmbeddings(embeddings);

        rust::Vec<int32_t> out;
        for (int id : ids) {
            out.push_back(id);
        }
        return out;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("identify_faces_frame failed: ") + e.what());
    }
}

// ====== Tracker API expansion ======

rust::Vec<ActiveTrack> tracker_get_active_tracks(const ObjectTrackerWrapper& tracker) {
    try {
        auto active = tracker.tracker.GetActiveTracks();
        rust::Vec<ActiveTrack> out;
        for (const auto& [id, points] : active) {
            ActiveTrack at;
            at.track_id = static_cast<int64_t>(id);
            rust::Vec<TrackPoint> pts;
            for (const auto& p : points) {
                TrackPoint tp;
                tp.x = p.x;
                tp.y = p.y;
                pts.push_back(tp);
            }
            at.points = std::move(pts);
            out.push_back(std::move(at));
        }
        return out;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("tracker_get_active_tracks failed: ") + e.what());
    }
}

bool tracker_is_track_valid(const ObjectTrackerWrapper& tracker, int64_t id) {
    try {
        return tracker.tracker.IsTrackValid(static_cast<size_t>(id));
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("tracker_is_track_valid failed: ") + e.what());
    }
}

bool tracker_is_track_forgotten(const ObjectTrackerWrapper& tracker, int64_t id) {
    try {
        return tracker.tracker.IsTrackForgotten(static_cast<size_t>(id));
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("tracker_is_track_forgotten failed: ") + e.what());
    }
}

void tracker_drop_forgotten_tracks(ObjectTrackerWrapper& tracker) {
    try {
        tracker.tracker.DropForgottenTracks();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("tracker_drop_forgotten_tracks failed: ") + e.what());
    }
}

void tracker_reset(ObjectTrackerWrapper& tracker) {
    try {
        tracker.tracker.Reset();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("tracker_reset failed: ") + e.what());
    }
}

// ====== Async pipeline ======

void enqueue_face_detection(FaceDetectorWrapper& det, const FrameRef& frame) {
    try {
        require_frame(frame, "enqueue_face_detection");
        det.detector->enqueue(*frame.mat);
        det.detector->submitRequest();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("enqueue_face_detection failed: ") + e.what());
    }
}

rust::Vec<Detection> fetch_face_results(FaceDetectorWrapper& det) {
    try {
        det.detector->wait();
        auto results = det.detector->fetchResults();

        rust::Vec<Detection> out;
        for (const auto& r : results) {
            Detection d;
            d.x = r.rect.x;
            d.y = r.rect.y;
            d.width = r.rect.width;
            d.height = r.rect.height;
            d.confidence = r.confidence;
            d.label = -1;
            out.push_back(d);
        }
        return out;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("fetch_face_results failed: ") + e.what());
    }
}

void enqueue_action_detection(ActionDetectorWrapper& det, const FrameRef& frame) {
    try {
        require_frame(frame, "enqueue_action_detection");
        det.detector->enqueue(*frame.mat);
        det.detector->submitRequest();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("enqueue_action_detection failed: ") + e.what());
    }
}

rust::Vec<ActionResult> fetch_action_results(ActionDetectorWrapper& det) {
    try {
        det.detector->wait();
        auto results = det.detector->fetchResults();

        rust::Vec<ActionResult> out;
        for (const auto& r : results) {
            ActionResult a;
            a.x = r.rect.x;
            a.y = r.rect.y;
            a.width = r.rect.width;
            a.height = r.rect.height;
            a.detection_conf = r.detection_conf;
            a.action_conf = r.action_conf;
            a.label = r.label;
            out.push_back(a);
        }
        return out;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("fetch_action_results failed: ") + e.what());
    }
}

// ====== Video output — annotated frame drawing + writing ======

std::unique_ptr<AnnotatedFrame> clone_frame_for_drawing(const FrameRef& frame) {
    try {
        require_frame(frame, "clone_frame_for_drawing");
        auto af = std::make_unique<AnnotatedFrame>();
        af->mat = frame.mat->clone();
        return af;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("clone_frame_for_drawing failed: ") + e.what());
    }
}

void draw_detection(AnnotatedFrame& frame, int32_t x, int32_t y, int32_t w, int32_t h,
                    rust::Str label, uint8_t r, uint8_t g, uint8_t b) {
    try {
        cv::Scalar color(b, g, r);  // OpenCV uses BGR
        cv::Rect rect(x, y, w, h);
        cv::rectangle(frame.mat, rect, color, 2);

        std::string label_str(label.data(), label.size());
        if (!label_str.empty()) {
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(label_str, cv::FONT_HERSHEY_PLAIN, 1, 1, &baseLine);
            cv::rectangle(frame.mat,
                cv::Point(x, y - label_size.height - baseLine),
                cv::Point(x + label_size.width, y),
                color, cv::FILLED);
            cv::putText(frame.mat, label_str, cv::Point(x, y - baseLine / 2),
                        cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("draw_detection failed: ") + e.what());
    }
}

std::unique_ptr<VideoWriterWrapper> create_video_writer(rust::Str path, double fps) {
    try {
        auto wrapper = std::make_unique<VideoWriterWrapper>();
        wrapper->path = std::string(path.data(), path.size());
        wrapper->fps = fps;
        wrapper->initialized = false;
        return wrapper;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("create_video_writer failed: ") + e.what());
    }
}

void write_frame(VideoWriterWrapper& writer, const AnnotatedFrame& frame) {
    try {
        if (!writer.initialized) {
            writer.writer = cv::VideoWriter(
                writer.path,
                cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                writer.fps,
                frame.mat.size());
            if (!writer.writer.isOpened()) {
                throw std::runtime_error("Cannot open video writer: " + writer.path);
            }
            writer.initialized = true;
        }
        writer.writer.write(frame.mat);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("write_frame failed: ") + e.what());
    }
}

// Dummy main — not called when linked into Rust
int main() { return 1; }
