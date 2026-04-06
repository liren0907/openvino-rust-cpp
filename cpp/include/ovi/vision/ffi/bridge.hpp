// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// CXX bridge header for ovi-vision-sys crate.

#pragma once

#define CXX_RS_CPP_VERSION 17
#define CXXBRIDGE_CXX_ABI_VERSION 1

#include <string>
#include <memory>
#include <vector>

#include "openvino/openvino.hpp"
#include "ovi/vision/cnn.hpp"
#include "ovi/vision/detector.hpp"
#include "ovi/vision/action_detector.hpp"
#include "ovi/vision/tracker.hpp"
#include "ovi/vision/face_reid.hpp"
#include "ovi/core/images_capture.hpp"

// ====== Opaque wrapper types — MUST be defined BEFORE lib.rs.h ======

struct OvCore {
    ov::Core core;
};

struct FrameDataWrapper {
    std::unique_ptr<ImagesCapture> capture;
    cv::Mat current_frame;
    double fps_val;
};

struct FaceDetectorWrapper {
    detection::DetectorConfig config;
    std::unique_ptr<detection::FaceDetection> detector;
    FaceDetectorWrapper(const detection::DetectorConfig& cfg)
        : config(cfg), detector(std::make_unique<detection::FaceDetection>(cfg)) {}
};

struct ActionDetectorWrapper {
    ActionDetectorConfig config;
    std::unique_ptr<ActionDetection> detector;
    ActionDetectorWrapper(const ActionDetectorConfig& cfg)
        : config(cfg), detector(std::make_unique<ActionDetection>(cfg)) {}
};

struct ObjectTrackerWrapper {
    TrackerParams params;
    Tracker tracker;
    ObjectTrackerWrapper(const TrackerParams& p) : params(p), tracker(p) {}
};

struct LandmarksDetectorWrapper {
    std::unique_ptr<VectorCNN> detector;
};

struct FaceReidentifierWrapper {
    std::unique_ptr<VectorCNN> detector;
};

struct FaceGalleryWrapper {
    std::unique_ptr<VectorCNN> landmarks;
    std::unique_ptr<VectorCNN> reid;
    std::unique_ptr<EmbeddingsGallery> gallery;
};

struct FrameRef {
    const cv::Mat* mat;
};

struct AnnotatedFrame {
    cv::Mat mat;
};

struct VideoWriterWrapper {
    cv::VideoWriter writer;
    bool initialized;
    std::string path;
    double fps;
};

// ====== Include CXX generated header AFTER type definitions ======
#include "ovi-vision-sys/src/lib.rs.h"

// ====== Factory / lifecycle function declarations ======

std::unique_ptr<OvCore> create_core();

std::unique_ptr<FrameDataWrapper> open_video(rust::Str path, bool loop_video, uint32_t read_limit);
bool read_frame(FrameDataWrapper& capture);
int32_t frame_width(const FrameDataWrapper& capture);
int32_t frame_height(const FrameDataWrapper& capture);
double video_fps(const FrameDataWrapper& capture);

std::unique_ptr<FaceDetectorWrapper> create_face_detector(
    const OvCore& core, rust::Str model, rust::Str device,
    float confidence, int32_t input_h, int32_t input_w,
    float expand_ratio);

std::unique_ptr<ActionDetectorWrapper> create_action_detector(
    const OvCore& core, rust::Str model, rust::Str device,
    float det_thresh, float act_thresh, uint32_t num_actions);

std::unique_ptr<ObjectTrackerWrapper> create_tracker(const TrackerConfig& config);

std::unique_ptr<FaceGalleryWrapper> create_gallery(
    const OvCore& core,
    rust::Str gallery_path,
    rust::Str fd_model, rust::Str fd_device,
    float fd_confidence, float expand_ratio,
    rust::Str lm_model, rust::Str lm_device,
    rust::Str reid_model, rust::Str reid_device,
    double reid_threshold, int32_t min_size_fr, bool crop_gallery,
    bool greedy_matching);
rust::String get_gallery_label(const FaceGalleryWrapper& gallery, int32_t id);
int32_t gallery_size(const FaceGalleryWrapper& gallery);

// Standalone landmarks detection
std::unique_ptr<LandmarksDetectorWrapper> create_landmarks_detector(
    const OvCore& core, rust::Str model, rust::Str device, int32_t max_batch_size);
rust::Vec<FaceLandmarks> compute_landmarks(
    LandmarksDetectorWrapper& det, const FrameRef& frame, const rust::Vec<Detection>& faces);

// Standalone face embedding extraction
std::unique_ptr<FaceReidentifierWrapper> create_face_embedder(
    const OvCore& core, rust::Str model, rust::Str device, int32_t max_batch_size);
rust::Vec<FaceEmbedding> compute_embeddings(
    FaceReidentifierWrapper& embedder, const FrameRef& frame, const rust::Vec<Detection>& faces);

// Gallery API expansion
rust::Vec<rust::String> gallery_get_all_labels(const FaceGalleryWrapper& gallery);
bool gallery_label_exists(const FaceGalleryWrapper& gallery, rust::Str label);

// Frame-based API
std::unique_ptr<FrameRef> current_frame(const FrameDataWrapper& capture);
rust::Vec<Detection> detect_faces_frame(FaceDetectorWrapper& det, const FrameRef& frame);
rust::Vec<ActionResult> detect_actions_frame(ActionDetectorWrapper& det, const FrameRef& frame);
rust::Vec<TrackedResult> track_frame(ObjectTrackerWrapper& tracker,
                                     const rust::Vec<Detection>& detections,
                                     int32_t frame_idx,
                                     const FrameRef& frame);
rust::Vec<int32_t> identify_faces_frame(
    FaceGalleryWrapper& gallery,
    const FrameRef& frame,
    const rust::Vec<Detection>& faces);

// Async pipeline
void enqueue_face_detection(FaceDetectorWrapper& det, const FrameRef& frame);
rust::Vec<Detection> fetch_face_results(FaceDetectorWrapper& det);
void enqueue_action_detection(ActionDetectorWrapper& det, const FrameRef& frame);
rust::Vec<ActionResult> fetch_action_results(ActionDetectorWrapper& det);

// Tracker API expansion
rust::Vec<ActiveTrack> tracker_get_active_tracks(const ObjectTrackerWrapper& tracker);
bool tracker_is_track_valid(const ObjectTrackerWrapper& tracker, int64_t id);
bool tracker_is_track_forgotten(const ObjectTrackerWrapper& tracker, int64_t id);
void tracker_drop_forgotten_tracks(ObjectTrackerWrapper& tracker);
void tracker_reset(ObjectTrackerWrapper& tracker);

// Video output — annotated frame drawing + writing
std::unique_ptr<AnnotatedFrame> clone_frame_for_drawing(const FrameRef& frame);
void draw_detection(AnnotatedFrame& frame, int32_t x, int32_t y, int32_t w, int32_t h,
                    rust::Str label, uint8_t r, uint8_t g, uint8_t b);
std::unique_ptr<VideoWriterWrapper> create_video_writer(rust::Str path, double fps);
void write_frame(VideoWriterWrapper& writer, const AnnotatedFrame& frame);
