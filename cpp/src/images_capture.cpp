// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino_vision/common/images_capture.hpp"
#include "openvino_vision/common/slog.hpp"

class VideoCapture : public ImagesCapture {
    cv::VideoCapture cap;
    double fps_;
    size_t frameCount = 0;
    size_t readLimit;
    bool loop;
    std::string name;

public:
    VideoCapture(const std::string& name, bool loop, size_t readLimit, size_t firstFrameId)
        : fps_(30), readLimit(readLimit), loop(loop), name(name) {

        slog::info << "Opening video: " << name << slog::endl;

        cap.open(name);

        if (!cap.isOpened()) {
            slog::err << "Failed to open video: " << name << slog::endl;
            throw std::runtime_error("Cannot open video: " + name);
        }

        fps_ = cap.get(cv::CAP_PROP_FPS);
        auto width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        auto height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        auto count = cap.get(cv::CAP_PROP_FRAME_COUNT);

        slog::info << "Video details: " << width << "x" << height
                  << ", FPS: " << fps_
                  << ", Frame count: " << count << slog::endl;

        if (firstFrameId > 0) {
            slog::info << "Setting start frame to: " << firstFrameId << slog::endl;
            cap.set(cv::CAP_PROP_POS_FRAMES, firstFrameId);
        }
    }

    double fps() const override {
        return fps_;
    }

    cv::Mat read() override {
        if (readLimit != ~0 && frameCount >= readLimit) {
            slog::info << "Reached frame limit: " << readLimit << slog::endl;
            return cv::Mat();
        }

        cv::Mat img;
        bool success = cap.read(img);

        if (!success) {
            slog::info << "Failed to read frame " << frameCount << slog::endl;
            if (loop) {
                slog::info << "Reopening video: " << name << slog::endl;
                cap.open(name);
                success = cap.read(img);
                if (!success) {
                    slog::err << "Failed to read frame after reopening video" << slog::endl;
                    return cv::Mat();
                }
            } else {
                return cv::Mat();
            }
        }

        if (img.empty()) {
            slog::info << "Empty frame " << frameCount << slog::endl;
        } else {
            slog::debug << "Read frame " << frameCount << ": " << img.cols << "x" << img.rows << slog::endl;
        }

        frameCount++;
        return img;
    }
};

class ImageCapture : public ImagesCapture {
    cv::Mat img;
    bool canRead = true;
    double fps_;

public:
    ImageCapture(const std::string& name, double fps) : fps_(fps) {
        img = cv::imread(name);
        if (img.empty()) {
            throw std::runtime_error("Cannot open image: " + name);
        }
    }

    double fps() const override {
        return fps_;
    }

    cv::Mat read() override {
        if (!canRead) {
            return cv::Mat();
        }

        canRead = false;
        return img.clone();
    }
};

std::unique_ptr<ImagesCapture> openImagesCapture(const std::string& source,
                                                bool loop,
                                                read_type type,
                                                size_t first,
                                                size_t readLimit) {
    const std::string imageExtensions[] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif"};
    bool isImage = false;

    for (const auto& ext : imageExtensions) {
        if (source.size() >= ext.size() &&
            source.compare(source.size() - ext.size(), ext.size(), ext) == 0) {
            isImage = true;
            break;
        }
    }

    if (isImage) {
        return std::make_unique<ImageCapture>(source, 30.0);
    } else {
        return std::make_unique<VideoCapture>(source, loop, readLimit, first);
    }
}
