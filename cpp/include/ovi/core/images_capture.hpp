// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

enum class read_type {
    safe, // handles empty frames
    efficient  // stops pipeline in case of empty data
};

class ImagesCapture {
public:
    virtual double fps() const = 0;
    virtual cv::Mat read() = 0;
    virtual ~ImagesCapture() = default;
};

std::unique_ptr<ImagesCapture> openImagesCapture(const std::string& source,
                                                 bool loop,
                                                 read_type type,
                                                 size_t first = 0,
                                                 size_t readLimit = ~0);

class LazyVideoWriter {
public:
    LazyVideoWriter(const std::string& filename, double fps, size_t frameLimit = ~0)
        : filename(filename), fps(fps), frameLimit(frameLimit), frameCount(0) {}

    void write(const cv::Mat& frame) {
        if (filename.empty() || (frameLimit != ~0 && frameCount >= frameLimit)) {
            return;
        }

        if (!videoWriter.isOpened()) {
            videoWriter = cv::VideoWriter(filename, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                         fps, frame.size());
            if (!videoWriter.isOpened()) {
                throw std::runtime_error("Cannot open video writer. Check filename and access rights.");
            }
        }

        videoWriter.write(frame);
        frameCount++;
    }

private:
    std::string filename;
    double fps;
    size_t frameLimit;
    size_t frameCount;
    cv::VideoWriter videoWriter;
};
