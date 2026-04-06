// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <string>
#include <sstream>
#include <vector>
#include <opencv2/core.hpp>

class PerformanceMetrics {
public:
    enum MetricTypes {
        FPS,
        MS
    };

    struct Metrics {
        float fps;
        float ms;
        void update(float fps_, float ms_) {
            fps = fps_;
            ms = ms_;
        }
    };

    void update(
        const std::chrono::time_point<std::chrono::steady_clock> &lastTime = {},
        const cv::Mat &frame = cv::Mat(),
        const cv::Point &position = cv::Point(),
        int font = cv::FONT_HERSHEY_COMPLEX,
        double fontScale = 0.65,
        cv::Scalar color = cv::Scalar(200, 10, 10),
        int thickness = 2,
        MetricTypes metricType = MetricTypes::FPS) {

        auto now = std::chrono::steady_clock::now();

        if (lastTime != std::chrono::time_point<std::chrono::steady_clock>()) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime).count();

            if (duration > 0) {
                metrics.update(1000.f / static_cast<float>(duration),
                              static_cast<float>(duration));
            }
        }

        if (!frame.empty() && position != cv::Point()) {
            std::ostringstream out;
            std::string text;

            if (metricType == MetricTypes::FPS) {
                out << std::fixed << std::setprecision(1) << metrics.fps << " FPS";
                text = out.str();
            } else {
                out << std::fixed << std::setprecision(1) << metrics.ms << " ms";
                text = out.str();
            }

            // Create a mutable copy of the frame to draw on
            cv::Mat mutable_frame = frame.clone();
            cv::putText(mutable_frame, text, position, font, fontScale, color, thickness);
        }

        lastTimePoint = now;
    }

    const Metrics& getTotal() const {
        return metrics;
    }

    void logTotal() const {
        std::cout << "\tFPS: " << std::fixed << std::setprecision(1) << metrics.fps << std::endl;
        std::cout << "\tLatency: " << std::fixed << std::setprecision(1) << metrics.ms << " ms" << std::endl;
    }

private:
    Metrics metrics{0.0f, 0.0f};
    std::chrono::time_point<std::chrono::steady_clock> lastTimePoint = std::chrono::steady_clock::now();
};
