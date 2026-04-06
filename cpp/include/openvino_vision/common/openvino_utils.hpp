// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>
#include <openvino_vision/common/slog.hpp>

// Utility function to check if a size/shape is empty
inline bool isSizeEmpty(const ov::Shape& shape) {
    return shape.empty() || std::any_of(shape.begin(), shape.end(), [](size_t dim) { return dim == 0; });
}

// Overload for cv::Size
inline bool isSizeEmpty(const cv::Size& size) {
    return size.width <= 0 || size.height <= 0;
}

inline void logBasicModelInfo(const std::shared_ptr<ov::Model>& model) {
    slog::info << "Model name: " << model->get_friendly_name() << slog::endl;

    // Log inputs
    slog::info << "Model inputs:" << slog::endl;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        const auto& type = input.get_element_type();
        const auto& shape = input.get_shape();

        slog::info << "  " << name << ", " << type << ", " << shape << slog::endl;
    }

    // Log outputs
    slog::info << "Model outputs:" << slog::endl;
    for (const auto& output : model->outputs()) {
        const auto& name = output.get_any_name();
        const auto& type = output.get_element_type();
        const auto& shape = output.get_shape();

        slog::info << "  " << name << ", " << type << ", " << shape << slog::endl;
    }
}

inline void logCompiledModelInfo(
    const ov::CompiledModel& compiled_model,
    const std::string& model_path,
    const std::string& device_name,
    const std::string& model_type) {

    slog::info << "Model " << model_path << " loaded on " << device_name << " device for "
              << model_type << " inference" << slog::endl;

    try {
        slog::info << "  Performance hint: " << compiled_model.get_property(ov::hint::performance_mode) << slog::endl;
    } catch (...) {
        // Skip properties that can't be displayed
    }
}

inline void resize2tensor(const cv::Mat& frame, ov::Tensor& tensor) {
    const ov::Shape& tensor_shape = tensor.get_shape();

    std::stringstream shape_ss;
    shape_ss << "Tensor shape: [";
    for (size_t i = 0; i < tensor_shape.size(); i++) {
        shape_ss << tensor_shape[i];
        if (i < tensor_shape.size() - 1) shape_ss << ",";
    }
    shape_ss << "]";
    slog::debug << shape_ss.str() << slog::endl;

    if (frame.empty()) {
        throw std::runtime_error("Input frame is empty");
    }

    size_t height, width, channels;
    bool is_nchw = false;

    if (tensor_shape.size() == 4) {
        if (tensor_shape[1] == 3) {
            is_nchw = true;
            channels = tensor_shape[1];
            height = tensor_shape[2];
            width = tensor_shape[3];
        } else if (tensor_shape[3] == 3) {
            is_nchw = false;
            height = tensor_shape[1];
            width = tensor_shape[2];
            channels = tensor_shape[3];
        } else {
            slog::warn << "Cannot determine layout from shape: " << shape_ss.str() << slog::endl;
            is_nchw = true;
            channels = tensor_shape[1];
            height = tensor_shape[2];
            width = tensor_shape[3];
        }
    } else {
        throw std::runtime_error("Unsupported tensor shape for resizing: " + shape_ss.str());
    }

    if (width == 0 || height == 0 || channels == 0) {
        throw std::runtime_error("Invalid dimensions: " + std::to_string(width) + "x" +
                               std::to_string(height) + "x" + std::to_string(channels));
    }

    if (channels != frame.channels()) {
        throw std::runtime_error("Channel count mismatch: image has " +
                               std::to_string(frame.channels()) +
                               " but tensor expects " + std::to_string(channels));
    }

    slog::debug << "Resizing image from " << frame.cols << "x" << frame.rows
              << " to " << width << "x" << height << slog::endl;

    cv::Mat resized_image;
    cv::resize(frame, resized_image, cv::Size(static_cast<int>(width), static_cast<int>(height)));

    uint8_t* tensor_data = tensor.data<uint8_t>();

    if (is_nchw) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    tensor_data[c * height * width + h * width + w] =
                        resized_image.at<cv::Vec3b>(h, w)[c];
                }
            }
        }
    } else {
        std::memcpy(tensor_data, resized_image.data, height * width * channels);
    }
}
