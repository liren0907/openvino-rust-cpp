// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

template<typename T> inline T clamp(T value, T low, T high) {
    return std::min(std::max(value, low), high);
}

template <typename T, size_t N>
size_t arraySize(const T (&)[N]) {
    return N;
}

inline cv::Size roiSize(const cv::Rect& roi) {
    return cv::Size(roi.width, roi.height);
}

inline cv::Rect fitIntoSize(cv::Rect rectangle, cv::Size size) {
    cv::Rect result = rectangle;
    if (result.x < 0) {
        result.width += result.x;
        result.x = 0;
    }
    if (result.y < 0) {
        result.height += result.y;
        result.y = 0;
    }

    if (result.x + result.width > size.width) {
        result.width = size.width - result.x;
    }
    if (result.y + result.height > size.height) {
        result.height = size.height - result.y;
    }
    return result;
}

inline void AlignFaces(std::vector<cv::Mat>* faces, const std::vector<cv::Mat>* landmarks) {
    for (size_t i = 0; i < faces->size(); i++) {
        auto& currFace = faces->at(i);
        const auto& currLandmark = landmarks->at(i);

        // LEFT_EYE_COORDINATE
        float l_x = currLandmark.at<float>(0, 0);
        float l_y = currLandmark.at<float>(0, 1);

        // RIGHT_EYE_COORDINATE
        float r_x = currLandmark.at<float>(1, 0);
        float r_y = currLandmark.at<float>(1, 1);

        float dy = r_y - l_y;
        float dx = r_x - l_x;
        float angle = atan2(dy, dx) * 180.0f / CV_PI;

        cv::Point2f center(currFace.cols * 0.5f, currFace.rows * 0.5f);
        cv::Mat rotMatrix = cv::getRotationMatrix2D(center, angle, 1.0f);
        cv::Mat rotFace;
        cv::warpAffine(currFace, rotFace, rotMatrix, currFace.size());
        currFace = rotFace;
    }
}
