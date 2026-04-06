// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
#include <opencv2/core.hpp>

/**
 * @brief This class implements the Kuhn-Munkres algorithm for solving the
 * assignment problem.
 */
class KuhnMunkres {
public:
    KuhnMunkres() = default;

    /**
     * @brief Constructor that accepts a flag indicating whether to use greedy matching.
     * @param use_greedy_matching If true, uses greedy matching instead of full algorithm.
     */
    explicit KuhnMunkres(bool use_greedy_matching) : use_greedy(use_greedy_matching) {}

    /**
     * @brief Solves the assignment problem for the specified cost matrix.
     * @param cost Cost matrix (workers in rows, jobs in columns).
     * @return Pairs: (row, column).
     */
    template <typename T>
    std::vector<std::pair<size_t, size_t>> Solve(const std::vector<std::vector<T>>& cost) {
        if (use_greedy) {
            return SolveGreedy(cost);
        }

        auto pair_cost = cost;
        for (auto& row : pair_cost) {
            auto min = *std::min_element(row.begin(), row.end());
            std::transform(row.begin(), row.end(), row.begin(),
                [min](T val) { return val - min; });
        }

        // Transpose cost
        std::vector<std::vector<T>> transposed(pair_cost[0].size(), std::vector<T>(pair_cost.size()));
        for (size_t i = 0; i < pair_cost.size(); i++) {
            for (size_t j = 0; j < pair_cost[0].size(); j++) {
                transposed[j][i] = pair_cost[i][j];
            }
        }

        for (auto& column : transposed) {
            auto min = *std::min_element(column.begin(), column.end());
            std::transform(column.begin(), column.end(), column.begin(),
                [min](T val) { return val - min; });
        }

        // Transpose back
        for (size_t i = 0; i < pair_cost.size(); i++) {
            for (size_t j = 0; j < pair_cost[0].size(); j++) {
                pair_cost[i][j] = transposed[j][i];
            }
        }

        std::vector<std::pair<size_t, size_t>> indices;
        std::vector<int> row_cover(pair_cost.size(), 0);
        std::vector<int> col_cover(pair_cost[0].size(), 0);

        for (size_t i = 0; i < pair_cost.size(); i++) {
            for (size_t j = 0; j < pair_cost[0].size(); j++) {
                if (fabs(pair_cost[i][j]) < 1e-8 && row_cover[i] == 0 && col_cover[j] == 0) {
                    indices.push_back(std::make_pair(i, j));
                    row_cover[i] = 1;
                    col_cover[j] = 1;
                }
            }
        }

        return indices;
    }

    /**
     * @brief Solves the assignment problem for the specified cv::Mat cost matrix.
     * @param cost Cost matrix (workers in rows, jobs in columns) as cv::Mat.
     * @return Pairs: (row, column).
     */
    std::vector<std::pair<size_t, size_t>> Solve(const cv::Mat& cost) {
        // Convert cv::Mat to std::vector<std::vector<float>>
        std::vector<std::vector<float>> cost_vec(cost.rows, std::vector<float>(cost.cols));
        for (int i = 0; i < cost.rows; i++) {
            for (int j = 0; j < cost.cols; j++) {
                cost_vec[i][j] = cost.at<float>(i, j);
            }
        }

        return Solve(cost_vec);
    }

    /**
     * @brief A cost function for association of two objects.
     * @param cost_matrix The cost matrix of matching objects.
     * @param similarity_threshold The similarity threshold for matching.
     * @param use_min_cost Whether to use the min cost heuristic.
     * @return The best association set.
     */
    template <typename T>
    std::vector<std::pair<size_t, size_t>> Solve(const std::vector<std::vector<T>>& cost_matrix,
                                                 float similarity_threshold,
                                                 bool use_min_cost = false) {
        const float LOG_TRESH = -std::log(similarity_threshold);

        std::vector<std::vector<T>> cost_matrix_copy = cost_matrix;
        for (auto& row : cost_matrix_copy) {
            for (auto& element : row) {
                if (element > LOG_TRESH) {
                    element = LOG_TRESH;
                }
            }
        }

        auto res = Solve(cost_matrix_copy);
        if (use_min_cost) {
            return res;
        }

        std::vector<std::pair<size_t, size_t>> filtered_res;
        for (const auto& pr : res) {
            const auto i = pr.first;
            const auto j = pr.second;
            if (cost_matrix[i][j] < LOG_TRESH) {
                filtered_res.push_back(pr);
            }
        }

        return filtered_res;
    }

private:
    bool use_greedy = false;

    /**
     * @brief Greedy algorithm for assignment.
     * @param cost Cost matrix.
     * @return Pairs: (row, column).
     */
    template <typename T>
    std::vector<std::pair<size_t, size_t>> SolveGreedy(const std::vector<std::vector<T>>& cost) {
        std::vector<std::pair<size_t, size_t>> indices;
        std::vector<bool> row_used(cost.size(), false);
        std::vector<bool> col_used(cost[0].size(), false);

        for (size_t i = 0; i < cost.size(); i++) {
            T min_val = std::numeric_limits<T>::max();
            size_t min_row = 0, min_col = 0;

            for (size_t row = 0; row < cost.size(); row++) {
                if (row_used[row]) continue;

                for (size_t col = 0; col < cost[0].size(); col++) {
                    if (col_used[col]) continue;

                    if (cost[row][col] < min_val) {
                        min_val = cost[row][col];
                        min_row = row;
                        min_col = col;
                    }
                }
            }

            if (min_val != std::numeric_limits<T>::max()) {
                indices.push_back(std::make_pair(min_row, min_col));
                row_used[min_row] = true;
                col_used[min_col] = true;
            }
        }

        return indices;
    }
};
