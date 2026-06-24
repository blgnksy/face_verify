/**
 * face_verify/classical_recognizer — LBP histogram and template-matching
 * classical face recognizer. Copyright (C) 2026  Bilgin Aksoy
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once

#include "face_verify/types.hpp"

namespace fv {

class ClassicalRecognizer {
public:
  ClassicalRecognizer() = default;

  // ── Feature extraction ──

  // Compute spatial LBP histogram for an aligned grayscale face
  // Divides face into grid_x * grid_y cells, computes 59-bin uniform LBP per
  // cell
  cv::Mat compute_lbp_histogram(const cv::Mat &aligned_gray, int grid_x = 8,
                                int grid_y = 8);

  // ── Comparison ──

  // Compare two LBP spatial histograms (chi-square distance → similarity [0,1])
  float compare_lbp(const cv::Mat &hist_a, const cv::Mat &hist_b);

  // Normalized cross-correlation between two aligned face images
  float compare_template(const cv::Mat &face_a, const cv::Mat &face_b);

  // Combined classical score from LBP + template matching
  // Returns score in [0, 1]
  float verify(const cv::Mat &live_gray, const cv::Mat &ref_lbp_hist,
               const cv::Mat &ref_aligned_gray, float lbp_weight = 0.6f,
               float template_weight = 0.4f);

private:
  // Compute LBP image (uniform patterns, 59 bins: 58 uniform + 1 non-uniform)
  cv::Mat compute_lbp_image(const cv::Mat &gray);

  // Uniform LBP lookup table
  static const std::array<int, 256> &uniform_lbp_table();
};

} // namespace fv
