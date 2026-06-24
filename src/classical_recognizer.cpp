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
#include "face_verify/classical_recognizer.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace fv {

// ── Uniform LBP lookup table ──
// Maps each 8-bit LBP pattern to a bin index.
// 58 uniform patterns (at most 2 0→1 or 1→0 transitions) → bins 0..57
// All non-uniform patterns → bin 58
const std::array<int, 256> &ClassicalRecognizer::uniform_lbp_table() {
  static std::array<int, 256> table;
  static bool initialized = false;

  if (!initialized) {
    int bin = 0;
    for (int i = 0; i < 256; ++i) {
      // Count transitions (bitwise circular)
      uint8_t val = static_cast<uint8_t>(i);
      uint8_t shifted = (val >> 1) | ((val & 1) << 7);
      uint8_t transitions = val ^ shifted;
      int num_transitions = __builtin_popcount(transitions);

      if (num_transitions <= 2) {
        table[i] = bin++;
      } else {
        table[i] = -1; // mark as non-uniform for now
      }
    }
    // Assign all non-uniform to the last bin
    int non_uniform_bin = bin;
    for (int i = 0; i < 256; ++i) {
      if (table[i] == -1)
        table[i] = non_uniform_bin;
    }
    initialized = true;
  }
  return table;
}

cv::Mat ClassicalRecognizer::compute_lbp_image(const cv::Mat &gray) {
  CV_Assert(gray.type() == CV_8UC1);

  cv::Mat lbp(gray.rows - 2, gray.cols - 2, CV_8UC1);

  for (int y = 1; y < gray.rows - 1; ++y) {
    for (int x = 1; x < gray.cols - 1; ++x) {
      uint8_t center = gray.at<uint8_t>(y, x);
      uint8_t code = 0;

      // 8-neighbor LBP, clockwise from top-left
      code |= (gray.at<uint8_t>(y - 1, x - 1) >= center) << 7;
      code |= (gray.at<uint8_t>(y - 1, x) >= center) << 6;
      code |= (gray.at<uint8_t>(y - 1, x + 1) >= center) << 5;
      code |= (gray.at<uint8_t>(y, x + 1) >= center) << 4;
      code |= (gray.at<uint8_t>(y + 1, x + 1) >= center) << 3;
      code |= (gray.at<uint8_t>(y + 1, x) >= center) << 2;
      code |= (gray.at<uint8_t>(y + 1, x - 1) >= center) << 1;
      code |= (gray.at<uint8_t>(y, x - 1) >= center) << 0;

      lbp.at<uint8_t>(y - 1, x - 1) = code;
    }
  }
  return lbp;
}

cv::Mat ClassicalRecognizer::compute_lbp_histogram(const cv::Mat &aligned_gray,
                                                   int grid_x, int grid_y) {
  cv::Mat gray;
  if (aligned_gray.channels() > 1) {
    cv::cvtColor(aligned_gray, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray =
        aligned_gray.clone(); // clone to avoid CLAHE mutating the caller's Mat
  }

  // Resize to standard size for consistent grid
  cv::resize(gray, gray, cv::Size(112, 112));

  // Apply CLAHE for illumination normalization
  auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
  clahe->apply(gray, gray);

  cv::Mat lbp = compute_lbp_image(gray);
  const auto &table = uniform_lbp_table();

  int num_bins = 59; // 58 uniform + 1 non-uniform
  int cell_w = lbp.cols / grid_x;
  int cell_h = lbp.rows / grid_y;

  // Spatial histogram: grid_x * grid_y cells, each with num_bins bins
  cv::Mat histogram(1, grid_x * grid_y * num_bins, CV_32F, cv::Scalar(0));

  for (int gy = 0; gy < grid_y; ++gy) {
    for (int gx = 0; gx < grid_x; ++gx) {
      int x0 = gx * cell_w;
      int y0 = gy * cell_h;
      int x1 = (gx == grid_x - 1) ? lbp.cols : x0 + cell_w;
      int y1 = (gy == grid_y - 1) ? lbp.rows : y0 + cell_h;

      int offset = (gy * grid_x + gx) * num_bins;

      for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
          int bin = table[lbp.at<uint8_t>(y, x)];
          histogram.at<float>(0, offset + bin) += 1.0f;
        }
      }

      // Normalize this cell's histogram to sum to 1
      float sum = 0.f;
      for (int b = 0; b < num_bins; ++b) {
        sum += histogram.at<float>(0, offset + b);
      }
      if (sum > 0.f) {
        for (int b = 0; b < num_bins; ++b) {
          histogram.at<float>(0, offset + b) /= sum;
        }
      }
    }
  }

  return histogram;
}

float ClassicalRecognizer::compare_lbp(const cv::Mat &hist_a,
                                       const cv::Mat &hist_b) {
  CV_Assert(hist_a.cols == hist_b.cols);

  // Chi-square distance
  double chi_sq = 0.0;
  for (int i = 0; i < hist_a.cols; ++i) {
    float a = hist_a.at<float>(0, i);
    float b = hist_b.at<float>(0, i);
    if (a + b > 1e-10f) {
      chi_sq += static_cast<double>((a - b) * (a - b)) / (a + b);
    }
  }

  // Convert distance to similarity [0, 1]
  // chi_sq ranges from 0 (identical) to ~2 (very different)
  // Use exponential decay: sim = exp(-chi_sq * k)
  float similarity = static_cast<float>(std::exp(-chi_sq * 0.5));
  return similarity;
}

float ClassicalRecognizer::compare_template(const cv::Mat &face_a,
                                            const cv::Mat &face_b) {
  cv::Mat a, b;

  // Ensure grayscale; clone to avoid CLAHE mutating the caller's Mat
  if (face_a.channels() > 1)
    cv::cvtColor(face_a, a, cv::COLOR_BGR2GRAY);
  else
    a = face_a.clone();
  if (face_b.channels() > 1)
    cv::cvtColor(face_b, b, cv::COLOR_BGR2GRAY);
  else
    b = face_b.clone();

  // Resize to same size
  cv::resize(a, a, cv::Size(112, 112));
  cv::resize(b, b, cv::Size(112, 112));

  // CLAHE normalization
  auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
  clahe->apply(a, a);
  clahe->apply(b, b);

  // Normalized cross-correlation
  a.convertTo(a, CV_32F);
  b.convertTo(b, CV_32F);

  // Zero-mean
  a -= cv::mean(a)[0];
  b -= cv::mean(b)[0];

  double norm_a = cv::norm(a);
  double norm_b = cv::norm(b);

  if (norm_a < 1e-6 || norm_b < 1e-6)
    return 0.f;

  double ncc = a.dot(b) / (norm_a * norm_b);

  // NCC ranges [-1, 1], map to [0, 1]
  return static_cast<float>((ncc + 1.0) / 2.0);
}

float ClassicalRecognizer::verify(const cv::Mat &live_gray,
                                  const cv::Mat &ref_lbp_hist,
                                  const cv::Mat &ref_aligned_gray,
                                  float lbp_weight, float template_weight) {
  cv::Mat live_hist = compute_lbp_histogram(live_gray);
  float lbp_score = compare_lbp(live_hist, ref_lbp_hist);
  float tmpl_score = compare_template(live_gray, ref_aligned_gray);

  float combined = lbp_weight * lbp_score + template_weight * tmpl_score;
  return std::clamp(combined, 0.0f, 1.0f);
}

} // namespace fv
