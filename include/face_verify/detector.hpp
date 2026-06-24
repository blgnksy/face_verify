/**
 * face_verify/detector — YuNet face detection and landmark-based alignment.
 * Copyright (C) 2026  Bilgin Aksoy
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
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <optional>

namespace fv {

class FaceDetector {
public:
  explicit FaceDetector(const Config &cfg);

  // Detect faces in a frame, return sorted by confidence (best first)
  std::vector<FaceDetection> detect(const cv::Mat &frame);

  // Convenience: get the best face or nullopt
  std::optional<FaceDetection> detect_best(const cv::Mat &frame);

  // Align face to 112x112 using 5-point landmarks (ArcFace alignment)
  static cv::Mat align_face(const cv::Mat &frame, const FaceDetection &det,
                            int target_size = 112);

private:
  cv::Ptr<cv::FaceDetectorYN> detector_;
  Config cfg_;
};

} // namespace fv
