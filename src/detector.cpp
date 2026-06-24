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
#include "face_verify/detector.hpp"
#include <algorithm>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace fv {

FaceDetector::FaceDetector(const Config &cfg) : cfg_(cfg) {
  detector_ = cv::FaceDetectorYN::create(
      cfg.detector_model.string(),
      "",                 // config (unused for ONNX)
      cv::Size(320, 320), // input size (will be updated per frame)
      cfg.detect_conf_threshold, cfg.detect_nms_threshold,
      5000, // top_k
      cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU);

  if (!detector_) {
    throw std::runtime_error("[detector] Failed to load YuNet model: " +
                             cfg.detector_model.string());
  }
  std::cerr << "[detector] YuNet loaded from " << cfg.detector_model << "\n";
}

std::vector<FaceDetection> FaceDetector::detect(const cv::Mat &frame) {
  std::vector<FaceDetection> results;

  cv::Mat input;
  if (frame.channels() == 1) {
    // IR frame: apply CLAHE to normalize near-IR contrast before detection.
    // YuNet is trained on visible-light images; raw IR has hot spots and
    // uneven reflectance that suppress confidence scores.
    cv::Mat enhanced;
    auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(frame, enhanced);
    cv::cvtColor(enhanced, input, cv::COLOR_GRAY2BGR);
  } else {
    input = frame;
  }

  // Update detector input size to match frame
  detector_->setInputSize(input.size());

  cv::Mat faces_mat;
  detector_->detect(input, faces_mat);

  if (faces_mat.empty())
    return results;

  // faces_mat: each row is [x, y, w, h, x_re, y_re, x_le, y_le,
  //                         x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
  for (int i = 0; i < faces_mat.rows; ++i) {
    const float *row = faces_mat.ptr<float>(i);

    FaceDetection det;
    det.bbox = cv::Rect2f(row[0], row[1], row[2], row[3]);
    det.confidence = row[14];

    // 5 landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
    det.landmarks[0] = {row[4], row[5]};   // right eye
    det.landmarks[1] = {row[6], row[7]};   // left eye
    det.landmarks[2] = {row[8], row[9]};   // nose tip
    det.landmarks[3] = {row[10], row[11]}; // right mouth corner
    det.landmarks[4] = {row[12], row[13]}; // left mouth corner

    results.push_back(det);
  }

  // Sort by confidence descending
  std::sort(results.begin(), results.end(), [](const auto &a, const auto &b) {
    return a.confidence > b.confidence;
  });

  return results;
}

std::optional<FaceDetection> FaceDetector::detect_best(const cv::Mat &frame) {
  auto faces = detect(frame);
  if (faces.empty())
    return std::nullopt;
  return faces[0];
}

cv::Mat FaceDetector::align_face(const cv::Mat &frame, const FaceDetection &det,
                                 int target_size) {
  // ArcFace standard alignment target landmarks for 112x112
  // These are the canonical positions for a frontally-aligned face
  static const cv::Point2f dst_pts[5] = {
      {38.2946f, 51.6963f}, // left eye
      {73.5318f, 51.5014f}, // right eye
      {56.0252f, 71.7366f}, // nose
      {41.5493f, 92.3655f}, // left mouth
      {70.7299f, 92.2041f}  // right mouth
  };

  // Scale target landmarks if target_size != 112
  float scale = static_cast<float>(target_size) / 112.0f;
  cv::Point2f scaled_dst[5];
  for (int i = 0; i < 5; ++i) {
    scaled_dst[i] = dst_pts[i] * scale;
  }

  // Source landmarks from detection
  // YuNet order: right_eye(0), left_eye(1), nose(2), right_mouth(3),
  // left_mouth(4) ArcFace order: left_eye, right_eye, nose, left_mouth,
  // right_mouth
  cv::Point2f src_pts[5] = {
      det.landmarks[1], // left eye
      det.landmarks[0], // right eye
      det.landmarks[2], // nose
      det.landmarks[4], // left mouth
      det.landmarks[3]  // right mouth
  };

  // Estimate similarity transform (requires exactly 5 point pairs → use partial
  // affine)
  cv::Mat src_mat(5, 1, CV_32FC2, src_pts);
  cv::Mat dst_mat(5, 1, CV_32FC2, scaled_dst);
  cv::Mat M = cv::estimateAffinePartial2D(src_mat, dst_mat);

  if (M.empty()) {
    // Fallback: just crop and resize the bounding box
    cv::Rect roi = det.bbox & cv::Rect2f(0, 0, frame.cols, frame.rows);
    cv::Mat cropped;
    frame(roi).copyTo(cropped);
    cv::resize(cropped, cropped, cv::Size(target_size, target_size));
    return cropped;
  }

  cv::Mat aligned;
  cv::warpAffine(frame, aligned, M, cv::Size(target_size, target_size));
  return aligned;
}

} // namespace fv
