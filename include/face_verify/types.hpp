/**
 * face_verify/types — shared data types and configuration struct.
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

#include <chrono>
#include <filesystem>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace fv {

namespace fs = std::filesystem;

// ── Bounding box + 5-point landmarks ──
struct FaceDetection {
  cv::Rect2f bbox;
  float confidence{0.f};
  // 5 landmarks: left_eye, right_eye, nose, mouth_left, mouth_right
  std::array<cv::Point2f, 5> landmarks;
};

// ── A single frame pair from both cameras ──
struct FramePair {
  cv::Mat rgb;          // BGR 8UC3
  cv::Mat ir;           // GREY 8UC1
  int64_t timestamp_ms; // capture time (monotonic)
};

// ── Stored face identity ──
struct FaceRecord {
  std::string label;
  std::vector<float> dl_embedding; // ArcFace embedding from RGB face
  std::vector<float>
      dl_embedding_ir;   // ArcFace embedding from IR face (IR-to-IR compare)
  cv::Mat lbp_histogram; // classical LBP spatial histogram (RGB)
  cv::Mat aligned_face_gray;    // 112x112 grayscale for template matching (RGB)
  cv::Mat ir_lbp_histogram;     // LBP from IR modality
  cv::Mat ir_aligned_face_gray; // IR aligned face
};

// ── Verification result ──
struct VerifyResult {
  float dl_score{0.f};        // cosine similarity from deep model
  float classical_score{0.f}; // combined classical score [0,1]
  float ensemble_score{0.f};  // final fused score
  bool accepted{false};
  std::string matched_label;
};

// ── Configuration ──
struct Config {
  // Camera devices
  std::string rgb_device = "/dev/video0";
  std::string ir_device = "/dev/video2";

  // Capture
  int rgb_width = 640;
  int rgb_height = 480;
  int ir_width = 640;
  int ir_height = 360;

  // Model paths
  fs::path detector_model = "models/face_detection_yunet_2023mar.onnx";
  fs::path recognizer_model = "models/w600k_mbf.onnx";

  // Detection
  float detect_conf_threshold = 0.6f;
  float detect_nms_threshold = 0.3f;

  // Verification thresholds
  float dl_threshold = 0.40f;
  float ensemble_threshold = 0.50f;

  // Ensemble weights
  float dl_weight = 0.60f;
  float classical_weight = 0.40f;

  // Data storage
  fs::path data_dir = "data";

  // Timing
  int num_frames = 3;          // frames to capture for verification
  int frame_interval_ms = 200; // between captures

  // Liveness
  float liveness_min_shift =
      0.5f; // min face bbox shift (px) required between frames
  float liveness_max_shift =
      80.0f; // max shift before rejecting (camera shake / different person)

  // Debug
  bool debug = false;             // verbose score logging to stderr / syslog
  bool debug_save_frames = false; // save raw captured frames to disk on enroll
};

// ── Utility: monotonic timestamp ──
inline int64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch())
      .count();
}

} // namespace fv
