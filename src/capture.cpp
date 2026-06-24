/**
 * face_verify/capture — dual-camera video capture and synchronized frame
 * extraction. Copyright (C) 2026  Bilgin Aksoy
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
#include "face_verify/capture.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <thread>

namespace fv {

DualCapture::DualCapture(const Config &cfg) : cfg_(cfg) {}

DualCapture::~DualCapture() { close(); }

bool DualCapture::open() {
  // Open RGB camera
  cap_rgb_.open(cfg_.rgb_device, cv::CAP_V4L2);
  if (!cap_rgb_.isOpened()) {
    std::cerr << "[capture] Failed to open RGB camera: " << cfg_.rgb_device
              << "\n";
    return false;
  }
  cap_rgb_.set(cv::CAP_PROP_FOURCC,
               cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  cap_rgb_.set(cv::CAP_PROP_FRAME_WIDTH, cfg_.rgb_width);
  cap_rgb_.set(cv::CAP_PROP_FRAME_HEIGHT, cfg_.rgb_height);

  // Open IR camera
  cap_ir_.open(cfg_.ir_device, cv::CAP_V4L2);
  if (!cap_ir_.isOpened()) {
    std::cerr << "[capture] Failed to open IR camera: " << cfg_.ir_device
              << "\n";
    cap_rgb_.release();
    return false;
  }
  // IR camera outputs GREY at 640x360
  cap_ir_.set(cv::CAP_PROP_FRAME_WIDTH, cfg_.ir_width);
  cap_ir_.set(cv::CAP_PROP_FRAME_HEIGHT, cfg_.ir_height);

  // Warm up: discard frames while auto-exposure/white-balance settle
  cv::Mat tmp;
  for (int i = 0; i < 40; ++i) {
    cap_rgb_.read(tmp);
    cap_ir_.read(tmp);
  }

  opened_ = true;
  std::cerr << "[capture] Cameras opened — RGB: " << cfg_.rgb_device
            << " IR: " << cfg_.ir_device << "\n";
  return true;
}

void DualCapture::close() {
  if (opened_) {
    cap_rgb_.release();
    cap_ir_.release();
    opened_ = false;
  }
}

bool DualCapture::is_open() const {
  return opened_ && cap_rgb_.isOpened() && cap_ir_.isOpened();
}

std::optional<FramePair> DualCapture::grab() {
  if (!is_open())
    return std::nullopt;

  FramePair pair;
  cv::Mat rgb_frame, ir_frame;
  bool ok_rgb = false, ok_ir = false;

  // Grab both as close together as possible
  // Use grab/retrieve pattern for tighter synchronization
  cap_rgb_.grab();
  cap_ir_.grab();

  ok_rgb = cap_rgb_.retrieve(rgb_frame);
  ok_ir = cap_ir_.retrieve(ir_frame);

  if (!ok_rgb || !ok_ir || rgb_frame.empty() || ir_frame.empty()) {
    std::cerr << "[capture] Frame grab failed — RGB: " << ok_rgb
              << " IR: " << ok_ir << "\n";
    return std::nullopt;
  }

  pair.rgb = rgb_frame;
  pair.timestamp_ms = now_ms();

  // Ensure IR is single-channel grayscale
  if (ir_frame.channels() == 1) {
    pair.ir = ir_frame;
  } else {
    cv::cvtColor(ir_frame, pair.ir, cv::COLOR_BGR2GRAY);
  }

  return pair;
}

std::vector<FramePair> DualCapture::grab_sequence(int n, int interval_ms) {
  std::vector<FramePair> frames;
  frames.reserve(n);

  for (int i = 0; i < n; ++i) {
    auto fp = grab();
    if (fp) {
      frames.push_back(std::move(*fp));
    }
    if (i < n - 1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
  }
  return frames;
}

} // namespace fv
