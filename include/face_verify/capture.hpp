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
#pragma once

#include "face_verify/types.hpp"
#include <opencv2/videoio.hpp>
#include <optional>

namespace fv {

class DualCapture {
public:
  explicit DualCapture(const Config &cfg);
  ~DualCapture();

  // Non-copyable
  DualCapture(const DualCapture &) = delete;
  DualCapture &operator=(const DualCapture &) = delete;

  bool open();
  void close();
  bool is_open() const;

  // Grab a synchronized frame pair (blocking)
  std::optional<FramePair> grab();

  // Grab N frame pairs with interval_ms between them
  std::vector<FramePair> grab_sequence(int n, int interval_ms);

private:
  Config cfg_;
  cv::VideoCapture cap_rgb_;
  cv::VideoCapture cap_ir_;
  bool opened_{false};
};

} // namespace fv
