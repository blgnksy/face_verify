/**
 * face_verify/ensemble — decision-level score fusion and liveness check.
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

namespace fv {

class Ensemble {
public:
  explicit Ensemble(const Config &cfg);

  // Fuse deep learning and classical scores into a final decision
  VerifyResult fuse(float dl_score_rgb, float dl_score_ir,
                    float classical_score_rgb, float classical_score_ir,
                    const std::string &matched_label);

  // Check micro-movement across multiple detections (liveness hint)
  // Returns true if face bbox shifted between frames (real face)
  static bool check_liveness(const std::vector<FaceDetection> &detections,
                             float min_shift_pixels = 1.0f,
                             float max_shift_pixels = 80.0f,
                             bool debug = false);

private:
  Config cfg_;
};

} // namespace fv
