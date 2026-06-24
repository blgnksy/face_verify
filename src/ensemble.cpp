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
#include "face_verify/ensemble.hpp"
#include <cmath>
#include <iostream>

namespace fv {

Ensemble::Ensemble(const Config &cfg) : cfg_(cfg) {}

VerifyResult Ensemble::fuse(float dl_score_rgb, float dl_score_ir,
                            float classical_score_rgb, float classical_score_ir,
                            const std::string &matched_label) {
  VerifyResult result;
  result.matched_label = matched_label;

  // ── Per-modality fusion (RGB + IR) ──
  // IR gets slightly more weight because it's harder to spoof
  float dl_score = 0.4f * dl_score_rgb + 0.6f * dl_score_ir;
  float classical_score =
      0.4f * classical_score_rgb + 0.6f * classical_score_ir;

  result.dl_score = dl_score;
  result.classical_score = classical_score;

  // ── Cross-method fusion (DL + Classical) ──
  result.ensemble_score =
      cfg_.dl_weight * dl_score + cfg_.classical_weight * classical_score;

  // ── Decision logic ──
  // Strategy: ensemble score must pass, AND deep learning must independently
  // pass a minimum bar (since it's the stronger signal)
  bool dl_ok = dl_score >= cfg_.dl_threshold;
  bool ensemble_ok = result.ensemble_score >= cfg_.ensemble_threshold;

  result.accepted = dl_ok && ensemble_ok;

  if (cfg_.debug) {
    std::cerr << "[ensemble] DL(rgb=" << dl_score_rgb << " ir=" << dl_score_ir
              << " fused=" << dl_score << ") "
              << "Classical(rgb=" << classical_score_rgb
              << " ir=" << classical_score_ir << " fused=" << classical_score
              << ") "
              << "Ensemble=" << result.ensemble_score << " -> "
              << (result.accepted ? "ACCEPT" : "REJECT") << "\n";
  }

  return result;
}

bool Ensemble::check_liveness(const std::vector<FaceDetection> &detections,
                              float min_shift, float max_shift, bool debug) {
  if (detections.size() < 2)
    return false;

  bool any_movement = false;

  for (size_t i = 1; i < detections.size(); ++i) {
    float cx0 = detections[i - 1].bbox.x + detections[i - 1].bbox.width / 2.f;
    float cy0 = detections[i - 1].bbox.y + detections[i - 1].bbox.height / 2.f;
    float cx1 = detections[i].bbox.x + detections[i].bbox.width / 2.f;
    float cy1 = detections[i].bbox.y + detections[i].bbox.height / 2.f;

    float dist =
        std::sqrt((cx1 - cx0) * (cx1 - cx0) + (cy1 - cy0) * (cy1 - cy0));

    if (debug) {
      std::cerr << "[liveness] frame " << i - 1 << "->" << i
                << " shift=" << dist << "px"
                << " (need " << min_shift << ".." << max_shift << ")\n";
    }

    if (dist > max_shift)
      return false;
    if (dist >= min_shift)
      any_movement = true;
  }

  return any_movement;
}

} // namespace fv
