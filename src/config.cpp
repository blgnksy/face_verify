/**
 * face_verify/config — configuration file loading and saving.
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
#include "face_verify/config.hpp"
#include <fstream>
#include <stdexcept>

namespace fv {

namespace {

std::string trim(const std::string &s) {
  const char *ws = " \t\r\n";
  size_t start = s.find_first_not_of(ws);
  if (start == std::string::npos)
    return {};
  size_t end = s.find_last_not_of(ws);
  return s.substr(start, end - start + 1);
}

bool parse_bool(const std::string &v) {
  return v == "true" || v == "1" || v == "yes";
}

} // namespace

void load_config(Config &cfg, const fs::path &path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("Cannot open config file: " + path.string());

  std::string models_dir;
  std::string line;

  while (std::getline(f, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#')
      continue;

    auto eq = line.find('=');
    if (eq == std::string::npos)
      continue;

    std::string key = trim(line.substr(0, eq));
    std::string val = trim(line.substr(eq + 1));
    if (val.empty())
      continue;

    if (key == "rgb_device")
      cfg.rgb_device = val;
    else if (key == "ir_device")
      cfg.ir_device = val;
    else if (key == "rgb_width")
      cfg.rgb_width = std::stoi(val);
    else if (key == "rgb_height")
      cfg.rgb_height = std::stoi(val);
    else if (key == "ir_width")
      cfg.ir_width = std::stoi(val);
    else if (key == "ir_height")
      cfg.ir_height = std::stoi(val);
    else if (key == "data_dir")
      cfg.data_dir = val;
    else if (key == "models_dir")
      models_dir = val;
    else if (key == "threshold")
      cfg.ensemble_threshold = std::stof(val);
    else if (key == "dl_threshold")
      cfg.dl_threshold = std::stof(val);
    else if (key == "dl_weight")
      cfg.dl_weight = std::stof(val);
    else if (key == "classical_weight")
      cfg.classical_weight = std::stof(val);
    else if (key == "detect_conf_threshold")
      cfg.detect_conf_threshold = std::stof(val);
    else if (key == "detect_nms_threshold")
      cfg.detect_nms_threshold = std::stof(val);
    else if (key == "num_frames")
      cfg.num_frames = std::stoi(val);
    else if (key == "frame_interval_ms")
      cfg.frame_interval_ms = std::stoi(val);
    else if (key == "liveness_min_shift")
      cfg.liveness_min_shift = std::stof(val);
    else if (key == "liveness_max_shift")
      cfg.liveness_max_shift = std::stof(val);
    else if (key == "debug")
      cfg.debug = parse_bool(val);
    else if (key == "debug_save_frames")
      cfg.debug_save_frames = parse_bool(val);
    // unknown keys are silently ignored
  }

  if (!models_dir.empty()) {
    cfg.detector_model =
        fs::path(models_dir) / "face_detection_yunet_2023mar.onnx";
    cfg.recognizer_model = fs::path(models_dir) / "w600k_mbf.onnx";
  }
}

void save_config(const Config &cfg, const fs::path &path) {
  std::ofstream f(path);
  if (!f)
    throw std::runtime_error("Cannot write config file: " + path.string());

  std::string models_dir = cfg.detector_model.parent_path().string();

  f << "# face_verify configuration\n"
       "# See README for documentation on all options.\n"
       "\n"
       "# ── Cameras ──────────────────────────────────────────────\n"
       "# RGB (color) camera device. Use `v4l2-ctl --list-devices` to "
       "identify.\n"
    << "rgb_device = " << cfg.rgb_device
    << "\n"
       "# IR (infrared) camera device. Required for liveness detection.\n"
    << "ir_device = " << cfg.ir_device
    << "\n"
       "\n"
       "# Camera capture resolution\n"
    << "rgb_width  = " << cfg.rgb_width << "\n"
    << "rgb_height = " << cfg.rgb_height << "\n"
    << "ir_width   = " << cfg.ir_width << "\n"
    << "ir_height  = " << cfg.ir_height
    << "\n"
       "\n"
       "# ── Paths ────────────────────────────────────────────────\n"
    << "data_dir   = " << cfg.data_dir.string() << "\n"
    << "models_dir = " << models_dir
    << "\n"
       "\n"
       "# ── Thresholds ───────────────────────────────────────────\n"
       "# ensemble threshold: main accept/reject gate (0.0–1.0)\n"
       "# Lower = accept more easily. Raise if impersonation is a concern.\n"
    << "threshold    = " << cfg.ensemble_threshold
    << "\n"
       "# Deep learning minimum (must pass independently of ensemble)\n"
    << "dl_threshold = " << cfg.dl_threshold
    << "\n"
       "\n"
       "# ── Ensemble weights ─────────────────────────────────────\n"
       "# dl_weight + classical_weight should sum to 1.0\n"
    << "dl_weight        = " << cfg.dl_weight << "\n"
    << "classical_weight = " << cfg.classical_weight
    << "\n"
       "\n"
       "# ── Detector tuning ──────────────────────────────────────\n"
       "# Lower conf to detect more faces; lower nms to keep only the best "
       "box\n"
    << "detect_conf_threshold = " << cfg.detect_conf_threshold << "\n"
    << "detect_nms_threshold  = " << cfg.detect_nms_threshold
    << "\n"
       "\n"
       "# ── Capture ──────────────────────────────────────────────\n"
       "# Number of frame pairs to capture per verify/enroll attempt\n"
    << "num_frames        = " << cfg.num_frames
    << "\n"
       "# Milliseconds between frame captures\n"
    << "frame_interval_ms = " << cfg.frame_interval_ms
    << "\n"
       "\n"
       "# ── Liveness ─────────────────────────────────────────────\n"
       "# Min face bbox shift (px) across frames required to pass liveness "
       "check\n"
       "# Lower if rejected while moving naturally; raise for stricter "
       "anti-spoof\n"
    << "liveness_min_shift = " << cfg.liveness_min_shift << "\n"
    << "liveness_max_shift = " << cfg.liveness_max_shift
    << "\n"
       "\n"
       "# ── Debug ────────────────────────────────────────────────\n"
       "# Print per-component scores to stderr (CLI) / syslog (PAM module)\n"
    << "debug             = " << (cfg.debug ? "true" : "false")
    << "\n"
       "# Save raw captured frame pairs to disk during enrollment (for "
       "diagnosis)\n"
    << "debug_save_frames = " << (cfg.debug_save_frames ? "true" : "false")
    << "\n";
}

} // namespace fv
