/**
 * face_verify/main — CLI entry point (enroll, verify, list, remove,
 * dump-config). Copyright (C) 2026  Bilgin Aksoy
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
#include "face_verify/classical_recognizer.hpp"
#include "face_verify/config.hpp"
#include "face_verify/deep_recognizer.hpp"
#include "face_verify/detector.hpp"
#include "face_verify/ensemble.hpp"
#include "face_verify/face_db.hpp"
#include "face_verify/types.hpp"
#include "face_verify/version.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

using namespace fv;

// ── Forward declarations ──
static int cmd_enroll(Config &cfg, const std::string &label);
static int cmd_verify(Config &cfg);
static int cmd_list(Config &cfg);

static void print_version() {
  std::cout << "face_verify " << FV_VERSION_STRING << "\n";
}

static void print_usage(const char *prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " enroll <label>   -- Enroll a new face\n"
      << "  " << prog
      << " verify           -- Verify against enrolled faces (exit 0=match, "
         "1=fail)\n"
      << "  " << prog << " list             -- List enrolled faces\n"
      << "  " << prog << " remove <label>   -- Remove an enrollment\n"
      << "  " << prog << " dump-config      -- Print current config to stdout\n"
      << "\nOptions:\n"
      << "  --version              Print version and exit\n"
      << "  --config <path>        Config file (default: "
         "/etc/face_verify/face_verify.conf)\n"
      << "  --rgb-dev <path>       RGB camera device (default: /dev/video0)\n"
      << "  --ir-dev <path>        IR camera device  (default: /dev/video2)\n"
      << "  --data-dir <path>      Enrollment data directory (default: data)\n"
      << "  --models-dir <path>    Model directory (default: models)\n"
      << "  --threshold <float>    Ensemble threshold (default: 0.50)\n"
      << "  --debug                Verbose per-component score logging\n";
}

static Config parse_args(int argc, char *argv[]) {
  Config cfg;

  // First pass: find --config to load the file before CLI overrides.
  // Search order: explicit --config arg > ./face_verify.conf >
  // /etc/face_verify/face_verify.conf
  fs::path config_path = fs::exists("face_verify.conf")
                             ? "face_verify.conf"
                             : "/etc/face_verify/face_verify.conf";
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--config" && i + 1 < argc)
      config_path = argv[++i];
  }

  if (fs::exists(config_path)) {
    try {
      load_config(cfg, config_path);
    } catch (const std::exception &e) {
      std::cerr << "Warning: " << e.what() << "\n";
    }
  }

  // Second pass: CLI flags override config file
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--rgb-dev" && i + 1 < argc)
      cfg.rgb_device = argv[++i];
    else if (arg == "--ir-dev" && i + 1 < argc)
      cfg.ir_device = argv[++i];
    else if (arg == "--data-dir" && i + 1 < argc)
      cfg.data_dir = argv[++i];
    else if (arg == "--models-dir" && i + 1 < argc) {
      std::string dir = argv[++i];
      cfg.detector_model = fs::path(dir) / "face_detection_yunet_2023mar.onnx";
      cfg.recognizer_model = fs::path(dir) / "w600k_mbf.onnx";
    } else if (arg == "--threshold" && i + 1 < argc)
      cfg.ensemble_threshold = std::stof(argv[++i]);
    else if (arg == "--debug")
      cfg.debug = true;
    else if (arg == "--config" && i + 1 < argc)
      ++i; // already consumed in first pass
  }
  return cfg;
}

// ══════════════════════════════════════════════════════════════
//  ENROLL
// ══════════════════════════════════════════════════════════════

static int cmd_enroll(Config &cfg, const std::string &label) {
  std::cerr << "=== Enrolling face: " << label << " ===\n";
  std::cerr << "Look at the camera...\n";

  DualCapture capture(cfg);
  if (!capture.open())
    return 1;

  FaceDetector detector(cfg);
  DeepRecognizer deep(cfg);
  ClassicalRecognizer classical;
  FaceDB db(cfg.data_dir, cfg.debug);

  auto frames = capture.grab_sequence(cfg.num_frames, cfg.frame_interval_ms);
  capture.close();

  if (frames.empty()) {
    std::cerr << "ERROR: No frames captured.\n";
    return 1;
  }

  if (cfg.debug || cfg.debug_save_frames) {
    for (size_t i = 0; i < frames.size(); ++i) {
      std::string rgb_path = "debug_rgb_" + std::to_string(i) + ".jpg";
      std::string ir_path = "debug_ir_" + std::to_string(i) + ".jpg";
      if (!cv::imwrite(rgb_path, frames[i].rgb))
        std::cerr << "[debug] WARNING: failed to write " << rgb_path << "\n";
      if (!cv::imwrite(ir_path, frames[i].ir))
        std::cerr << "[debug] WARNING: failed to write " << ir_path << "\n";
    }
    std::cerr << "[debug] Frames written to debug_rgb_N.jpg / debug_ir_N.jpg\n";
  }

  if (cfg.debug && !frames.empty()) {
    const auto &f0 = frames[0];
    cv::Scalar mean_rgb = cv::mean(f0.rgb);
    cv::Scalar mean_ir = cv::mean(f0.ir);
    std::cerr << "[debug] Frame size — RGB: " << f0.rgb.cols << "x"
              << f0.rgb.rows << " (mean brightness=" << mean_rgb[0] << ")"
              << "  IR: " << f0.ir.cols << "x" << f0.ir.rows
              << " (mean brightness=" << mean_ir[0] << ")\n";
  }

  cv::Mat best_rgb_aligned, best_ir_aligned;
  float best_rgb_conf = 0.f, best_ir_conf = 0.f;

  for (size_t fi = 0; fi < frames.size(); ++fi) {
    auto &fp = frames[fi];
    auto all_rgb = detector.detect(fp.rgb);
    auto all_ir = detector.detect(fp.ir);

    if (cfg.debug) {
      std::cerr << "[debug] Frame " << fi
                << " — RGB detections: " << all_rgb.size();
      if (!all_rgb.empty())
        std::cerr << " (best conf=" << all_rgb[0].confidence << ")";
      std::cerr << "  IR detections: " << all_ir.size();
      if (!all_ir.empty())
        std::cerr << " (best conf=" << all_ir[0].confidence << ")";
      std::cerr << "\n";
    }

    if (!all_rgb.empty() && all_rgb[0].confidence > best_rgb_conf) {
      best_rgb_conf = all_rgb[0].confidence;
      best_rgb_aligned = FaceDetector::align_face(fp.rgb, all_rgb[0], 112);
    }
    if (!all_ir.empty() && all_ir[0].confidence > best_ir_conf) {
      best_ir_conf = all_ir[0].confidence;
      best_ir_aligned = FaceDetector::align_face(fp.ir, all_ir[0], 112);
    }
  }

  if (best_rgb_aligned.empty() || best_ir_aligned.empty()) {
    std::cerr
        << "ERROR: Could not detect face in both cameras.\n"
        << "  Hint: run with --debug to see per-frame detection results,\n"
        << "        or add debug_save_frames=true to your config to inspect "
           "frames.\n";
    return 1;
  }

  FaceRecord record;
  record.label = label;

  record.dl_embedding = deep.get_embedding(best_rgb_aligned);
  record.dl_embedding_ir = deep.get_embedding(best_ir_aligned);

  cv::Mat rgb_gray;
  cv::cvtColor(best_rgb_aligned, rgb_gray, cv::COLOR_BGR2GRAY);
  record.lbp_histogram = classical.compute_lbp_histogram(rgb_gray);
  record.aligned_face_gray = rgb_gray.clone();

  cv::Mat ir_gray;
  if (best_ir_aligned.channels() > 1)
    cv::cvtColor(best_ir_aligned, ir_gray, cv::COLOR_BGR2GRAY);
  else
    ir_gray = best_ir_aligned;
  record.ir_lbp_histogram = classical.compute_lbp_histogram(ir_gray);
  record.ir_aligned_face_gray = ir_gray.clone();

  if (!db.enroll(record)) {
    std::cerr << "ERROR: Failed to save enrollment.\n";
    return 1;
  }

  std::cerr << "=== Enrollment complete: " << label
            << " (embedding dim=" << record.dl_embedding.size() << ") ===\n";
  return 0;
}

// ══════════════════════════════════════════════════════════════
//  VERIFY
// ══════════════════════════════════════════════════════════════

static int cmd_verify(Config &cfg) {
  FaceDB db(cfg.data_dir, cfg.debug);
  db.load();

  if (db.empty()) {
    std::cerr << "ERROR: No enrolled faces. Run 'enroll' first.\n";
    return 1;
  }

  DualCapture capture(cfg);
  if (!capture.open())
    return 1;

  FaceDetector detector(cfg);
  DeepRecognizer deep(cfg);
  ClassicalRecognizer classical;
  Ensemble ensemble(cfg);

  auto frames = capture.grab_sequence(cfg.num_frames, cfg.frame_interval_ms);
  capture.close();

  if (frames.empty()) {
    std::cerr << "ERROR: No frames captured.\n";
    return 1;
  }

  if (cfg.debug || cfg.debug_save_frames) {
    for (size_t i = 0; i < frames.size(); ++i) {
      std::string rgb_path = "verify_rgb_" + std::to_string(i) + ".jpg";
      std::string ir_path = "verify_ir_" + std::to_string(i) + ".jpg";
      if (!cv::imwrite(rgb_path, frames[i].rgb))
        std::cerr << "[debug] WARNING: failed to write " << rgb_path << "\n";
      if (!cv::imwrite(ir_path, frames[i].ir))
        std::cerr << "[debug] WARNING: failed to write " << ir_path << "\n";
    }
    std::cerr
        << "[debug] Frames written to verify_rgb_N.jpg / verify_ir_N.jpg\n";
  }

  if (cfg.debug && !frames.empty()) {
    const auto &f0 = frames[0];
    cv::Scalar mean_rgb = cv::mean(f0.rgb);
    cv::Scalar mean_ir = cv::mean(f0.ir);
    std::cerr << "[debug] Frame size — RGB: " << f0.rgb.cols << "x"
              << f0.rgb.rows << " (mean brightness=" << mean_rgb[0] << ")"
              << "  IR: " << f0.ir.cols << "x" << f0.ir.rows
              << " (mean brightness=" << mean_ir[0] << ")\n";
  }

  std::vector<FaceDetection> rgb_detections;
  cv::Mat best_rgb_aligned, best_ir_aligned;
  float best_rgb_conf = 0.f, best_ir_conf = 0.f;
  bool got_ir = false;

  for (size_t fi = 0; fi < frames.size(); ++fi) {
    auto &fp = frames[fi];
    auto all_rgb = detector.detect(fp.rgb);
    auto all_ir = detector.detect(fp.ir);

    if (cfg.debug) {
      std::cerr << "[debug] Frame " << fi
                << " — RGB detections: " << all_rgb.size();
      if (!all_rgb.empty())
        std::cerr << " (best conf=" << all_rgb[0].confidence << ")";
      std::cerr << "  IR detections: " << all_ir.size();
      if (!all_ir.empty())
        std::cerr << " (best conf=" << all_ir[0].confidence << ")";
      std::cerr << "\n";
    }

    if (!all_rgb.empty()) {
      const auto &det_rgb = all_rgb[0];
      rgb_detections.push_back(det_rgb);
      if (det_rgb.confidence > best_rgb_conf) {
        best_rgb_conf = det_rgb.confidence;
        best_rgb_aligned = FaceDetector::align_face(fp.rgb, det_rgb, 112);
      }
    }
    if (!all_ir.empty()) {
      const auto &det_ir = all_ir[0];
      got_ir = true;
      if (det_ir.confidence > best_ir_conf) {
        best_ir_conf = det_ir.confidence;
        best_ir_aligned = FaceDetector::align_face(fp.ir, det_ir, 112);
      }
    }
  }

  if (best_rgb_aligned.empty()) {
    std::cerr
        << "REJECT: No RGB face detected in any of " << frames.size()
        << " frames.\n"
        << "  Hint: run with --debug to see per-frame detection results,\n"
        << "        or set debug_save_frames=true in config to inspect "
           "captured frames.\n";
    return 1;
  }
  if (!got_ir || best_ir_aligned.empty()) {
    std::cerr << "REJECT: No IR face detected.\n"
              << "  Hint: check that the IR camera is at " << cfg.ir_device
              << " and test with: ffplay " << cfg.ir_device << "\n";
    return 1;
  }
  if (!Ensemble::check_liveness(rgb_detections, cfg.liveness_min_shift,
                                cfg.liveness_max_shift, cfg.debug)) {
    std::cerr << "REJECT: Liveness check failed.\n"
              << "  Hint: move your head slightly, or lower liveness_min_shift "
                 "in config.\n";
    return 1;
  }

  auto live_emb_rgb = deep.get_embedding(best_rgb_aligned);
  auto live_emb_ir = deep.get_embedding(best_ir_aligned);

  cv::Mat live_rgb_gray, live_ir_gray;
  cv::cvtColor(best_rgb_aligned, live_rgb_gray, cv::COLOR_BGR2GRAY);
  if (best_ir_aligned.channels() > 1)
    cv::cvtColor(best_ir_aligned, live_ir_gray, cv::COLOR_BGR2GRAY);
  else
    live_ir_gray = best_ir_aligned;

  VerifyResult best_result;

  for (const auto &record : db.records()) {
    float dl_rgb = deep.verify(live_emb_rgb, record.dl_embedding);
    // Compare IR live embedding against IR enrollment embedding when available;
    // fall back to RGB reference for backward compatibility with old
    // enrollments.
    const auto &ir_ref = record.dl_embedding_ir.empty()
                             ? record.dl_embedding
                             : record.dl_embedding_ir;
    float dl_ir = deep.verify(live_emb_ir, ir_ref);
    float cl_rgb = classical.verify(live_rgb_gray, record.lbp_histogram,
                                    record.aligned_face_gray);
    float cl_ir = classical.verify(live_ir_gray, record.ir_lbp_histogram,
                                   record.ir_aligned_face_gray);

    auto result = ensemble.fuse(dl_rgb, dl_ir, cl_rgb, cl_ir, record.label);
    if (result.ensemble_score > best_result.ensemble_score)
      best_result = result;
  }

  if (best_result.accepted) {
    std::cerr << "ACCEPT: Verified as " << best_result.matched_label
              << " (score=" << best_result.ensemble_score << ")\n";
    return 0;
  } else {
    std::cerr << "REJECT: Best match " << best_result.matched_label
              << " (score=" << best_result.ensemble_score << ")\n";
    return 1;
  }
}

// ══════════════════════════════════════════════════════════════
//  LIST / REMOVE
// ══════════════════════════════════════════════════════════════

static int cmd_list(Config &cfg) {
  FaceDB db(cfg.data_dir, cfg.debug);
  db.load();

  auto labels = db.list_labels();
  if (labels.empty()) {
    std::cout << "No enrolled faces.\n";
  } else {
    std::cout << "Enrolled faces (" << labels.size() << "):\n";
    for (const auto &l : labels)
      std::cout << "  - " << l << "\n";
  }
  return 0;
}

// ══════════════════════════════════════════════════════════════
//  MAIN
// ══════════════════════════════════════════════════════════════

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  if (std::string(argv[1]) == "--version") {
    print_version();
    return 0;
  }

  Config cfg = parse_args(argc, argv);
  std::string command = argv[1];

  try {
    if (command == "enroll") {
      if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " enroll <label>\n";
        return 1;
      }
      return cmd_enroll(cfg, argv[2]);

    } else if (command == "verify") {
      return cmd_verify(cfg);

    } else if (command == "list") {
      return cmd_list(cfg);

    } else if (command == "remove") {
      if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " remove <label>\n";
        return 1;
      }
      FaceDB db(cfg.data_dir, cfg.debug);
      db.load();
      if (db.remove(argv[2])) {
        std::cout << "Removed: " << argv[2] << "\n";
        return 0;
      } else {
        std::cerr << "Not found: " << argv[2] << "\n";
        return 1;
      }

    } else if (command == "dump-config") {
      save_config(cfg, "/dev/stdout");
      return 0;

    } else {
      print_usage(argv[0]);
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "FATAL: " << e.what() << "\n";
    return 1;
  }
}
