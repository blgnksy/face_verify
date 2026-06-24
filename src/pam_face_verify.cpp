/**
 * face_verify/pam_face_verify — PAM authentication module.
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
#define PAM_SM_AUTH
#include <security/pam_ext.h>
#include <security/pam_modules.h>
#include <syslog.h>

#include "face_verify/capture.hpp"
#include "face_verify/classical_recognizer.hpp"
#include "face_verify/config.hpp"
#include "face_verify/deep_recognizer.hpp"
#include "face_verify/detector.hpp"
#include "face_verify/ensemble.hpp"
#include "face_verify/face_db.hpp"

#include <opencv2/imgproc.hpp>
#include <string>

using namespace fv;

static constexpr const char *DEFAULT_CONFIG =
    "/etc/face_verify/face_verify.conf";

// Build Config from the optional config file and PAM module args.
// Precedence: defaults < config file < module args.
static Config build_config(pam_handle_t *pamh, int argc, const char **argv) {
  Config cfg;
  cfg.data_dir = "/etc/face_verify/data";
  cfg.detector_model =
      "/etc/face_verify/models/face_detection_yunet_2023mar.onnx";
  cfg.recognizer_model = "/etc/face_verify/models/w600k_mbf.onnx";

  // Allow overriding the config file path via module arg config=<path>
  fs::path config_path = DEFAULT_CONFIG;
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("config=", 0) == 0)
      config_path = a.substr(7);
  }

  if (fs::exists(config_path)) {
    try {
      load_config(cfg, config_path);
    } catch (const std::exception &e) {
      pam_syslog(pamh, LOG_WARNING, "face_verify: %s", e.what());
    }
  }

  // Module args take precedence over the config file
  std::string models_dir;
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("rgb_device=", 0) == 0)
      cfg.rgb_device = a.substr(11);
    else if (a.rfind("ir_device=", 0) == 0)
      cfg.ir_device = a.substr(10);
    else if (a.rfind("rgb_width=", 0) == 0) {
      try {
        cfg.rgb_width = std::stoi(a.substr(10));
      } catch (...) {
      }
    } else if (a.rfind("rgb_height=", 0) == 0) {
      try {
        cfg.rgb_height = std::stoi(a.substr(11));
      } catch (...) {
      }
    } else if (a.rfind("ir_width=", 0) == 0) {
      try {
        cfg.ir_width = std::stoi(a.substr(9));
      } catch (...) {
      }
    } else if (a.rfind("ir_height=", 0) == 0) {
      try {
        cfg.ir_height = std::stoi(a.substr(10));
      } catch (...) {
      }
    } else if (a.rfind("data_dir=", 0) == 0)
      cfg.data_dir = a.substr(9);
    else if (a.rfind("models_dir=", 0) == 0)
      models_dir = a.substr(11);
    else if (a.rfind("threshold=", 0) == 0) {
      try {
        cfg.ensemble_threshold = std::stof(a.substr(10));
      } catch (...) {
      }
    } else if (a.rfind("dl_threshold=", 0) == 0) {
      try {
        cfg.dl_threshold = std::stof(a.substr(13));
      } catch (...) {
      }
    } else if (a.rfind("dl_weight=", 0) == 0) {
      try {
        cfg.dl_weight = std::stof(a.substr(10));
      } catch (...) {
      }
    } else if (a.rfind("classical_weight=", 0) == 0) {
      try {
        cfg.classical_weight = std::stof(a.substr(17));
      } catch (...) {
      }
    } else if (a.rfind("detect_conf_threshold=", 0) == 0) {
      try {
        cfg.detect_conf_threshold = std::stof(a.substr(22));
      } catch (...) {
      }
    } else if (a.rfind("detect_nms_threshold=", 0) == 0) {
      try {
        cfg.detect_nms_threshold = std::stof(a.substr(21));
      } catch (...) {
      }
    } else if (a.rfind("num_frames=", 0) == 0) {
      try {
        cfg.num_frames = std::stoi(a.substr(11));
      } catch (...) {
      }
    } else if (a.rfind("frame_interval_ms=", 0) == 0) {
      try {
        cfg.frame_interval_ms = std::stoi(a.substr(18));
      } catch (...) {
      }
    } else if (a.rfind("liveness_min_shift=", 0) == 0) {
      try {
        cfg.liveness_min_shift = std::stof(a.substr(19));
      } catch (...) {
      }
    } else if (a.rfind("liveness_max_shift=", 0) == 0) {
      try {
        cfg.liveness_max_shift = std::stof(a.substr(19));
      } catch (...) {
      }
    } else if (a == "debug" || a == "debug=true")
      cfg.debug = true;
    else if (a == "debug=false")
      cfg.debug = false;
    else if (a == "debug_save_frames" || a == "debug_save_frames=true")
      cfg.debug_save_frames = true;
    else if (a == "debug_save_frames=false")
      cfg.debug_save_frames = false;
  }
  if (!models_dir.empty()) {
    cfg.detector_model =
        fs::path(models_dir) / "face_detection_yunet_2023mar.onnx";
    cfg.recognizer_model = fs::path(models_dir) / "w600k_mbf.onnx";
  }
  return cfg;
}

PAM_EXTERN int pam_sm_authenticate(pam_handle_t *pamh, int /*flags*/, int argc,
                                   const char **argv) {
  Config cfg = build_config(pamh, argc, argv);

  try {
    FaceDB db(cfg.data_dir, cfg.debug);
    db.load();
    if (db.empty()) {
      pam_syslog(pamh, LOG_WARNING, "face_verify: no enrolled faces in %s",
                 cfg.data_dir.c_str());
      return PAM_AUTH_ERR;
    }

    DualCapture capture(cfg);
    if (!capture.open()) {
      pam_syslog(pamh, LOG_ERR, "face_verify: failed to open cameras (%s, %s)",
                 cfg.rgb_device.c_str(), cfg.ir_device.c_str());
      return PAM_AUTH_ERR;
    }

    FaceDetector detector(cfg);
    DeepRecognizer deep(cfg);
    ClassicalRecognizer classical;
    Ensemble ensemble(cfg);

    auto frames = capture.grab_sequence(cfg.num_frames, cfg.frame_interval_ms);
    capture.close();

    if (frames.empty()) {
      pam_syslog(pamh, LOG_ERR, "face_verify: no frames captured");
      return PAM_AUTH_ERR;
    }

    std::vector<FaceDetection> rgb_detections;
    cv::Mat best_rgb_aligned, best_ir_aligned;
    float best_rgb_conf = 0.f, best_ir_conf = 0.f;
    bool got_ir = false;

    for (auto &fp : frames) {
      auto det_rgb = detector.detect_best(fp.rgb);
      auto det_ir = detector.detect_best(fp.ir);

      if (det_rgb) {
        rgb_detections.push_back(*det_rgb);
        if (det_rgb->confidence > best_rgb_conf) {
          best_rgb_conf = det_rgb->confidence;
          best_rgb_aligned = FaceDetector::align_face(fp.rgb, *det_rgb, 112);
        }
      }
      if (det_ir) {
        got_ir = true;
        if (det_ir->confidence > best_ir_conf) {
          best_ir_conf = det_ir->confidence;
          best_ir_aligned = FaceDetector::align_face(fp.ir, *det_ir, 112);
        }
      }
    }

    if (best_rgb_aligned.empty() || !got_ir || best_ir_aligned.empty()) {
      pam_syslog(pamh, LOG_INFO, "face_verify: no face detected in cameras");
      return PAM_AUTH_ERR;
    }

    if (!Ensemble::check_liveness(rgb_detections, cfg.liveness_min_shift,
                                  cfg.liveness_max_shift, cfg.debug)) {
      pam_syslog(pamh, LOG_INFO, "face_verify: liveness check failed");
      return PAM_AUTH_ERR;
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
      pam_syslog(pamh, LOG_NOTICE, "face_verify: ACCEPT as %s (score=%.3f)",
                 best_result.matched_label.c_str(), best_result.ensemble_score);
      return PAM_SUCCESS;
    } else {
      pam_syslog(pamh, LOG_INFO,
                 "face_verify: REJECT (best score=%.3f, threshold=%.2f)",
                 best_result.ensemble_score, cfg.ensemble_threshold);
      return PAM_AUTH_ERR;
    }

  } catch (const std::exception &e) {
    pam_syslog(pamh, LOG_ERR, "face_verify: exception: %s", e.what());
    return PAM_AUTH_ERR;
  }
}

PAM_EXTERN int pam_sm_setcred(pam_handle_t *, int, int, const char **) {
  return PAM_SUCCESS;
}
