#include "face_verify/config.hpp"
#include "face_verify/types.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace fv;
namespace fs = std::filesystem;

static int failures = 0;

#define ASSERT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "  FAIL: " #cond " (line " << __LINE__ << ")\n";            \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

#define ASSERT_NEAR(a, b, tol)                                                 \
  ASSERT(std::abs(static_cast<float>(a) - static_cast<float>(b)) <             \
         static_cast<float>(tol))

static fs::path write_config(const std::string &content) {
  fs::path p = fs::temp_directory_path() / "fv_test.conf";
  std::ofstream f(p);
  f << content;
  return p;
}

int main() {
  std::cerr << "=== Config Tests ===\n";

  // ── Parse all keys ──
  {
    auto p = write_config("# comment\n"
                          "\n"
                          "rgb_device = /dev/video4\n"
                          "ir_device = /dev/video6\n"
                          "rgb_width = 1280\n"
                          "rgb_height = 720\n"
                          "ir_width = 848\n"
                          "ir_height = 480\n"
                          "data_dir = /tmp/mydata\n"
                          "models_dir = /tmp/mymodels\n"
                          "threshold = 0.65\n"
                          "dl_threshold = 0.35\n"
                          "num_frames = 5\n"
                          "frame_interval_ms = 300\n"
                          "debug = true\n"
                          "debug_save_frames = true\n"
                          "unknown_key = ignored\n");

    Config cfg;
    load_config(cfg, p);

    ASSERT(cfg.rgb_device == "/dev/video4");
    ASSERT(cfg.ir_device == "/dev/video6");
    ASSERT(cfg.rgb_width == 1280);
    ASSERT(cfg.rgb_height == 720);
    ASSERT(cfg.ir_width == 848);
    ASSERT(cfg.ir_height == 480);
    ASSERT(cfg.data_dir == "/tmp/mydata");
    ASSERT(cfg.detector_model ==
           fs::path("/tmp/mymodels/face_detection_yunet_2023mar.onnx"));
    ASSERT(cfg.recognizer_model == fs::path("/tmp/mymodels/w600k_mbf.onnx"));
    ASSERT_NEAR(cfg.ensemble_threshold, 0.65f, 0.001f);
    ASSERT_NEAR(cfg.dl_threshold, 0.35f, 0.001f);
    ASSERT(cfg.num_frames == 5);
    ASSERT(cfg.frame_interval_ms == 300);
    ASSERT(cfg.debug == true);
    ASSERT(cfg.debug_save_frames == true);

    fs::remove(p);
  }

  // ── Boolean variants ──
  {
    auto p = write_config("debug = 1\ndebug_save_frames = yes\n");
    Config cfg;
    load_config(cfg, p);
    ASSERT(cfg.debug == true);
    ASSERT(cfg.debug_save_frames == true);
    fs::remove(p);
  }
  {
    auto p = write_config("debug = false\n");
    Config cfg;
    load_config(cfg, p);
    ASSERT(cfg.debug == false);
    fs::remove(p);
  }

  // ── Missing models_dir leaves model paths unchanged ──
  {
    auto p = write_config("rgb_device = /dev/video0\n");
    Config cfg;
    fs::path orig_det = cfg.detector_model;
    load_config(cfg, p);
    ASSERT(cfg.detector_model == orig_det); // unchanged
    fs::remove(p);
  }

  // ── Save/load round-trip ──
  {
    Config cfg;
    cfg.rgb_device = "/dev/video8";
    cfg.ensemble_threshold = 0.72f;
    cfg.debug = true;
    cfg.num_frames = 7;

    fs::path p = fs::temp_directory_path() / "fv_roundtrip.conf";
    save_config(cfg, p);

    Config cfg2;
    load_config(cfg2, p);
    ASSERT(cfg2.rgb_device == "/dev/video8");
    ASSERT_NEAR(cfg2.ensemble_threshold, 0.72f, 0.001f);
    ASSERT(cfg2.debug == true);
    ASSERT(cfg2.num_frames == 7);

    fs::remove(p);
  }

  // ── Nonexistent file throws ──
  {
    bool threw = false;
    try {
      Config cfg;
      load_config(cfg, "/nonexistent/path/face_verify.conf");
    } catch (const std::exception &) {
      threw = true;
    }
    ASSERT(threw);
  }

  if (failures == 0) {
    std::cerr << "All tests passed.\n";
    return 0;
  }
  std::cerr << failures << " test(s) failed.\n";
  return 1;
}
