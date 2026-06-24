#include "face_verify/ensemble.hpp"
#include "face_verify/types.hpp"
#include <cmath>
#include <iostream>

using namespace fv;

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

// ── Ensemble::fuse ────────────────────────────────────────────

static void test_fuse_math() {
  Config cfg; // default thresholds: dl=0.40, ensemble=0.50, weights: dl=0.6,
              // cl=0.4
  Ensemble ens(cfg);

  // dl_fused  = 0.4*0.8 + 0.6*0.9 = 0.32 + 0.54 = 0.86
  // cl_fused  = 0.4*0.7 + 0.6*0.8 = 0.28 + 0.48 = 0.76
  // ensemble  = 0.6*0.86 + 0.4*0.76 = 0.516 + 0.304 = 0.82
  auto r = ens.fuse(0.8f, 0.9f, 0.7f, 0.8f, "alice");
  ASSERT_NEAR(r.dl_score, 0.86f, 0.001f);
  ASSERT_NEAR(r.classical_score, 0.76f, 0.001f);
  ASSERT_NEAR(r.ensemble_score, 0.82f, 0.001f);
  ASSERT(r.accepted);
  ASSERT(r.matched_label == "alice");
}

static void test_fuse_accept_high_scores() {
  Config cfg;
  Ensemble ens(cfg);
  auto r = ens.fuse(0.9f, 0.9f, 0.9f, 0.9f, "bob");
  ASSERT(r.accepted);
}

static void test_fuse_reject_dl_below_threshold() {
  Config cfg; // dl_threshold = 0.40
  Ensemble ens(cfg);
  // DL fused = 0.3 < 0.40 → reject even if classical is high
  auto r = ens.fuse(0.3f, 0.3f, 0.99f, 0.99f, "eve");
  ASSERT(!r.accepted);
}

static void test_fuse_reject_low_ensemble() {
  Config cfg;
  Ensemble ens(cfg);
  auto r = ens.fuse(0.2f, 0.2f, 0.2f, 0.2f, "nobody");
  ASSERT(!r.accepted);
  ASSERT(r.ensemble_score < cfg.ensemble_threshold);
}

static void test_fuse_ir_weighted_more() {
  Config cfg;
  Ensemble ens(cfg);
  // IR weight=0.6 dominates RGB weight=0.4
  // rgb=0.0, ir=1.0 → fused = 0.4*0.0 + 0.6*1.0 = 0.6
  auto r = ens.fuse(0.0f, 1.0f, 0.0f, 1.0f, "test");
  ASSERT_NEAR(r.dl_score, 0.6f, 0.001f);
  ASSERT_NEAR(r.classical_score, 0.6f, 0.001f);
}

static void test_fuse_custom_threshold() {
  Config cfg;
  cfg.ensemble_threshold = 0.90f; // very strict
  Ensemble ens(cfg);
  auto r = ens.fuse(0.7f, 0.7f, 0.7f, 0.7f, "test");
  ASSERT(!r.accepted); // would accept at default threshold, not at 0.90
}

// ── Ensemble::check_liveness ──────────────────────────────────

static void test_liveness_too_few_detections() {
  ASSERT(!Ensemble::check_liveness({}));

  std::vector<FaceDetection> one;
  one.push_back({});
  ASSERT(!Ensemble::check_liveness(one));
}

static void test_liveness_no_movement() {
  FaceDetection d;
  d.bbox = {100.f, 100.f, 100.f, 100.f};
  std::vector<FaceDetection> dets = {d, d, d};
  ASSERT(!Ensemble::check_liveness(dets));
}

static void test_liveness_normal_movement() {
  FaceDetection d1, d2, d3;
  d1.bbox = {100.f, 100.f, 100.f, 100.f};
  d2.bbox = {101.5f, 101.f, 100.f,
             100.f}; // ~1.8 px shift — above default min 1.0
  d3.bbox = {103.f, 102.f, 100.f, 100.f}; // ~1.8 px shift
  std::vector<FaceDetection> dets = {d1, d2, d3};
  ASSERT(Ensemble::check_liveness(dets));
}

static void test_liveness_excessive_movement() {
  FaceDetection d1, d2;
  d1.bbox = {0.f, 0.f, 100.f, 100.f};
  d2.bbox = {200.f, 200.f, 100.f, 100.f}; // ~283 px — camera shake
  std::vector<FaceDetection> dets = {d1, d2};
  ASSERT(!Ensemble::check_liveness(dets));
}

int main() {
  std::cerr << "=== Ensemble Tests ===\n";

  test_fuse_math();
  test_fuse_accept_high_scores();
  test_fuse_reject_dl_below_threshold();
  test_fuse_reject_low_ensemble();
  test_fuse_ir_weighted_more();
  test_fuse_custom_threshold();
  test_liveness_too_few_detections();
  test_liveness_no_movement();
  test_liveness_normal_movement();
  test_liveness_excessive_movement();

  if (failures == 0) {
    std::cerr << "All tests passed.\n";
    return 0;
  }
  std::cerr << failures << " test(s) failed.\n";
  return 1;
}
