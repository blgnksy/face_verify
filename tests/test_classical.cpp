#include "face_verify/classical_recognizer.hpp"
#include <cmath>
#include <iostream>
#include <opencv2/core.hpp>

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

// Gradient image — varied texture, works well with LBP
static cv::Mat make_gradient(int size = 112) {
  cv::Mat img(size, size, CV_8U);
  for (int y = 0; y < size; ++y)
    for (int x = 0; x < size; ++x)
      img.at<uint8_t>(y, x) = static_cast<uint8_t>((x * 2 + y) % 256);
  return img;
}

// Checkerboard — high-contrast, distinct from gradient
static cv::Mat make_checkerboard(int size = 112, int cell = 8) {
  cv::Mat img(size, size, CV_8U);
  for (int y = 0; y < size; ++y)
    for (int x = 0; x < size; ++x)
      img.at<uint8_t>(y, x) = (((x / cell) + (y / cell)) % 2) ? 255 : 0;
  return img;
}

int main() {
  std::cerr << "=== Classical Recognizer Tests ===\n";

  ClassicalRecognizer cr;
  cv::Mat grad = make_gradient();
  cv::Mat check = make_checkerboard();

  // ── LBP histogram properties ──
  cv::Mat hist = cr.compute_lbp_histogram(grad);

  ASSERT(!hist.empty());
  ASSERT(hist.type() == CV_32F);

  // All values non-negative
  double minVal;
  cv::minMaxLoc(hist, &minVal);
  ASSERT(minVal >= 0.0);

  // ── compare_lbp: same histogram → similarity = 1.0 ──
  float same_lbp = cr.compare_lbp(hist, hist);
  ASSERT_NEAR(same_lbp, 1.0f, 0.001f);

  // ── compare_lbp: different textures → lower similarity ──
  cv::Mat hist2 = cr.compute_lbp_histogram(check);
  float diff_lbp = cr.compare_lbp(hist, hist2);
  ASSERT(diff_lbp >= 0.0f);
  ASSERT(diff_lbp <= 1.0f);
  ASSERT(diff_lbp < same_lbp);

  // ── compare_lbp: symmetry ──
  ASSERT_NEAR(cr.compare_lbp(hist, hist2), cr.compare_lbp(hist2, hist), 0.001f);

  // ── compare_template: same image → ~1.0 ──
  float same_tmpl = cr.compare_template(grad, grad);
  ASSERT(same_tmpl > 0.95f);

  // ── compare_template: different images → lower ──
  float diff_tmpl = cr.compare_template(grad, check);
  ASSERT(diff_tmpl >= 0.0f);
  ASSERT(diff_tmpl <= 1.0f);
  ASSERT(diff_tmpl < same_tmpl);

  // ── verify: same image → high combined score ──
  cv::Mat hist_grad = cr.compute_lbp_histogram(grad);
  float score = cr.verify(grad, hist_grad, grad);
  ASSERT(score >= 0.0f && score <= 1.0f);
  ASSERT(score > 0.85f);

  // ── verify: different image → lower score ──
  float score_diff = cr.verify(check, hist_grad, grad);
  ASSERT(score_diff < score);

  if (failures == 0) {
    std::cerr << "All tests passed.\n";
    return 0;
  }
  std::cerr << failures << " test(s) failed.\n";
  return 1;
}
