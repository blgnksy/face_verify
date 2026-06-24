#include "face_verify/face_db.hpp"
#include "face_verify/types.hpp"
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

static FaceRecord make_record(const std::string &label) {
  FaceRecord r;
  r.label = label;
  r.dl_embedding = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
  r.lbp_histogram = cv::Mat::zeros(1, 10, CV_32F);
  r.aligned_face_gray = cv::Mat::zeros(112, 112, CV_8U);
  r.ir_lbp_histogram = cv::Mat::zeros(1, 10, CV_32F);
  r.ir_aligned_face_gray = cv::Mat::zeros(112, 112, CV_8U);
  return r;
}

int main() {
  std::cerr << "=== FaceDB Tests ===\n";

  // Create an isolated temp directory
  std::string tmpl =
      (fs::temp_directory_path() / "face_verify_test_XXXXXX").string();
  char buf[512];
  std::strncpy(buf, tmpl.c_str(), sizeof(buf) - 1);
  buf[sizeof(buf) - 1] = '\0';
  char *tmppath = mkdtemp(buf);
  if (!tmppath) {
    std::cerr << "FATAL: Could not create temp directory\n";
    return 1;
  }
  fs::path dir(tmppath);

  // ── Empty DB ──
  {
    FaceDB db(dir);
    ASSERT(db.empty());
    ASSERT(db.list_labels().empty());
  }

  // ── Enroll ──
  {
    FaceDB db(dir);
    ASSERT(db.enroll(make_record("alice")));
    ASSERT(fs::exists(dir / "alice.yml"));
  }

  // ── Load and verify round-trip ──
  {
    FaceDB db(dir);
    ASSERT(db.load());
    ASSERT(!db.empty());

    auto labels = db.list_labels();
    ASSERT(labels.size() == 1);
    ASSERT(labels[0] == "alice");

    const auto &records = db.records();
    ASSERT(records.size() == 1);
    ASSERT(records[0].label == "alice");
    ASSERT(records[0].dl_embedding.size() == 5);
    ASSERT_NEAR(records[0].dl_embedding[0], 0.1f, 0.001f);
    ASSERT_NEAR(records[0].dl_embedding[4], 0.5f, 0.001f);
    ASSERT(!records[0].lbp_histogram.empty());
    ASSERT(!records[0].aligned_face_gray.empty());
  }

  // ── Enroll overwrites existing label ──
  {
    FaceDB db(dir);
    db.load();
    auto updated = make_record("alice");
    updated.dl_embedding = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f};
    ASSERT(db.enroll(updated));

    // Reload and check updated embedding
    FaceDB db2(dir);
    db2.load();
    ASSERT_NEAR(db2.records()[0].dl_embedding[0], 0.9f, 0.001f);
  }

  // ── Enroll multiple records ──
  {
    FaceDB db(dir);
    db.load();
    ASSERT(db.enroll(make_record("bob")));

    FaceDB db2(dir);
    db2.load();
    ASSERT(db2.records().size() == 2);
    auto labels = db2.list_labels();
    ASSERT(labels.size() == 2);
  }

  // ── Remove ──
  {
    FaceDB db(dir);
    db.load();
    ASSERT(db.remove("alice"));
    ASSERT(!fs::exists(dir / "alice.yml"));
    ASSERT(!db.remove("nonexistent"));

    FaceDB db2(dir);
    db2.load();
    ASSERT(db2.records().size() == 1);
    ASSERT(db2.records()[0].label == "bob");
  }

  // ── Remove last entry → empty ──
  {
    FaceDB db(dir);
    db.load();
    ASSERT(db.remove("bob"));
    ASSERT(db.empty());
  }

  fs::remove_all(dir);

  if (failures == 0) {
    std::cerr << "All tests passed.\n";
    return 0;
  }
  std::cerr << failures << " test(s) failed.\n";
  return 1;
}
