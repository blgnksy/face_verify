/**
 * face_verify/face_db — enrolled face record persistence (YAML-backed
 * database). Copyright (C) 2026  Bilgin Aksoy
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
#include "face_verify/face_db.hpp"
#include <algorithm>
#include <iostream>
#include <opencv2/core.hpp>

namespace fv {

FaceDB::FaceDB(const fs::path &data_dir, bool debug)
    : data_dir_(data_dir), debug_(debug) {
  fs::create_directories(data_dir_);
}

fs::path FaceDB::record_path(const std::string &label) const {
  return data_dir_ / (label + ".yml");
}

bool FaceDB::enroll(const FaceRecord &record) {
  try {
    cv::FileStorage fs(record_path(record.label).string(),
                       cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
      std::cerr << "[db] Failed to open file for writing: " << record.label
                << "\n";
      return false;
    }

    fs << "label" << record.label;

    auto write_emb = [&](const std::string &key,
                         const std::vector<float> &emb) {
      if (emb.empty())
        return;
      cv::Mat m(1, static_cast<int>(emb.size()), CV_32F,
                const_cast<float *>(emb.data()));
      fs << key << m.clone();
    };
    write_emb("dl_embedding", record.dl_embedding);
    write_emb("dl_embedding_ir", record.dl_embedding_ir);

    // Store classical features
    fs << "lbp_histogram" << record.lbp_histogram;
    fs << "aligned_face_gray" << record.aligned_face_gray;
    fs << "ir_lbp_histogram" << record.ir_lbp_histogram;
    fs << "ir_aligned_face_gray" << record.ir_aligned_face_gray;

    fs.release();

    // Update in-memory records
    auto it =
        std::find_if(records_.begin(), records_.end(),
                     [&](const auto &r) { return r.label == record.label; });
    if (it != records_.end()) {
      *it = record;
    } else {
      records_.push_back(record);
    }

    if (debug_)
      std::cerr << "[db] Enrolled: " << record.label << "\n";
    return true;
  } catch (const std::exception &e) {
    std::cerr << "[db] Enrollment error: " << e.what() << "\n";
    return false;
  }
}

bool FaceDB::load() {
  records_.clear();

  if (!fs::exists(data_dir_))
    return true; // empty DB is valid

  for (const auto &entry : fs::directory_iterator(data_dir_)) {
    if (entry.path().extension() != ".yml")
      continue;

    try {
      cv::FileStorage fs(entry.path().string(), cv::FileStorage::READ);
      if (!fs.isOpened())
        continue;

      FaceRecord record;
      fs["label"] >> record.label;

      auto read_emb = [&](const std::string &key, std::vector<float> &out) {
        cv::Mat m;
        fs[key] >> m;
        if (!m.empty())
          out.assign(m.ptr<float>(0), m.ptr<float>(0) + m.cols);
      };
      read_emb("dl_embedding", record.dl_embedding);
      read_emb("dl_embedding_ir",
               record.dl_embedding_ir); // absent in old files → empty

      fs["lbp_histogram"] >> record.lbp_histogram;
      fs["aligned_face_gray"] >> record.aligned_face_gray;
      fs["ir_lbp_histogram"] >> record.ir_lbp_histogram;
      fs["ir_aligned_face_gray"] >> record.ir_aligned_face_gray;

      fs.release();
      records_.push_back(std::move(record));

      if (debug_)
        std::cerr << "[db] Loaded: " << records_.back().label << "\n";
    } catch (const std::exception &e) {
      std::cerr << "[db] Failed to load " << entry.path() << ": " << e.what()
                << "\n";
    }
  }

  if (debug_)
    std::cerr << "[db] " << records_.size() << " face(s) loaded\n";
  return true;
}

bool FaceDB::remove(const std::string &label) {
  auto path = record_path(label);
  if (fs::exists(path)) {
    fs::remove(path);
  }
  auto it = std::remove_if(records_.begin(), records_.end(),
                           [&](const auto &r) { return r.label == label; });
  bool found = (it != records_.end());
  records_.erase(it, records_.end());
  return found;
}

std::vector<std::string> FaceDB::list_labels() const {
  std::vector<std::string> labels;
  labels.reserve(records_.size());
  for (const auto &r : records_) {
    labels.push_back(r.label);
  }
  return labels;
}

} // namespace fv
