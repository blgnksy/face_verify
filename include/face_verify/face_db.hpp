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
#pragma once

#include "face_verify/types.hpp"

namespace fv {

class FaceDB {
public:
  explicit FaceDB(const fs::path &data_dir, bool debug = false);

  // Save a face record to disk
  bool enroll(const FaceRecord &record);

  // Load all enrolled faces from disk
  bool load();

  // Get all enrolled records
  const std::vector<FaceRecord> &records() const { return records_; }

  // Check if any faces are enrolled
  bool empty() const { return records_.empty(); }

  // Delete a specific enrollment
  bool remove(const std::string &label);

  // List enrolled labels
  std::vector<std::string> list_labels() const;

private:
  fs::path data_dir_;
  bool debug_{false};
  std::vector<FaceRecord> records_;

  fs::path record_path(const std::string &label) const;
};

} // namespace fv
