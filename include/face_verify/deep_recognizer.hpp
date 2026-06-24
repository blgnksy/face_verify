/**
 * face_verify/deep_recognizer — ArcFace deep-learning face recognition via ONNX
 * Runtime. Copyright (C) 2026  Bilgin Aksoy
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
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace fv {

class DeepRecognizer {
public:
  explicit DeepRecognizer(const Config &cfg);
  ~DeepRecognizer();

  // Non-copyable (ONNX session is not copyable)
  DeepRecognizer(const DeepRecognizer &) = delete;
  DeepRecognizer &operator=(const DeepRecognizer &) = delete;

  // Extract 512-d embedding from an aligned 112x112 face (BGR or grayscale)
  std::vector<float> get_embedding(const cv::Mat &aligned_face);

  // Cosine similarity between two embeddings
  static float cosine_similarity(const std::vector<float> &a,
                                 const std::vector<float> &b);

  // Verify: compare a live embedding against a stored reference
  float verify(const std::vector<float> &live,
               const std::vector<float> &reference);

  int embedding_dim() const { return embedding_dim_; }

private:
  // Preprocess aligned face to NCHW float tensor
  std::vector<float> preprocess(const cv::Mat &aligned_face);

  Ort::Env env_;
  Ort::Session session_{nullptr};
  Ort::AllocatorWithDefaultOptions allocator_;

  std::vector<const char *> input_names_;
  std::vector<const char *> output_names_;
  // We store the actual strings to keep the const char* pointers valid
  std::vector<std::string> input_name_strs_;
  std::vector<std::string> output_name_strs_;

  int input_h_{112};
  int input_w_{112};
  int embedding_dim_{512};
};

} // namespace fv
