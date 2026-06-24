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
#include "face_verify/deep_recognizer.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/imgproc.hpp>

namespace fv {

DeepRecognizer::DeepRecognizer(const Config &cfg)
    : env_(ORT_LOGGING_LEVEL_WARNING, "face_verify") {
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(2);
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  std::string model_path = cfg.recognizer_model.string();
  session_ = Ort::Session(env_, model_path.c_str(), opts);

  // Query input/output names
  size_t num_inputs = session_.GetInputCount();
  size_t num_outputs = session_.GetOutputCount();

  for (size_t i = 0; i < num_inputs; ++i) {
    auto name = session_.GetInputNameAllocated(i, allocator_);
    input_name_strs_.push_back(name.get());
  }
  for (size_t i = 0; i < num_outputs; ++i) {
    auto name = session_.GetOutputNameAllocated(i, allocator_);
    output_name_strs_.push_back(name.get());
  }

  // Build const char* arrays from stored strings
  for (auto &s : input_name_strs_)
    input_names_.push_back(s.c_str());
  for (auto &s : output_name_strs_)
    output_names_.push_back(s.c_str());

  // Get input shape to determine H, W
  auto input_shape =
      session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  // Expected: [1, 3, 112, 112] or [batch, 3, H, W]
  if (input_shape.size() == 4) {
    input_h_ = static_cast<int>(input_shape[2]);
    input_w_ = static_cast<int>(input_shape[3]);
  }

  // Get output embedding dimension
  auto output_shape =
      session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  if (output_shape.size() == 2) {
    embedding_dim_ = static_cast<int>(output_shape[1]);
  }

  std::cerr << "[deep] ArcFace loaded — input: " << input_h_ << "x" << input_w_
            << " embedding: " << embedding_dim_ << "d\n";
}

DeepRecognizer::~DeepRecognizer() = default;

std::vector<float> DeepRecognizer::preprocess(const cv::Mat &aligned_face) {
  cv::Mat face;

  // Ensure BGR 3-channel
  if (aligned_face.channels() == 1) {
    cv::cvtColor(aligned_face, face, cv::COLOR_GRAY2BGR);
  } else {
    face = aligned_face;
  }

  // Resize if needed
  if (face.rows != input_h_ || face.cols != input_w_) {
    cv::resize(face, face, cv::Size(input_w_, input_h_));
  }

  // Convert to float32
  face.convertTo(face, CV_32F);

  // ArcFace normalization: (pixel - 127.5) / 127.5 → range [-1, 1]
  face = (face - 127.5f) / 127.5f;

  // HWC → CHW (NCHW layout for ONNX)
  std::vector<cv::Mat> channels(3);
  cv::split(face, channels);

  std::vector<float> tensor;
  tensor.reserve(3 * input_h_ * input_w_);
  for (auto &ch : channels) {
    tensor.insert(tensor.end(), ch.ptr<float>(0),
                  ch.ptr<float>(0) + input_h_ * input_w_);
  }

  return tensor;
}

std::vector<float> DeepRecognizer::get_embedding(const cv::Mat &aligned_face) {
  auto tensor_data = preprocess(aligned_face);

  // Create input tensor
  std::array<int64_t, 4> input_shape = {1, 3, input_h_, input_w_};
  auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, tensor_data.data(), tensor_data.size(), input_shape.data(),
      input_shape.size());

  // Run inference
  auto output_tensors =
      session_.Run(Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor,
                   1, output_names_.data(), output_names_.size());

  // Extract embedding
  float *output_data = output_tensors[0].GetTensorMutableData<float>();
  std::vector<float> embedding(output_data, output_data + embedding_dim_);

  // L2 normalize
  float norm = 0.f;
  for (float v : embedding)
    norm += v * v;
  norm = std::sqrt(norm);
  if (norm > 1e-6f) {
    for (float &v : embedding)
      v /= norm;
  }

  return embedding;
}

float DeepRecognizer::cosine_similarity(const std::vector<float> &a,
                                        const std::vector<float> &b) {
  assert(a.size() == b.size());
  float dot = 0.f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
  }
  // Both vectors are L2-normalized, so dot product = cosine similarity
  return dot;
}

float DeepRecognizer::verify(const std::vector<float> &live,
                             const std::vector<float> &reference) {
  return cosine_similarity(live, reference);
}

} // namespace fv
