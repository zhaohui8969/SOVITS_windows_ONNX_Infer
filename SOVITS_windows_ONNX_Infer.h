// SOVITS_windows_ONNX_Infer.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。

#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

void func_load_config_file();

int enable_cuda(OrtSessionOptions* session_options);

std::vector<Ort::Value> func_hubert_benchmark(Ort::Session& session, OrtMemoryInfo* memory_info, Ort::RunOptions& runOptions);
std::vector<Ort::Value> func_vits_benchmark(Ort::Session& session, OrtMemoryInfo* memory_info, Ort::RunOptions& runOptions);

void func_hubert_get_embed(Ort::Session& session, OrtMemoryInfo* memory_info, Ort::RunOptions& runOptions, std::vector<float> fSamples, std::vector<Ort::Value>* returnList);
void func_vits_get_audio(Ort::Session& session, OrtMemoryInfo* memory_info, Ort::RunOptions& runOptions, std::vector<float> fHiddentUnit, std::vector<int64_t> iLength, std::vector<int64_t> iPitch, std::vector<int64_t> iSid, std::vector<Ort::Value>* returnList);
