// SOVITS_windows_ONNX_Infer.cpp: 定义应用程序的入口点。
//
#define UNICODE
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <algorithm>
#include "SOVITS_windows_ONNX_Infer.h"
#include <string>

#pragma comment(lib, "onnxruntime.lib")
using namespace std;

int main()
{

	STRUCT_PROJECT_CONFIG projectConfig;
	wchar_t sModelfile[100];
	wcscpy(sModelfile, L"hubert.onnx");
	projectConfig.sONNXModelFile = sModelfile;

	Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "SOVITS_windows_ONNX_Infer" };

	Ort::Session session_{ env, projectConfig.sONNXModelFile, Ort::SessionOptions{nullptr} };

	//  source = {"source":np.expand_dims(np.expand_dims(wav16,0),0)}
	//	units = np.array(hubertsession.run(['embed'], source)[0])

	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	std::array<float, 1 * 1 * 16000> fTensorSource = {};
	std::array<int64_t, 3> iTensorSourceShape{ 1, 1, 16000 };

	std::array<float, 1 * 50 * 256> fTensorEmbed = {};
	std::array<int64_t, 3> iTensorEmbedShape{ 1, 50, 256 };

	auto inputTensorSource = Ort::Value::CreateTensor<float>(allocator_info, fTensorSource.data(), fTensorSource.size(), iTensorSourceShape.data(), iTensorSourceShape.size());
	auto inputTensorEmbed = Ort::Value::CreateTensor<float>(allocator_info, fTensorEmbed.data(), fTensorEmbed.size(), iTensorEmbedShape.data(), iTensorEmbedShape.size());

	const char* input_names[] = { "source" };
	const char* output_names[] = { "embed" };

	Ort::RunOptions runOptions;
	session_.Run(runOptions, input_names, &inputTensorSource, 1, output_names, &inputTensorEmbed, 1);

	return 0;
}
