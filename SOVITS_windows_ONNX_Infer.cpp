// SOVITS_windows_ONNX_Infer.cpp: 定义应用程序的入口点。
//
#define UNICODE
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <algorithm>
#include "SOVITS_windows_ONNX_Infer.h"
#include <string>
#include "spdlog/spdlog.h"
#include <math.h>
#include <numeric>
#include <chrono>
using namespace std::chrono;

#pragma comment(lib, "onnxruntime.lib")
using namespace std;

int main()
{
	LOG_INFO("程序启动!");
	LOG_INFO("读取配置文件...");
	STRUCT_PROJECT_CONFIG projectConfig;
	wchar_t sModelfile[100];
	wcscpy(sModelfile, L"hubert.onnx");
	projectConfig.sONNXModelFile = sModelfile;

	Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "SOVITS_windows_ONNX_Infer" };

	LOG_INFO("载入HuBERT ONNX模型...");
	Ort::Session hubertORTSession{ env, projectConfig.sONNXModelFile, Ort::SessionOptions{nullptr} };

	// HuBERT
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

	std::vector<long long> fUseTimeList;
	milliseconds  tStart;
	milliseconds tUseTime;
	for (int i = 0; i < 20; i++) {
		tStart = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
		hubertORTSession.Run(runOptions, input_names, &inputTensorSource, 1, output_names, &inputTensorEmbed, 1);
		tUseTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch()) - tStart;
		fUseTimeList.push_back(tUseTime.count());
		LOG_INFO("单次推理耗时:%ldms", tUseTime.count());
	}
	long fAvgUseTime = std::accumulate(fUseTimeList.begin(), fUseTimeList.end(), 0) / fUseTimeList.size();
	LOG_INFO("平均耗时:%ldms", fAvgUseTime);

	

	// Get F0

	// VITS
	LOG_INFO("载入VITS ONNX模型...");




	// HTTP SERVER
	spdlog::info("启动HTTP服务");
	spdlog::info("程序退出!");
	return 0;
}
