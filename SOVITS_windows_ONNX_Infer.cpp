// SOVITS_windows_ONNX_Infer.cpp: 定义应用程序的入口点。
//
#define UNICODE
#include "SOVITS_windows_ONNX_Infer.h"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <algorithm>
#include <string>
#include "spdlog/spdlog.h"
#include <math.h>
#include <numeric>
#include <chrono>
using namespace std::chrono;

#pragma comment(lib, "onnxruntime.lib")
using namespace std;

/*
void func_inspect_session(Ort::Session& session) {
	Ort::AllocatorWithDefaultOptions ort_alloc;
	size_t sInputCount = session.GetInputCount();
	for (int i = 0; i < sInputCount; i++) {
		auto typeInfo = session.GetInputTypeInfo(i);
		auto name = session.GetInputName(i, ort_alloc);
		auto typeInfo = session.GetInputTypeInfo(i);
		auto typeAndShape = typeInfo.GetTensorTypeAndShapeInfo();
		auto type = typeAndShape.GetElementType();
		auto shape = typeAndShape.GetShape();
	}
}
*/

int main()
{
	LOG_INFO("程序启动!");
	LOG_INFO("读取配置文件...");
	
	std::vector<long long> fUseTimeList;
	milliseconds  tStart;
	milliseconds tUseTime;
	long fAvgUseTime;

	STRUCT_PROJECT_CONFIG projectConfig;
	wchar_t sModelfile[100];
	wcscpy(sModelfile, L"hubert.onnx");
	projectConfig.sONNXModelFile = sModelfile;

	Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "SOVITS_windows_ONNX_Infer" };
	Ort::RunOptions runOptions;
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	LOG_INFO("载入HuBERT ONNX模型...");
	Ort::Session hubertORTSession{ env, projectConfig.sONNXModelFile, Ort::SessionOptions{nullptr} };
	//func_inspect_session(hubertORTSession);
	LOG_INFO("载入完成...");

	// HuBERT
	//  source = {"source":np.expand_dims(np.expand_dims(wav16,0),0)}
	//	units = np.array(hubertsession.run(['embed'], source)[0])


	std::array<float, 1 * 1 * 16000> fTensorSource = {};
	std::array<int64_t, 3> iTensorSourceShape{ 1, 1, 16000 };

	//std::array<float, 1 * 50 * 256> fTensorEmbed = {};
	//std::array<int64_t, 3> iTensorEmbedShape{ 1, 50, 256 };

	std::vector<Ort::Value> huberInputList;
	huberInputList.emplace_back(
		Ort::Value::CreateTensor<float>(allocator_info, fTensorSource.data(), fTensorSource.size(), iTensorSourceShape.data(), iTensorSourceShape.size())
	);

	const char* ONNXInputNames[] = { "source" };
	const char* ONNXoutputNames[] = { "embed" };



	for (int i = 0; i < 20; i++) {
		tStart = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
		std::vector<Ort::Value> returnList = hubertORTSession.Run(runOptions, ONNXInputNames, huberInputList.data(), huberInputList.size(), ONNXoutputNames, 1);
		auto data = returnList.data();
		auto typeAndShape = data->GetTensorTypeAndShapeInfo();
		auto shape = typeAndShape.GetShape();
		auto fData = data->GetTensorMutableData<float>();	
		tUseTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch()) - tStart;
		fUseTimeList.push_back(tUseTime.count());
		LOG_INFO("ONNX 1s音频单次推理耗时:%lldms", tUseTime.count());
	}
	fAvgUseTime = std::accumulate(fUseTimeList.begin(), fUseTimeList.end(), 0) / fUseTimeList.size();
	LOG_INFO("ONNX 1s音频平均耗时:%lldms", fAvgUseTime);


	// VITS
	LOG_INFO("载入VITS ONNX模型...");
	Ort::Session VITSORTSession{ env, L"121_epochs.onnx", Ort::SessionOptions{nullptr} };
	LOG_INFO("载入完成...");

	/*
	test_hidden_unit = torch.rand(1, 50, 256)
	test_lengths = torch.LongTensor([50])
	test_pitch = (torch.rand(1, 50) * 128).long()
	test_sid = torch.LongTensor([0])
	ONNXInputNames = ["hidden_unit", "lengths", "pitch", "sid"]
	ONNXoutputNames = ["audio", ]
	*/

	std::array<float, 1 * 50 * 256> fHiddentUnit = {};
	std::array<int64_t, 1> iLength = {};
	std::array<int64_t, 1 * 50> iPitch = {};
	std::array<int64_t, 1> iSid = {};

	std::array<int64_t, 3> iHiddentUnitShape{ 1, 50,256 };
	std::array<int64_t, 1> iLengthShape{ 1 };
	std::array<int64_t, 2> iPitchShape{ 1, 50 };
	std::array<int64_t, 1> iSidShape{ 1 };

	std::vector<Ort::Value> VITSInputValues;
	VITSInputValues.emplace_back(
		Ort::Value::CreateTensor<float>(allocator_info, fHiddentUnit.data(), fHiddentUnit.size(), iHiddentUnitShape.data(), iHiddentUnitShape.size())
	);
	VITSInputValues.emplace_back(
		Ort::Value::CreateTensor<int64_t>(allocator_info, iLength.data(), iLength.size(), iLengthShape.data(), iLengthShape.size())
	);
	VITSInputValues.emplace_back(
		Ort::Value::CreateTensor<int64_t>(allocator_info, iPitch.data(), iPitch.size(), iPitchShape.data(), iPitchShape.size())
	);
	VITSInputValues.emplace_back(
		Ort::Value::CreateTensor<int64_t>(allocator_info, iSid.data(), iSid.size(), iSidShape.data(), iSidShape.size())
	);

	const char* VITSInputNames[] = { "hidden_unit", "lengths", "pitch", "sid" };
	const char* VITSoutputNames[] = { "audio" };

	for (int i = 0; i < 20; i++) {
		tStart = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
		std::vector<Ort::Value> returnList = VITSORTSession.Run(runOptions, VITSInputNames, VITSInputValues.data(), VITSInputValues.size(), VITSoutputNames, 1);
		auto data = returnList.data();
		auto typeAndShape = data->GetTensorTypeAndShapeInfo();
		auto shape = typeAndShape.GetShape();
		auto fData = data->GetTensorMutableData<float>();
		tUseTime = duration_cast<milliseconds>(system_clock::now().time_since_epoch()) - tStart;
		fUseTimeList.push_back(tUseTime.count());
		LOG_INFO("VITS 1s音频单次推理耗时:%ldms", tUseTime.count());
	}
	fAvgUseTime = std::accumulate(fUseTimeList.begin(), fUseTimeList.end(), 0) / fUseTimeList.size();
	LOG_INFO("VITS 1s音频平均耗时:%ldms", fAvgUseTime);

	// HTTP SERVER
	spdlog::info("启动HTTP服务");
	spdlog::info("程序退出!");


	// Get F0

	return 0;
}
