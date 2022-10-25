// SOVITS_windows_ONNX_Infer.cpp: 定义应用程序的入口点。
//
#define UNICODE
#define COBJMACROS
#include "SOVITS_windows_ONNX_Infer.h"
#include <onnxruntime_cxx_api.h>
#include "providers.h"
#include <array>
#include <cmath>
#include <algorithm>
#include <string>
#include "spdlog/spdlog.h"
#include <math.h>
#include <numeric>
#include <chrono>
#include <dxgi.h>
using namespace std::chrono;

#pragma comment(lib, "onnxruntime.lib")
#pragma comment(lib, "onnxruntime_providers_cuda.lib")
#pragma comment(lib, "onnxruntime_providers_shared.lib")
#pragma comment(lib, "dxgi.lib")

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

const OrtApi* g_ort = NULL;

std::vector <IDXGIAdapter*> EnumerateAdapters(void)
{
	IDXGIAdapter* pAdapter;
	std::vector <IDXGIAdapter*> vAdapters;
	IDXGIFactory* pFactory = NULL;


	// Create a DXGIFactory object.
	if (FAILED(CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory)))
	{
		return vAdapters;
	}


	for (UINT i = 0;
		pFactory->EnumAdapters(i, &pAdapter) != DXGI_ERROR_NOT_FOUND;
		++i)
	{
		vAdapters.push_back(pAdapter);
	}


	if (pFactory)
	{
		pFactory->Release();
	}

	return vAdapters;

}

int enable_cuda(OrtSessionOptions* session_options) {
	// OrtCUDAProviderOptions is a C struct. C programming language doesn't have constructors/destructors.
	OrtCUDAProviderOptions o;
	// Here we use memset to initialize every field of the above data struct to zero.
	memset(&o, 0, sizeof(o));
	// But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
	// type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
	o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
	o.gpu_mem_limit = SIZE_MAX;
	OrtStatus* onnx_status = g_ort->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
	if (onnx_status != NULL) {
		const char* msg = g_ort->GetErrorMessage(onnx_status);
		LOG_INFO("启用CUDA错误:%s", msg);
		g_ort->ReleaseStatus(onnx_status);
		return -1;
	}
	return 0;
}

void func_check_dx_device() {
	auto adapters = EnumerateAdapters();
	LOG_INFO("DX设备数量:%d", adapters.size());
	for (int i = 0; i < adapters.size(); i++) {
		DXGI_ADAPTER_DESC desc;
		adapters[i]->GetDesc(&desc);
		//spdlog::info(desc.Description);
		char tmp[100];
		wcstombs(tmp, desc.Description, 100);
		LOG_INFO("设备%d:%s", i, tmp);
	}
}

int main()
{
	LOG_INFO("程序启动!");
	LOG_INFO("读取配置文件...");

	func_check_dx_device();
	int ret = 0;

	std::vector<long long> fUseTimeList;
	int iSkipWarmupStep = 5;
	int iBenchMarkStep = 100;
	milliseconds  tStart;
	milliseconds tUseTime;
	long fAvgUseTime;

	STRUCT_PROJECT_CONFIG projectConfig;
	wchar_t sModelfile[100];
	wcscpy(sModelfile, L"hubert.onnx");
	projectConfig.sONNXModelFile = sModelfile;

	g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "SOVITS_windows_ONNX_Infer" };
	Ort::RunOptions runOptions;
	OrtMemoryInfo* memory_info;

	auto ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	ortApi->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
	//auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	LOG_INFO("载入HuBERT ONNX模型...");
	Ort::SessionOptions sessionOptions;

	LOG_INFO("启用CUDA...");
	ret = enable_cuda(sessionOptions);
	if (ret) {
		LOG_INFO("CUD不可用");
	}
	else {
		LOG_INFO("CUDA启用成功");
	}
	//OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0);
	//sessionOptions.DisableMemPattern();
	//sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
	Ort::Session hubertORTSession{ env, projectConfig.sONNXModelFile, sessionOptions };
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
		Ort::Value::CreateTensor<float>(memory_info, fTensorSource.data(), fTensorSource.size(), iTensorSourceShape.data(), iTensorSourceShape.size())
	);

	const char* ONNXInputNames[] = { "source" };
	const char* ONNXoutputNames[] = { "embed" };

	fUseTimeList.clear();
	for (int i = 0; i < iBenchMarkStep; i++) {
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
	fAvgUseTime = std::accumulate(fUseTimeList.begin() + iSkipWarmupStep, fUseTimeList.end(), 0) / (fUseTimeList.size() - iSkipWarmupStep);
	LOG_INFO("ONNX 1s音频平均耗时:%lldms", fAvgUseTime);


	// VITS
	LOG_INFO("载入VITS ONNX模型...");
	Ort::Session VITSORTSession{ env, L"121_epochs.onnx", sessionOptions };
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
		Ort::Value::CreateTensor<float>(memory_info, fHiddentUnit.data(), fHiddentUnit.size(), iHiddentUnitShape.data(), iHiddentUnitShape.size())
	);
	VITSInputValues.emplace_back(
		Ort::Value::CreateTensor<int64_t>(memory_info, iLength.data(), iLength.size(), iLengthShape.data(), iLengthShape.size())
	);
	VITSInputValues.emplace_back(
		Ort::Value::CreateTensor<int64_t>(memory_info, iPitch.data(), iPitch.size(), iPitchShape.data(), iPitchShape.size())
	);
	VITSInputValues.emplace_back(
		Ort::Value::CreateTensor<int64_t>(memory_info, iSid.data(), iSid.size(), iSidShape.data(), iSidShape.size())
	);

	const char* VITSInputNames[] = { "hidden_unit", "lengths", "pitch", "sid" };
	const char* VITSoutputNames[] = { "audio" };

	fUseTimeList.clear();
	for (int i = 0; i < iBenchMarkStep; i++) {
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
	fAvgUseTime = std::accumulate(fUseTimeList.begin() + iSkipWarmupStep, fUseTimeList.end(), 0) / (fUseTimeList.size() - iSkipWarmupStep);
	LOG_INFO("VITS 1s音频平均耗时:%ldms", fAvgUseTime);

	// HTTP SERVER
	LOG_INFO("启动HTTP服务");
	LOG_INFO("程序退出!");

	// Get F0

	return 0;
}
