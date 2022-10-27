// SOVITS_windows_ONNX_Infer.cpp: 定义应用程序的入口点。

#include "httplib.h"
#include "SOVITS_windows_ONNX_Infer.h"
#include "onnxruntime/providers.h"
#include "dx_utils.h"
#include "audio_utils.h"
#include "f0_utils.h"
#include "spdlog/spdlog.h"
#include <string.h>
#include <numeric>
#include <chrono>
#include <cmath>
#include "AudioFile.h"
#include "samplerate.h"
#include "world/dio.h"
#include "world/stonemask.h"
#include "json/json.h"
#include "utils.h"

using namespace std::chrono;
using namespace std;
using namespace httplib;

/*全局唯一*/
const OrtApi* g_ort = NULL;
httplib::Server svr;
const int iNumberOfChanel = 1;
const size_t DATA_CHUNK_SIZE = 4;
string sJsonConfigFileName = "config.json";

/*模型配置*/
int iHubertDim = 256;
int iHubertInputSampleRate = 16000;
int iVITSOutputSampleRate = 32000;
int iFinalOutSampleRate = 44100;
wchar_t sHuBERTONNXFileName[1024];
wchar_t sVITSONNXFileName[1024];
string sHubertInputTensorName = "source";
string sHubertOutputTensorName = "embed";

/*性能评估配置*/
bool bBenchmark = false;
int iSkipWarmupStep = 5;
int iBenchMarkStep = 100;

/*本地离线测试配置*/
bool bLocalTransTest = false;
string sLocalTestInputAudioFileName = "test30.wav";
string sLocalTestOutputAudioFileName = "test30_output.wav";
int iHTTPListenPort = 6842;

/*
* 载入配置文件
*/
void func_load_config_file() {
	std::ifstream t_pc_file(sJsonConfigFileName, std::ios::binary);
	std::stringstream buffer_pc_file;
	buffer_pc_file << t_pc_file.rdbuf();

	Json::Value jsonRoot;
	buffer_pc_file >> jsonRoot;

	iHubertDim = jsonRoot["iHubertDim"].asInt();
	iHubertInputSampleRate = jsonRoot["iHubertInputSampleRate"].asInt();
	iVITSOutputSampleRate = jsonRoot["iVITSOutputSampleRate"].asInt();
	iFinalOutSampleRate = jsonRoot["iFinalOutSampleRate"].asInt();
	mbstowcs(sHuBERTONNXFileName, jsonRoot["sHuBERTONNXFileName"].asString().c_str(), 1024);
	mbstowcs(sVITSONNXFileName, jsonRoot["sVITSONNXFileName"].asString().c_str(), 1024);
	sHubertInputTensorName = jsonRoot["sHubertInputTensorName"].asString();
	sHubertOutputTensorName = jsonRoot["sHubertOutputTensorName"].asString();

	bBenchmark = jsonRoot["bBenchmark"].asBool();
	iSkipWarmupStep = jsonRoot["iSkipWarmupStep"].asInt();
	iBenchMarkStep = jsonRoot["iBenchMarkStep"].asInt();

	bLocalTransTest = jsonRoot["bLocalTransTest"].asBool();
	sLocalTestInputAudioFileName = jsonRoot["sLocalTestInputAudioFileName"].asString();
	sLocalTestOutputAudioFileName = jsonRoot["sLocalTestOutputAudioFileName"].asString();
	iHTTPListenPort = jsonRoot["iHTTPListenPort"].asInt();
}

/*
* 启用CUDA支持
*/
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

/*
* HUBERT 提取音频特征
*/
void func_hubert_get_embed(
	Ort::Session& session,
	OrtMemoryInfo* memory_info,
	Ort::RunOptions& runOptions,
	std::vector<float> fSamples,
	std::vector<Ort::Value>* returnList
) {

	// HuBERT
	//  source = {"source":np.expand_dims(np.expand_dims(wav16,0),0)}
	//	units = np.array(hubertsession.run(['embed'], source)[0])

	std::array<int64_t, 3> iTensorSourceShape{ 1, 1, fSamples.size() };

	std::vector<Ort::Value> huberInputList;
	huberInputList.emplace_back(
		Ort::Value::CreateTensor<float>(memory_info, fSamples.data(), fSamples.size(), iTensorSourceShape.data(), iTensorSourceShape.size())
	);

	const char* HuBERTInputNames[] = { sHubertInputTensorName.c_str()};
	const char* HuBERTOutputNames[] = { sHubertOutputTensorName.c_str() };
	*returnList = session.Run(runOptions, HuBERTInputNames, huberInputList.data(), huberInputList.size(), HuBERTOutputNames, 1);
}

/*
* VITS 生成音频
*/
void func_vits_get_audio(
	Ort::Session& session,
	OrtMemoryInfo* memory_info,
	Ort::RunOptions& runOptions,
	std::vector<float> fHiddentUnit,
	std::vector<int64_t> iLength,
	std::vector<int64_t> iPitch,
	std::vector<int64_t> iSid,
	std::vector<Ort::Value>* returnList
) {

	/*
	test_hidden_unit = torch.rand(1, 50, 256)
	test_lengths = torch.LongTensor([50])
	test_pitch = (torch.rand(1, 50) * 128).long()
	test_sid = torch.LongTensor([0])
	HuBERTInputNames = ["hidden_unit", "lengths", "pitch", "sid"]
	HuBERTOutputNames = ["audio", ]
	*/

	int iUnitNum = fHiddentUnit.size() / iHubertDim;
	std::array<int64_t, 3> iHiddentUnitShape{ 1, iUnitNum, iHubertDim };
	std::array<int64_t, 1> iLengthShape{ 1 };
	std::array<int64_t, 2> iPitchShape{ 1, iUnitNum };
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
	*returnList = session.Run(runOptions, VITSInputNames, VITSInputValues.data(), VITSInputValues.size(), VITSoutputNames, 1);
}


/*
* HUBERT 测速
*/
std::vector<Ort::Value> func_hubert_benchmark(
	Ort::Session& session,
	OrtMemoryInfo* memory_info,
	Ort::RunOptions& runOptions
) {
	std::vector<long long> fUseTimeList;
	long long  tStart;
	long long tUseTime;
	long fAvgUseTime;

	std::vector<float> fHubertTestInput(iHubertInputSampleRate);
	fUseTimeList.clear();
	float* fEmbedData;
	int iEmbedSize;

	std::vector<Ort::Value> returnList;
	for (int i = 0; i < iBenchMarkStep; i++) {
		tStart = func_get_timestamp();
		func_hubert_get_embed(
			session,
			memory_info,
			runOptions,
			fHubertTestInput,
			&returnList);
		tUseTime = func_get_timestamp() - tStart;
		fUseTimeList.push_back(tUseTime);
		LOG_INFO("ONNX 1s音频单次推理耗时:%lldms", tUseTime);
	}
	fAvgUseTime = std::accumulate(fUseTimeList.begin() + iSkipWarmupStep, fUseTimeList.end(), 0) / (fUseTimeList.size() - iSkipWarmupStep);
	LOG_INFO("ONNX 1s音频平均耗时:%lldms", fAvgUseTime);
	return returnList;
}

/*
* VITS 测速
*/
std::vector<Ort::Value> func_vits_benchmark(
	Ort::Session& session,
	OrtMemoryInfo* memory_info,
	Ort::RunOptions& runOptions
) {
	std::vector<long long> fUseTimeList;
	long long  tStart;
	long long tUseTime;
	long fAvgUseTime;

	int iHiddentUnitNum = 50;
	std::vector<float> fTestHiddentUnit(iHiddentUnitNum * iHubertDim);
	std::vector<int64_t> iTestLength(1);
	iTestLength.push_back(iHiddentUnitNum);
	std::vector<int64_t> iTestPitch(iHiddentUnitNum);
	memset(iTestPitch.data(), 1, iHiddentUnitNum);
	std::vector<int64_t> iTestSid(1);
	iTestSid.push_back(0);

	std::vector<Ort::Value> returnList;
	fUseTimeList.clear();
	for (int i = 0; i < iBenchMarkStep; i++) {
		tStart = func_get_timestamp();
		func_vits_get_audio(
			session,
			memory_info,
			runOptions,
			fTestHiddentUnit,
			iTestLength,
			iTestPitch,
			iTestSid,
			&returnList);
		tUseTime = func_get_timestamp() - tStart;
		fUseTimeList.push_back(tUseTime);
		LOG_INFO("VITS 1s音频单次推理耗时:%lldms", tUseTime);
	}
	fAvgUseTime = std::accumulate(fUseTimeList.begin() + iSkipWarmupStep, fUseTimeList.end(), 0) / (fUseTimeList.size() - iSkipWarmupStep);
	LOG_INFO("VITS 1s音频平均耗时:%ldms", fAvgUseTime);
	return returnList;
}

int64_t func_bin_op_dot(int64_t a, int64_t b) {
	return a * b;
}

/*
* 连乘，用于获取shape对于的数组大小
*/
int64_t func_get_shape_elements_size(std::vector<int64_t> shape) {
	return std::reduce(shape.begin(), shape.end(), 1, func_bin_op_dot);
}

int main()
{
	LOG_INFO("程序启动!");
	LOG_INFO("读取配置文件...");
	func_load_config_file();

	func_check_dx_device();
	int ret = 0;

	g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "SOVITS_windows_ONNX_Infer" };
	Ort::RunOptions runOptions;
	OrtMemoryInfo* memory_info;

	auto ortApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	ortApi->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
	Ort::SessionOptions sessionOptions;

	LOG_INFO("启用CUDA...");
	ret = enable_cuda(sessionOptions);
	if (ret) {
		LOG_INFO("CUDA不可用");
	}
	else {
		LOG_INFO("CUDA启用成功");
	}

	LOG_INFO("载入HuBERT ONNX模型...");
	Ort::Session hubertORTSession{ env, sHuBERTONNXFileName, sessionOptions};
	LOG_INFO("载入完成...");

	LOG_INFO("载入VITS ONNX模型...");
	Ort::Session VITSORTSession{ env, sVITSONNXFileName, sessionOptions };
	LOG_INFO("载入完成...");

	if (bBenchmark) {
		LOG_INFO("跑分...");
		// HUBERT benchmark
		std::vector<Ort::Value> hubertReturnTensor = func_hubert_benchmark(
			hubertORTSession,
			memory_info,
			runOptions
		);

		// VITS benchmark
		func_vits_benchmark(
			VITSORTSession,
			memory_info,
			runOptions
		);
	}

	if (bLocalTransTest) {
		LOG_INFO("进行本地离线处理测试...");
		// 从音频文件读取数据
		AudioFile<double> tmpAudioFile;
		tmpAudioFile.load(sLocalTestInputAudioFileName);
		int sampleRate = tmpAudioFile.getSampleRate();
		int bitDepth = tmpAudioFile.getBitDepth();

		int numSamples = tmpAudioFile.getNumSamplesPerChannel();
		double lengthInSeconds = tmpAudioFile.getLengthInSeconds();

		int numChannels = tmpAudioFile.getNumChannels();
		bool isMono = tmpAudioFile.isMono();
		bool isStereo = tmpAudioFile.isStereo();

		LOG_INFO("音频长度：%fs", lengthInSeconds);

		long long tStart = func_get_timestamp();
		// 重采样
		float* fReSampleInBuffer = (float*)malloc(numSamples * sizeof(float));
		float* fReSampleOutBuffer = fReSampleInBuffer;
		int iResampleNumbers = numSamples;
		for (int i = 0; i < numSamples; i++) {
			fReSampleInBuffer[i] = tmpAudioFile.samples[0][i];
		}
		if (sampleRate != iHubertInputSampleRate) {
			double fScaleRate = 1.f * iHubertInputSampleRate / sampleRate;
			iResampleNumbers = fScaleRate * numSamples;
			fReSampleOutBuffer = (float*)(std::malloc(sizeof(float) * (iResampleNumbers + 128)));
			func_audio_resample(fReSampleInBuffer, fReSampleOutBuffer, fScaleRate, numSamples, iResampleNumbers);
		}
		long long tResampleDone = func_get_timestamp();
		LOG_INFO("重采样耗时:%lldms", tResampleDone - tStart);

		// 进行HUBERT推理
		std::vector<float> fHubertTestInput(iResampleNumbers);
		for (int i = 0; i < iResampleNumbers; i++) {
			fHubertTestInput[i] = (float)(fReSampleOutBuffer[i]);
		}
		std::vector<Ort::Value> hubertReturnTensor;
		func_hubert_get_embed(
			hubertORTSession,
			memory_info,
			runOptions,
			fHubertTestInput,
			&hubertReturnTensor);
		auto embedValue = hubertReturnTensor.data();
		int iEmbedTensorSize = func_get_shape_elements_size(embedValue->GetTensorTypeAndShapeInfo().GetShape());

		long long tHubertDone = func_get_timestamp();
		LOG_INFO("HUBERT推理耗时:%lldms", tHubertDone - tResampleDone);

		// 进行VITS推理
		float* fEmbedData = embedValue->GetTensorMutableData<float>();
		std::vector<float> fEmbed(fEmbedData, fEmbedData + iEmbedTensorSize);
		int iHiddentUnitNum = iEmbedTensorSize / iHubertDim;
		std::vector<int64_t> iTestLength(1);
		iTestLength[0] = iHiddentUnitNum;
		std::vector<int64_t> iTestPitch(iHiddentUnitNum);
		int iPitchTrans = 10;
		//std::fill(iTestPitch.begin(), iTestPitch.end(), iPitchTrans);
		// 将音频数据转为double，用于获取F0
		std::vector<double> dAudioSamples(fHubertTestInput.begin(), fHubertTestInput.end());
		//std::transform(fHubertTestInput.begin(), fHubertTestInput.end(), dAudioSamples.begin(), [](float i) {return (double)i; });
		iTestPitch = func_prepare_f0(dAudioSamples, sampleRate, iHiddentUnitNum, iPitchTrans);
		std::vector<int64_t> iTestSid(1);
		iTestSid[0] = 0;

		std::vector<Ort::Value> vitsReturnTensor;
		func_vits_get_audio(
			VITSORTSession,
			memory_info,
			runOptions,
			fEmbed,
			iTestLength,
			iTestPitch,
			iTestSid,
			&vitsReturnTensor);

		float* fVITSAudio = vitsReturnTensor.data()->GetTensorMutableData<float>();
		int iVITSAudioSize = func_get_shape_elements_size(vitsReturnTensor.data()->GetTensorTypeAndShapeInfo().GetShape());
		std::vector<double> vVITSAudio(fVITSAudio, fVITSAudio + iVITSAudioSize);

		long long tVITSDone = func_get_timestamp();
		LOG_INFO("VITS推理耗时:%lldms", tVITSDone - tHubertDone);
		LOG_INFO("pipeline总耗时:%lldms", tVITSDone - tStart);

		// 写数据到音频文件

		AudioFile<double>::AudioBuffer audioBuffer;
		audioBuffer.resize(iNumberOfChanel);
		audioBuffer[0].resize(iVITSAudioSize);
		for (int i = 0; i < iVITSAudioSize; i++) {
			audioBuffer[0][i] = fVITSAudio[i];
		}

		AudioFile<double> audioFile;
		audioFile.shouldLogErrorsToConsole(true);
		audioFile.setAudioBuffer(audioBuffer);
		audioFile.setAudioBufferSize(iNumberOfChanel, iVITSAudioSize);
		audioFile.setBitDepth(24);
		audioFile.setSampleRate(iVITSOutputSampleRate);
		// 保存到文件
		audioFile.save(sLocalTestOutputAudioFileName);
		// 保存音频文件到内存
		//std::vector<uint8_t> vModelInputMemoryBuffer = std::vector<uint8_t>(0);
		//audioFile.saveToWaveMemory(&vModelInputMemoryBuffer);
	}


	// HTTP SERVER
	LOG_INFO("启动HTTP服务");
	svr.set_logger([](const auto& req, const auto& res) {
		//LOG_INFO(http_log(req, res).c_str());
		});
	svr.Get("/", [](const Request& req, Response& res) {
		res.set_content("SOVITS HTTP推理服务启动成功！", "text/plain");
		});

	svr.Post("/voiceChangeModel", [&](const auto& req, auto& res) {
		LOG_INFO("======开始处理======");
		bool ret;
		int iPitchChange = 0;
		int iSampleRate = 0;
		int iSpeakerId = 0;
		ret = req.has_file("fPitchChange");
		if (ret) {
			iPitchChange = stoi(req.get_file_value("fPitchChange").content);
		}
		ret = req.has_file("sampleRate");
		if (ret) {
			iSampleRate = stoi(req.get_file_value("sampleRate").content);
		}
		ret = req.has_file("sSpeakId");
		if (ret) {
			iSpeakerId = stoi(req.get_file_value("sSpeakId").content);
		}

		string sample = req.get_file_value("sample").content;
		std::vector<uint8_t> vSample(sample.length());
		for (int i = 0; i < sample.length(); i++) {
			vSample[i] = sample[i];
		}

		AudioFile<double> tmpAudioFile;
		tmpAudioFile.loadFromMemory(vSample);
		int sampleRate = tmpAudioFile.getSampleRate();
		int bitDepth = tmpAudioFile.getBitDepth();

		int numSamples = tmpAudioFile.getNumSamplesPerChannel();
		double lengthInSeconds = tmpAudioFile.getLengthInSeconds();

		int numChannels = tmpAudioFile.getNumChannels();
		bool isMono = tmpAudioFile.isMono();
		bool isStereo = tmpAudioFile.isStereo();

		LOG_INFO("音频长度：%fs", lengthInSeconds);

		long long tStart = func_get_timestamp();
		// 重采样
		float* fReSampleInBuffer = (float*)malloc(numSamples * sizeof(float));
		float* fReSampleOutBuffer = fReSampleInBuffer;
		int iResampleNumbers = numSamples;
		for (int i = 0; i < numSamples; i++) {
			fReSampleInBuffer[i] = tmpAudioFile.samples[0][i];
		}
		if (sampleRate != iHubertInputSampleRate) {
			double fScaleRate = 1.f * iHubertInputSampleRate / sampleRate;
			iResampleNumbers = fScaleRate * numSamples;
			fReSampleOutBuffer = (float*)(std::malloc(sizeof(float) * (iResampleNumbers + 128)));
			func_audio_resample(fReSampleInBuffer, fReSampleOutBuffer, fScaleRate, numSamples, iResampleNumbers);
		}
		long long tResampleDone = func_get_timestamp();
		LOG_INFO("重采样耗时:%lldms", tResampleDone - tStart);

		// 进行HUBERT推理
		std::vector<float> fHubertTestInput(iResampleNumbers);
		for (int i = 0; i < iResampleNumbers; i++) {
			fHubertTestInput[i] = (float)(fReSampleOutBuffer[i]);
		}
		std::vector<Ort::Value> hubertReturnTensor;
		func_hubert_get_embed(
			hubertORTSession,
			memory_info,
			runOptions,
			fHubertTestInput,
			&hubertReturnTensor);
		auto embedValue = hubertReturnTensor.data();
		int iEmbedTensorSize = func_get_shape_elements_size(embedValue->GetTensorTypeAndShapeInfo().GetShape());

		long long tHubertDone = func_get_timestamp();
		LOG_INFO("HUBERT推理耗时:%lldms", tHubertDone - tResampleDone);

		// 进行VITS推理
		float* fEmbedData = embedValue->GetTensorMutableData<float>();
		std::vector<float> fEmbed(fEmbedData, fEmbedData + iEmbedTensorSize);
		int iHiddentUnitNum = iEmbedTensorSize / iHubertDim;
		std::vector<int64_t> iTestLength(1);
		iTestLength[0] = iHiddentUnitNum;
		std::vector<int64_t> iTestPitch(iHiddentUnitNum);
		//std::fill(iTestPitch.begin(), iTestPitch.end(), iPitchTrans);
		// 将音频数据转为double，用于获取F0
		std::vector<double> dAudioSamples(fHubertTestInput.begin(), fHubertTestInput.end());
		//std::transform(fHubertTestInput.begin(), fHubertTestInput.end(), dAudioSamples.begin(), [](float i) {return (double)i; });
		iTestPitch = func_prepare_f0(dAudioSamples, sampleRate, iHiddentUnitNum, iPitchChange);
		std::vector<int64_t> iTestSid(1);
		iTestSid[0] = iSpeakerId;

		std::vector<Ort::Value> vitsReturnTensor;
		func_vits_get_audio(
			VITSORTSession,
			memory_info,
			runOptions,
			fEmbed,
			iTestLength,
			iTestPitch,
			iTestSid,
			&vitsReturnTensor);

		float* fVITSAudio = vitsReturnTensor.data()->GetTensorMutableData<float>();
		int iVITSAudioSize = func_get_shape_elements_size(vitsReturnTensor.data()->GetTensorTypeAndShapeInfo().GetShape());
		std::vector<double> vVITSAudio(fVITSAudio, fVITSAudio + iVITSAudioSize);

		long long tVITSDone = func_get_timestamp();
		LOG_INFO("VITS推理耗时:%lldms", tVITSDone - tHubertDone);
		LOG_INFO("pipeline总耗时:%lldms", tVITSDone - tStart);

		AudioFile<double>::AudioBuffer audioBuffer;
		audioBuffer.resize(iNumberOfChanel);
		audioBuffer[0].resize(iVITSAudioSize);
		for (int i = 0; i < iVITSAudioSize; i++) {
			audioBuffer[0][i] = fVITSAudio[i];
		}

		AudioFile<double> audioFile;
		audioFile.shouldLogErrorsToConsole(true);
		audioFile.setAudioBuffer(audioBuffer);
		audioFile.setAudioBufferSize(iNumberOfChanel, iVITSAudioSize);
		audioFile.setBitDepth(24);
		audioFile.setSampleRate(iVITSOutputSampleRate);
		// 保存音频文件到内存
		std::vector<uint8_t> vModelInputMemoryBuffer;
		audioFile.saveToWaveMemory(&vModelInputMemoryBuffer);

		string returnData(vModelInputMemoryBuffer.data(), vModelInputMemoryBuffer.data() + vModelInputMemoryBuffer.size());
		string contentType = "audio/x-wav";
		res.set_content(returnData, contentType);
		});

	svr.listen("0.0.0.0", iHTTPListenPort);

	LOG_INFO("程序退出!");

	// Get F0

	return 0;
}
