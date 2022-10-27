// SOVITS_windows_ONNX_Infer.cpp: 定义应用程序的入口点。

#include "httplib.h"
#include "SOVITS_windows_ONNX_Infer.h"
#include "onnxruntime/providers.h"
#include "spdlog/spdlog.h"
#include <string.h>
#include <numeric>
#include <chrono>
#include <cmath>
#include "AudioFile.h"
#include "samplerate.h"
#include "world/dio.h"
#include "world/stonemask.h"

using namespace std::chrono;
using namespace std;
using namespace httplib;

const OrtApi* g_ort = NULL;
httplib::Server svr;
const int iHubertDim = 256;
const int iHubertInputSampleRate = 16000;
const int iVITSOutputSampleRate = 32000;
const int iFinalOutSampleRate = 44100;
const int iNumberOfChanel = 1;
const size_t DATA_CHUNK_SIZE = 4;

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

/*
* 枚举DX设备
*/
std::vector <IDXGIAdapter*> EnumerateAdapters()
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

/*
* 检查DX设备
*/
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
* 音频重采样
*/
void func_audio_resample(float* fInBuffer, float* fOutBuffer, double src_ratio, long lInSize, long lOutSize) {
	SRC_DATA data;
	data.src_ratio = src_ratio;
	data.input_frames = lInSize;
	data.output_frames = lOutSize;
	data.data_in = fInBuffer;
	data.data_out = fOutBuffer;
	int error = src_simple(&data, SRC_SINC_FASTEST, 1);

	if (error > 0) {
		char buff[100];
		const char* cError = src_strerror(error);
		snprintf(buff, sizeof(buff), "Resample error%s\n", cError);
		LOG_INFO("重采样错误:%s", buff);
	}
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

	const char* HuBERTInputNames[] = { "source" };
	const char* HuBERTOutputNames[] = { "embed" };
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
* 获取时间戳
*/
long long func_get_timestamp() {
	return (duration_cast<milliseconds>(system_clock::now().time_since_epoch())).count();
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
	int iSkipWarmupStep = 5;
	int iBenchMarkStep = 100;
	long long  tStart;
	long long tUseTime;
	long fAvgUseTime;

	std::vector<float> fHubertTestInput(16000);
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
* DIO F0 extraction algorithm.

		Parameters
		----------
		x : ndarray
			Input waveform signal.
		fs : int
			Sample rate of input signal in Hz.
		f0_floor : float
			Lower F0 limit in Hz.
			Default: 71.0
		f0_ceil : float
			Upper F0 limit in Hz.
			Default: 800.0
		channels_in_octave : float
			Resolution of multiband processing; normally shouldn't be changed.
			Default: 2.0
		frame_period : float
			Period between consecutive frames in milliseconds.
			Default: 5.0
		speed : int
			The F0 estimator may downsample the input signal using this integer factor
			(range [1;12]). The algorithm will then operate on a signal at fs/speed Hz
			to reduce computational complexity, but high values may negatively impact
			accuracy.
			Default: 1 (no downsampling)
		allowed_range : float
			Threshold for voiced/unvoiced decision. Can be any value >= 0, but 0.02 to 0.2
			is a reasonable range. Lower values will cause more frames to be considered
			unvoiced (in the extreme case of `threshold=0`, almost all frames will be unvoiced).
			Default: 0.1

		Returns
		-------
		f0 : ndarray
			Estimated F0 contour.
		temporal_positions : ndarray
			Temporal position of each frame.
*/
void func_get_dio(
	std::vector<double> x,
	int fs,
	float f0_floor,
	float f0_ceil,
	float channels_in_octave,
	float frame_period,
	int speed,
	float allowed_range,
	std::vector<double>* f0,
	std::vector<double>* temporal_positions
) {
	/*
	cdef int x_length = <int>len(x)
	cdef DioOption option
	InitializeDioOption(&option)
	option.channels_in_octave = channels_in_octave
	option.f0_floor = f0_floor
	option.f0_ceil = f0_ceil
	option.frame_period = frame_period
	option.speed = speed
	option.allowed_range = allowed_range
	f0_length = GetSamplesForDIO(fs, x_length, option.frame_period)
	cdef np.ndarray[double, ndim=1, mode="c"] f0 = \
		np.zeros(f0_length, dtype=np.dtype('float64'))
	cdef np.ndarray[double, ndim=1, mode="c"] temporal_positions = \
		np.zeros(f0_length, dtype=np.dtype('float64'))
	Dio(&x[0], x_length, fs, &option, &temporal_positions[0], &f0[0])
	return f0, temporal_positions
	*/
	int x_length = x.size();
	DioOption option;
	InitializeDioOption(&option);
	option.channels_in_octave = channels_in_octave;
	option.f0_floor = f0_floor;
	option.f0_ceil = f0_ceil;
	option.frame_period = frame_period;
	option.speed = speed;
	option.allowed_range = allowed_range;
	int f0_length = GetSamplesForDIO(fs, x_length, option.frame_period);
	f0->resize(f0_length);
	temporal_positions->resize(f0_length);
	Dio(x.data(), x_length, fs, &option, temporal_positions->data(), f0->data());
}


/*
* StoneMask F0 refinement algorithm.

		Parameters
		----------
		x : ndarray
			Input waveform signal.
		f0 : ndarray
			Input F0 contour.
		temporal_positions : ndarray
			Temporal positions of each frame.
		fs : int
			Sample rate of input signal in Hz.

		Returns
		-------
		refined_f0 : ndarray
			Refined F0 contour.
*/
std::vector<double> func_stonemask(
	std::vector<double> x,
	std::vector<double>* f0,
	std::vector<double>* temporal_positions,
	int fs
) {
	/*
	cdef int x_length = <int>len(x)
	cdef int f0_length = <int>len(f0)
	cdef np.ndarray[double, ndim=1, mode="c"] refined_f0 = \
		np.zeros(f0_length, dtype=np.dtype('float64'))
	StoneMask(&x[0], x_length, fs, &temporal_positions[0],
		&f0[0], f0_length, &refined_f0[0])
	return refined_f0
	*/
	int x_length = x.size();
	int f0_length = f0->size();
	std::vector<double>refined_f0(f0_length);
	StoneMask(x.data(), x_length, fs, temporal_positions->data(), f0->data(), f0_length, refined_f0.data());
	return refined_f0;
}

std::vector<int64_t> func_prepare_f0(
	std::vector<double> dSamples,
	int iSamplerate,
	int iHiddentUnitNum,
	int iPitch
) {
	/*
		def transcribe(self, source, sr, length, transform):
			feature_pit = self.feature_input.compute_f0(source, sr)
			feature_pit = feature_pit * 2 ** (transform / 12)
			feature_pit = resize2d_f0(feature_pit, length)
			coarse_pit = self.feature_input.coarse_f0(feature_pit)
			return coarse_pit
	*/

	// why 32000?
	int sr = 32000;
	int iHop = 320;
	float fDefaultF0Floor = 71.f;
	float channels_in_octave = 2.f;
	float frame_period = 1000.f * iHop / sr;
	int speed = 1;
	float allowed_range = 0.1f;
	int iF0Ceil = 800;
	std::vector<double> f0;
	std::vector<double> t;
	func_get_dio(dSamples, sr, fDefaultF0Floor, iF0Ceil, channels_in_octave, frame_period, speed, allowed_range, &f0, &t);
	f0 = func_stonemask(dSamples, &f0, &t, sr);
	std::transform(
		f0.begin(),
		f0.end(),
		f0.begin(),
		[iPitch](double i) {
			return i * pow(2, 1.f * iPitch / 12.f);
		});
	// feature_pit = resize2d_f0(feature_pit, length) ????,Dio的sr修正后是否不需要这步？
	// 计算coarse_f0
	double f0_min = 1127.f * log(1 + 50.f / 700.f);
	double f0_max = 1127.f * log(1 + 1100.f / 700.f);
	/*
	f0_mel = 1127 * np.log(1 + f0 / 700)
	f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
	f0_mel[f0_mel <= 1] = 1
	f0_mel[f0_mel > 255] = 255
	f0_coarse = np.rint(f0_mel).astype(np.int)
	*/
	std::transform(
		f0.begin(),
		f0.end(),
		f0.begin(),
		[f0_min, f0_max](double i) {
			i = 1127 * log(1 + i / 700.f);
			if (i > 0) {
				i = (i - f0_min) * 254.f / (f0_max - f0_min) + 1;
			}
			if (i <= 1) {
				i = 1;
			}
			if (i > 255) {
				i = 255;
			}
			i = rint(i);
			return i;
		});
	std::vector<int64_t> iF0(f0.size());
	for (int i = 0; i < f0.size(); i++) {
		iF0[i] = f0[i];
	}
	return iF0;
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
	int iSkipWarmupStep = 5;
	int iBenchMarkStep = 100;
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

	func_check_dx_device();
	int ret = 0;

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
	Ort::SessionOptions sessionOptions;

	LOG_INFO("启用CUDA...");
	ret = enable_cuda(sessionOptions);
	if (ret) {
		LOG_INFO("CUD不可用");
	}
	else {
		LOG_INFO("CUDA启用成功");
	}

	LOG_INFO("载入HuBERT ONNX模型...");
	Ort::Session hubertORTSession{ env, projectConfig.sONNXModelFile, sessionOptions };
	LOG_INFO("载入完成...");

	LOG_INFO("载入VITS ONNX模型...");
	Ort::Session VITSORTSession{ env, L"121_epochs.onnx", sessionOptions };
	LOG_INFO("载入完成...");

	bool bBenchmark = false;
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

	bool bLocalTransTest = false;
	if (bLocalTransTest) {
		LOG_INFO("进行本地离线处理测试...");
		// 从音频文件读取数据
		AudioFile<double> tmpAudioFile;
		tmpAudioFile.load("test30.wav");
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
		audioFile.save("test30_output.wav");
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

	svr.listen("0.0.0.0", 6842);

	LOG_INFO("程序退出!");

	// Get F0

	return 0;
}
