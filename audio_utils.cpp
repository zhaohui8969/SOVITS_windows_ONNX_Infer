#include "spdlog/spdlog.h"
#include "utils.h"
#include <stdio.h>
#include <samplerate.h>

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