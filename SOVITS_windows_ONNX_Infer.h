// SOVITS_windows_ONNX_Infer.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。

#pragma once

#include <dxgi.h>
#include <onnxruntime/onnxruntime_cxx_api.h>

#pragma comment(lib, "dxgi.lib")

struct STRUCT_PROJECT_CONFIG {
	wchar_t* sONNXModelFile;
	std::string sONNXInputTensorName;
	std::string sONNXOutputTensorName;
	std::string sBindIpAddress;
	std::string sBindPort;

};

#define LOG_INFO(...)  \
 do{char buf[256]; snprintf(buf, 256,__VA_ARGS__);  spdlog::info(buf);}while(0)

long long func_get_timestamp();
