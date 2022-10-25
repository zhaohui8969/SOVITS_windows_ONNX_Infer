// SOVITS_windows_ONNX_Infer.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。

#pragma once

#include <iostream>
#include <string>
#include <cstdarg>

struct STRUCT_PROJECT_CONFIG {
	wchar_t* sONNXModelFile;
	std::string sONNXInputTensorName;
	std::string sONNXOutputTensorName;
	std::string sBindIpAddress;
	std::string sBindPort;

};

#define LOG_INFO(...)  \
 do{char buf[256]; snprintf(buf, 256,__VA_ARGS__);  spdlog::info(buf);}while(0)