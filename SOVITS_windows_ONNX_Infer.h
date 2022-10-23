// SOVITS_windows_ONNX_Infer.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。

#pragma once

#include <iostream>
#include <string>

struct STRUCT_PROJECT_CONFIG {
	wchar_t* sONNXModelFile;
	std::string sONNXInputTensorName;
	std::string sONNXOutputTensorName;
	std::string sBindIpAddress;
	std::string sBindPort;

};