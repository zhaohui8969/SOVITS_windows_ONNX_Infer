#pragma once

#include <dxgi.h>

#pragma comment(lib, "dxgi.lib")

/*
* 枚举DX设备
*/
std::vector<IDXGIAdapter*> EnumerateAdapters();

/*
* 检查DX设备
*/
void func_check_dx_device();
