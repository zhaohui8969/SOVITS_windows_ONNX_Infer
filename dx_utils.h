#pragma once

#include <dxgi.h>

#pragma comment(lib, "dxgi.lib")

/*
* ö��DX�豸
*/
std::vector<IDXGIAdapter*> EnumerateAdapters();

/*
* ���DX�豸
*/
void func_check_dx_device();
