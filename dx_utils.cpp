#include <vector>
#include <dxgi.h>
#include "spdlog/spdlog.h"
#include "utils.h"

/*
* ö��DX�豸
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
* ���DX�豸
*/
void func_check_dx_device() {
	auto adapters = EnumerateAdapters();
	LOG_INFO("DX�豸����:%d", adapters.size());
	for (int i = 0; i < adapters.size(); i++) {
		DXGI_ADAPTER_DESC desc;
		adapters[i]->GetDesc(&desc);
		//spdlog::info(desc.Description);
		char tmp[100];
		wcstombs(tmp, desc.Description, 100);
		LOG_INFO("�豸%d:%s", i, tmp);
	}
}
