#pragma once

#define LOG_INFO(...)  \
 do{char buf[256]; snprintf(buf, 256,__VA_ARGS__);  spdlog::info(buf);}while(0)

/*
* ��ȡʱ���
*/
long long func_get_timestamp();