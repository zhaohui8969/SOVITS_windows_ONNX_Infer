# CMakeList.txt: ONNX_MINST_Demo 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required(VERSION 3.13)

project ("SOVITS_windows_ONNX_Infer")

FILE(GLOB WORLD_SRCS "world_src/*.cpp")
FILE(GLOB WORLD_HEADERS "world_src/world/*.h")
# 将源代码添加到此项目的可执行文件。
add_executable (${PROJECT_NAME} "SOVITS_windows_ONNX_Infer.cpp" "SOVITS_windows_ONNX_Infer.h" ${WORLD_SRCS} ${WORLD_HEADERS})

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/world_src")

#target_link_options(${PROJECT_NAME} PRIVATE "/SUBSYSTEM:WINDOWS")
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/lib/onnxruntime.lib)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/lib/onnxruntime_providers_cuda.lib)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/lib/onnxruntime_providers_shared.lib)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/lib/samplerate.lib)

set(fmt_DIR D:/codes/vcpkg/packages/fmt_x64-windows/share/fmt)
find_package(fmt CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt-header-only)

set(spdlog_DIR D:/codes/vcpkg/packages/spdlog_x64-windows/share/spdlog)
find_package(spdlog CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog spdlog::spdlog_header_only)
