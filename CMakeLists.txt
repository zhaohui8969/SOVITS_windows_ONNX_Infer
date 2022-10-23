# CMakeList.txt: ONNX_MINST_Demo 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required(VERSION 3.13)

project ("SOVITS_windows_ONNX_Infer")

# 将源代码添加到此项目的可执行文件。
add_executable (${PROJECT_NAME} "SOVITS_windows_ONNX_Infer.cpp" "SOVITS_windows_ONNX_Infer.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
endif()

target_include_directories(${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include/onnxruntime)

#target_link_options(${PROJECT_NAME} PRIVATE "/SUBSYSTEM:WINDOWS")
target_link_libraries(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/lib/onnxruntime.lib)