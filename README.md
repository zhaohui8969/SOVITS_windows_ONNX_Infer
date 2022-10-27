# SOVITS_windows_ONNX_Infer

分发地址：[https://github.com/zhaohui8969/SOVITS_windows_ONNX_Infer](https://github.com/zhaohui8969/SOVITS_windows_ONNX_Infer)

SOVITS推理后端的Windows版本，无需配置Python、torch、cuda等环境，直接一键运行

# 构建好的程序以及模型下载

由于集成了CUDA环境，体积较大，放在了百度云盘，云盘里同时提供了SOVITS2.2的预训练模型。

链接: [https://pan.baidu.com/s/12hTGAVoxZWByOgFWQN-zuQ?pwd=6666](https://pan.baidu.com/s/12hTGAVoxZWByOgFWQN-zuQ?pwd=6666) 提取码: 6666 复制这段内容后打开百度网盘手机App，操作更方便哦

# 使用方法

## 配置文件说明

程序目录下的`config.json`为配置文件

```json
{
  // 模型配置
  "iHubertDim": 256,
  "iHubertInputSampleRate": 16000,
  "iVITSOutputSampleRate": 32000,
  "iFinalOutSampleRate": 44100,
  // HUBER预训练模型的存放路径
  "sHuBERTONNXFileName": "hubert.onnx",
  // VITS预训练模型的存放路径
  "sVITSONNXFileName": "354_epochs.onnx",
  "sHubertInputTensorName": "source",
  "sHubertOutputTensorName": "embed",

  // 跑分测试
  // 是否开启跑分测试
  "bBenchmark": true,
  "iSkipWarmupStep": 5,
  "iBenchMarkStep": 100,

  // 离线处理本地WAV测试
  "bLocalTransTest": true,
  "sLocalTestInputAudioFileName": "test30.wav",
  "sLocalTestOutputAudioFileName": "test30_output.wav",

  // 程序HTTP服务端口
  "iHTTPListenPort": 6842
}
```

## 跑分测试

当`config.json`中`bBenchmark`为`true`时，程序会在启动后进行短暂的跑分测试，用于确保环境正常。

## 运行HTTP服务程序

直接双击`SOVITS_windows_ONNX_Infer.exe`便可启动程序，可以使用POSTMAN等HTTP工具直接对本地WAV进行处理，亦或使用[VST](https://github.com/zhaohui8969/VST_NetProcess-)插件结合DAW实现实时处理。

# 技术细节

该程序为[sovits_f0_infer](https://github.com/IceKyrin/sovits_f0_infer)的C++ ONNX版本，致力于摆脱pytorch、CUDA环境安装步骤，实现一键启动。

程序处理流程为`接收到音频` -> `重采样到16Khz` -> `HUBERT提取特征` / `World提取F0特征` -> `VITS生成音频`

# pytorch模型转ONNX

可以参考这个代码[onnx_export.py](https://github.com/IceKyrin/sovits_f0_infer/blob/main/onnx_export.py)

# 使用的第三方技术

- [onnxruntime](https://github.com/microsoft/onnxruntime)
- F0计算工具[World](https://github.com/mmorise/World)
- 音频重采样工具[libsamplerate](https://github.com/libsndfile/libsamplerate)
- HTTP服务[cpp-httplib](https://github.com/yhirose/cpp-httplib)
- Wav文件处理[AudioFile](https://github.com/adamstark/AudioFile)
- JSON组件[jsoncpp](https://github.com/open-source-parsers/jsoncpp)
- 日志组件[spdlog](https://github.com/gabime/spdlog)

# 联系方式

QQ: 896919430

QQ群: 588056461

Email: natas_hw@163.com
