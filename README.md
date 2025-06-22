# Blue_OCR

A OCR Implement with OpenVINO and CPP

因为先前买的新笔记本的CPU是Intel Ultra 9 185H，里面的NPU一直用不上，所以用OpenVINO自己写了个OCR工具，想用上这点算力来做点事情。

原文本监测和识别模型来自[PaddleOCR V5](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/OCR.html#1-ocr)，推理框架使用OpenVINO，因此**暂时只支持Windows+Intel设备（CPU，GPU，NPU）进行推理。**



## 使用说明

执行一次bin目录下的exe文件，可以自由截取一张图片进行推理，结果会用记事本弹出。目前交互方式比较简单，推荐整进环境变量在Terminal调用，或者整进Powertoys去用快捷键调用。

v0.0.2支持选择执行推理的Intel设备：（CPU，GPU，NPU）

**若你想用NPU推理**——请将主目录下的config.ini文件中**[Device]**的内容改为 *det_device = NPU* 即可用NPU执行文本检测部分的推理，*rec_device* 暂时不支持使用NPU。因为文本长度不一，而NPU不支持动态维度下的推理。

**若你的处理器型号较老且有集显**——使用AUTO让OpenVINO后端自动选择推理设备有可能会导致推理失败，因为老型号的集显可能并没有相关的指令集。假如指定GPU设备执行kernel.error，请尝试使用CPU推理。
