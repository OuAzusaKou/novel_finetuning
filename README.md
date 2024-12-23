# GLM 模型微调项目

这个项目包含了对 GLM-1.5B-Chat 模型进行微调的代码，支持全参数微调和 LoRA 微调两种方式。

## 文件结构

- `process_data.py`: 数据预处理脚本
- `train.py`: 全参数微调训练脚本
- `train_lora.py`: LoRA 微调训练脚本
- `inference.py`: 模型推理脚本
- `ds_config.json`: DeepSpeed 配置文件（全参数微调）
- `ds_config_lora.json`: DeepSpeed 配置文件（LoRA 微调）

## 环境要求

- Python 3.10+
- PyTorch
- Transformers
- DeepSpeed
- PEFT (用于 LoRA)

## 使用方法

1. 数据处理： 