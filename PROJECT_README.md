# MiniMind-V 项目概览

本仓库实现了 MiniMind-V 多模态大模型的最小可行训练与推理流程。本说明用于快速理解代码结构、核心组件之间的协作方式，以及如何在本地运行训练与推理脚本。

## 目录结构与作用

| 目录 / 文件 | 说明 |
| --- | --- |
| `model/` | 语言模型 (`model_minimind.py`) 与视觉语言整合模块 (`model_vlm.py`)，包含 MiniMind 配置、Transformer Block、MoE、CLIP 对齐层等核心实现。 |
| `dataset/` | 数据集定义与预处理逻辑，`lm_dataset.py` 将图文对话样本打包为训练张量。 |
| `trainer/` | 训练脚本与公共工具，包括预训练 (`train_pretrain_vlm.py`)、SFT (`train_sft_vlm.py`) 以及学习率、分布式初始化、断点续训等辅助函数。 |
| `eval_vlm.py` | 命令行推理入口，支持基于本地权重或 Transformers Hub 权重的评测。 |
| `scripts/` | 示例 shell 脚本，用于展示训练/推理命令的拼装方式。 |
| `dataset/` | 存放示例数据与图像资源的位置，可按需替换为自定义数据集。 |
| `PROJECT_README.md` | 当前文档，概述仓库结构与运行方式。 |

## 运行环境准备

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```
2. **准备模型权重**：
   - 将 CLIP 视觉编码器权重放置在 `model/vision_model/clip-vit-base-patch16`。
   - 语言模型权重默认保存在 `out/` 目录，可通过训练脚本自动生成。
3. **数据集组织**：
   - JSONL 文本数据包含字段 `image`（逗号分隔的图像文件名）与 `conversations`（轮次对话列表）。
   - 对应图像文件放在 `dataset/<name>_images` 目录下。

## 训练流程

### 1. 视觉-语言预训练

```bash
python trainer/train_pretrain_vlm.py \
  --data_path ../dataset/pretrain_data.jsonl \
  --images_path ../dataset/pretrain_images \
  --save_dir ../out \
  --epochs 4 \
  --batch_size 16
```

关键参数：
- `--freeze_llm 1`：仅训练视觉对齐层，保持语言模型冻结。
- `--use_wandb`：启用日志记录。
- `--from_resume 1`：自动续训上一次的断点。

### 2. 指令监督微调（SFT）

```bash
python trainer/train_sft_vlm.py \
  --data_path ../dataset/sft_data.jsonl \
  --images_path ../dataset/sft_images \
  --save_dir ../out \
  --epochs 2 \
  --batch_size 4
```

该阶段通常加载预训练权重（`--from_weight pretrain_vlm`），并将 `--freeze_llm` 设为 0 以进行全参更新。

## 推理与评测

```bash
python eval_vlm.py \
  --load_from model \
  --save_dir out \
  --weight sft_vlm \
  --image_dir ./dataset/eval_images \
  --max_new_tokens 256
```

脚本会遍历指定目录中的所有图片，自动构建对话模板并串流输出模型回复。常用调参项包括：
- `--temperature`、`--top_p`：控制采样随机性。
- `--use_moe`：加载 MoE 结构的模型权重。
- `--hidden_size`、`--num_hidden_layers`：根据保存的权重规格选择骨干尺寸。

## 常见问题排查

- **显存不足**：可减小 `--max_new_tokens`、`--batch_size` 或切换到 `bfloat16` 混合精度。
- **找不到 CLIP 权重**：确认 `vision_model_path` 指向的目录包含 `pytorch_model.bin` 与配置文件。
- **多 GPU 训练**：设置环境变量 `RANK`、`WORLD_SIZE` 等后，脚本会自动进入分布式模式，并在断点恢复时自动适配 GPU 数量变化。

通过以上步骤即可快速体验 MiniMind-V 的训练与推理流程，更多细节可查阅源码中的注释与注解。
