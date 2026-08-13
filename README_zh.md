# Jogg-Avatar 14B

[English](README.md) | [简体中文](README_zh.md)

Jogg-Avatar 是一个基于 Wan2.1-T2V-14B 的音频驱动 720p 数字人视频生成模型。
输入一张参考图、驱动音频和文本提示词，即可生成口型与音频同步的数字人视频。

本仓库只包含 Jogg-Avatar 14B 图生视频（I2V）的训练与推理代码。

https://github.com/user-attachments/assets/560b72a5-8384-4892-a293-0766acbcf106

## 在我们的产品中体验

<p align="center">
  Jogg-Avatar 所采用的数字人生成技术也已应用于我们的商业产品。无需本地部署模型，
  即可体验 AI 数字人视频创作。
</p>

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://www.chanjing.cc/home/">
        <img src="assets/brand/chanjing-logo.png" alt="蝉镜 AI" height="52">
      </a>
      <br>
      <strong>蝉镜 AI</strong>
      <br>
      <sub>面向国内创作者与企业的 AI 视频创作平台</sub>
      <br><br>
      <a href="https://www.chanjing.cc/home/"><strong>访问蝉镜</strong></a>
    </td>
    <td align="center" width="50%">
      <a href="https://www.jogg.ai/">
        <picture>
          <source media="(prefers-color-scheme: dark)" srcset="assets/brand/joggai-logo-dark.png">
          <img src="assets/brand/joggai-logo.png" alt="JoggAI" height="52">
        </picture>
      </a>
      <br>
      <sub>面向全球营销与内容团队的 AI 视频创作平台</sub>
      <br><br>
      <a href="https://www.jogg.ai/"><strong>访问 JoggAI</strong></a>
    </td>
  </tr>
</table>

## 项目时间线

- 2025-10：发布 Jogg-Avatar 14B 训练代码
- 2025-11：发布 Jogg-Avatar 14B 推理代码
- 2026-08：完成仓库清理、uv 环境迁移与推理易用性修复

## 安装

项目使用 [uv](https://docs.astral.sh/uv/) 管理环境。锁文件基于 Python 3.13、
PyTorch 2.8.0 和 CUDA 12.8 wheel。

```bash
git clone https://github.com/chanjing-ai/Jogg-Avatar.git
cd Jogg-Avatar

# 推理环境
uv sync

# 训练环境
uv sync --extra train
```

FlashAttention 为可选依赖，720p 推理强烈建议安装：

```bash
uv sync --extra build
uv pip install flash-attn==2.8.3 --no-build-isolation
```

## 模型准备

下载 Wan2.1 基座模型、Wav2Vec 音频编码器，并且只下载 Jogg-Avatar 14B 权重：

```bash
mkdir -p models

# 国内建议通过 ModelScope 下载体积较大的 Wan2.1 基座模型。
uvx --from modelscope==1.37.1 modelscope download Wan-AI/Wan2.1-T2V-14B \
  --local_dir models/Wan2.1-T2V-14B

uv run hf download facebook/wav2vec2-base-960h \
  --local-dir models/wav2vec2-base-960h
uv run hf download cicada-ai/jogg-avatar \
  --include "Jogg-Avatar-14B/*" \
  --local-dir models
```

默认配置使用以下目录结构：

```text
models/
├── Wan2.1-T2V-14B/
│   ├── diffusion_pytorch_model-00001-of-00006.safetensors
│   ├── ...
│   ├── diffusion_pytorch_model-00006-of-00006.safetensors
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── Wan2.1_VAE.pth
├── Jogg-Avatar-14B/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── wav2vec2-base-960h/
```

## 推理

加载 14B 模型前，先检查模型目录和输入素材：

```bash
uv run python script/inference.py \
  --config configs/inference_smoke.yaml \
  --prompt "一个人自然地面对镜头说话" \
  --image_path /path/to/reference.jpg \
  --audio_path /path/to/driving.wav \
  --validate_only
```

快速功能验证使用 128x128、1 个去噪步：

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nproc_per_node=1 \
  script/inference.py \
  --config configs/inference_smoke.yaml \
  --prompt "一个人自然地面对镜头说话" \
  --image_path /path/to/reference.jpg \
  --audio_path /path/to/driving.wav \
  --output_dir demo_out/14b-smoke
```

已实测的 720p 画质配置使用两张 GPU。进程数必须与配置中的 `sp_size`
保持一致：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  script/inference.py \
  --config configs/inference_quality.yaml \
  --prompt "写实近景人像，自然地面对镜头说话" \
  --image_path /path/to/reference.jpg \
  --audio_path /path/to/driving.wav \
  --output_dir demo_out/14b-quality
```

画质配置使用 720x1280、20 步、不启用 TeaCache。在两张 RTX 4090 上，1 秒
样本约需 9 分钟，每张卡约使用 15 GB 显存。长音频会按 33 帧重叠窗口分段生成，
耗时近似线性增长。单卡可使用 `configs/inference.yaml`，默认通过 CPU offload 降低显存占用。

输出目录会包含生成的 MP4、合并后的音轨和本次使用的提示词。

批量推理时使用 `--input_file`。空行和以 `#` 开头的行会被忽略，其他行格式为：

```text
提示词@@参考图路径@@驱动音频路径
```

`configs/inference.yaml` 中常用的推理参数：

- `guidance_scale`：文本引导强度，建议范围为 4 到 6。
- `audio_scale`：独立音频引导强度；留空时跟随文本引导强度。
- `num_steps`：建议 20 到 50，步数越多通常质量越高、耗时越长。
- `overlap_frame`：分段生成的重叠帧数，必须满足 `1 + 4*n`。
- `tea_cache_l1_thresh`：可设为 0.05 到 0.15，在速度与质量之间权衡。
- `num_persistent_param_in_dit`：显存不足时可减小该值。
- `max_tokens`：每个生成窗口的 token 预算，窗口帧数会根据实际输出分辨率计算。

## 训练

先安装训练依赖：

```bash
uv sync --extra train
```

预处理读取 `<dataset_path>/hallo3_videos_clip_all.csv`，训练读取
`<dataset_path>/all.csv`。两个 CSV 都需要 `file_name` 列；预处理 CSV 还读取
`text` 列，并要求视频存在同名 `.wav` 音频。

预处理会为每个视频生成 `<video>.tensors.vae2.2.pth`，训练直接从同一缓存读取
音频和文本特征。可选的 `<video>_mouth_info_sm.json` 用于人脸居中裁剪增强；没有
该文件时使用完整画面训练。

生成 VAE、文本和音频特征：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python examples/wanvideo/train_wan_avatar.py \
  --task data_process \
  --dataset_path /path/to/dataset \
  --output_path /path/to/preprocess-output \
  --text_encoder_path models/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth \
  --vae_path models/Wan2.1-T2V-14B/Wan2.1_VAE.pth \
  --wav2vec_path models/wav2vec2-base-960h \
  --num_frames 121 \
  --height 720 \
  --width 720
```

训练 LoRA 与音频条件模块：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run python \
  examples/wanvideo/train_wan_avatar.py \
  --task train \
  --dataset_path /path/to/dataset \
  --output_path models/jogg-avatar-14b-train \
  --dit_path models/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,models/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,models/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,models/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,models/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,models/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors \
  --max_epochs 100 \
  --learning_rate 1e-4 \
  --lora_rank 128 \
  --lora_alpha 64 \
  --use_gradient_checkpointing \
  --use_gradient_checkpointing_offload
```

## 致谢

本项目基于 [Wan](https://github.com/Wan-Video/Wan2.1)，并参考了
[OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar) 与
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)。

## 许可证

项目使用 [Apache License 2.0](LICENSE)。Wan2.1 基座模型的许可证与使用条款
同样适用。
