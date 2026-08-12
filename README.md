# Jogg-Avatar 14B

[English](README.md) | [简体中文](README_zh.md)

Jogg-Avatar is an audio-driven 720p avatar video generation model based on
Wan2.1-T2V-14B. It takes a reference image, a driving audio track, and a text
prompt, then generates a talking-avatar video with synchronized lip motion.

This repository contains only the Jogg-Avatar 14B image-to-video (I2V) training
and inference pipeline.

https://github.com/user-attachments/assets/becb1b28-890a-4316-9103-1b98411c4f86

## Project Timeline

- 2025-10: released Jogg-Avatar 14B training code
- 2025-11: released Jogg-Avatar 14B inference code
- 2026-08: completed repository cleanup, uv migration, and inference usability fixes

## Installation

The environment is managed by [uv](https://docs.astral.sh/uv/). The lock file
uses Python 3.13, PyTorch 2.8.0, and CUDA 12.8 wheels.

```bash
git clone https://github.com/chanjing-ai/Jogg-Avatar.git
cd Jogg-Avatar

# Inference
uv sync

# Training
uv sync --extra train
```

FlashAttention is optional but strongly recommended for 720p inference:

```bash
uv sync --extra build
uv pip install flash-attn==2.8.3 --no-build-isolation
```

## Model Setup

Download the Wan2.1 base model, Wav2Vec audio encoder, and only the 14B
Jogg-Avatar checkpoint:

```bash
mkdir -p models

# ModelScope is recommended for the large Wan2.1 base model in China.
uvx --from modelscope==1.37.1 modelscope download Wan-AI/Wan2.1-T2V-14B \
  --local_dir models/Wan2.1-T2V-14B

uv run hf download facebook/wav2vec2-base-960h \
  --local-dir models/wav2vec2-base-960h
uv run hf download cicada-ai/jogg-avatar \
  --include "Jogg-Avatar-14B/*" \
  --local-dir models
```

The default config expects:

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

## Inference

Validate the model layout and input media before loading the 14B model:

```bash
uv run python script/inference.py \
  --config configs/inference_smoke.yaml \
  --prompt "A person speaking naturally to the camera" \
  --image_path /path/to/reference.jpg \
  --audio_path /path/to/driving.wav \
  --validate_only
```

Run a fast end-to-end smoke test at 128x128 with one denoising step:

```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --standalone --nproc_per_node=1 \
  script/inference.py \
  --config configs/inference_smoke.yaml \
  --prompt "A person speaking naturally to the camera" \
  --image_path /path/to/reference.jpg \
  --audio_path /path/to/driving.wav \
  --output_dir demo_out/14b-smoke
```

For the tested 720p quality profile, use two GPUs. The process count must equal
`sp_size` in the config:

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run torchrun --standalone --nproc_per_node=2 \
  script/inference.py \
  --config configs/inference_quality.yaml \
  --prompt "A realistic close-up portrait speaking naturally to the camera" \
  --image_path /path/to/reference.jpg \
  --audio_path /path/to/driving.wav \
  --output_dir demo_out/14b-quality
```

The quality profile uses 720x1280, 20 steps, and no TeaCache. On two RTX 4090
GPUs, a one-second sample took about nine minutes and used about 15 GB VRAM per
GPU. Longer audio is generated in 33-frame overlapping windows, so runtime grows
approximately linearly. A single-GPU run can use `configs/inference.yaml`; it is
slower and uses CPU offload by default.

The output directory contains the generated MP4, muxed audio, and prompt.

For batch inference, use `--input_file`. Empty lines and lines beginning with
`#` are ignored; every other line uses:

```text
prompt@@reference_image_path@@driving_audio_path
```

Useful inference controls in `configs/inference.yaml`:

- `guidance_scale`: text guidance strength; 4 to 6 is a practical range.
- `audio_scale`: independent audio guidance. When unset, it follows text guidance.
- `num_steps`: 20 to 50; more steps generally improve quality at higher cost.
- `overlap_frame`: overlap between generated chunks; it must equal `1 + 4*n`.
- `tea_cache_l1_thresh`: set around 0.05 to 0.15 to trade quality for speed.
- `num_persistent_param_in_dit`: reduce this value when GPU memory is limited.
- `max_tokens`: token budget per generation window. Window length is calculated
  from the selected output resolution.

## Training

Install training dependencies first:

```bash
uv sync --extra train
```

The preprocessing metadata file is
`<dataset_path>/hallo3_videos_clip_all.csv`. Training reads
`<dataset_path>/all.csv`. Both files require a `file_name` column; preprocessing
also reads the `text` column and a same-stem `.wav` audio file.

Preprocessing writes `<video>.tensors.vae2.2.pth`; training reads the same cache
for audio and prompt embeddings. An optional `<video>_mouth_info_sm.json` enables
face-centered crop augmentation. Without it, training uses the full frame.

Generate VAE, text, and audio features:

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

Train LoRA and audio-conditioning modules:

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

## Repository Scope

The generic DiffSynth applications, unrelated diffusion models, legacy model
placeholders, and non-Wan examples were removed. The remaining source tree is
limited to the Jogg-Avatar 14B model, training entrypoint, inference entrypoint,
and configuration.

## Acknowledgments

This project builds on [Wan](https://github.com/Wan-Video/Wan2.1) and was
informed by [OmniAvatar](https://github.com/Omni-Avatar/OmniAvatar) and
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).

## License

Released under the [Apache License 2.0](LICENSE). The Wan2.1 base model license
and usage terms also apply.
