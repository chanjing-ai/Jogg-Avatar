---
license: apache-2.0
pipeline_tag: image-to-video
base_model: Wan-AI/Wan2.1-T2V-14B
tags:
  - audio-driven-video-generation
  - talking-head
  - lip-sync
  - wan2.1
  - image-to-video
---

# Chanjing-Avatar 14B

Chanjing-Avatar 14B is an audio-driven 720p avatar video generation model based
on [Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B). It adds audio
conditioning and LoRA adapters to the Wan video diffusion model.

Source code and complete inference instructions:
[chanjing-ai/Chanjing-Avatar](https://github.com/chanjing-ai/Chanjing-Avatar)

## Chanjing-Avatar Model Family

- [Chanjing-Avatar 14B](https://huggingface.co/cicada-ai/Chanjing-Avatar-14B): 720p image-to-video generation from a reference image and driving audio.
- [Chanjing-Avatar V2V 5B](https://huggingface.co/cicada-ai/Chanjing-Avatar-V2V-5B): video-to-video generation that preserves source motion and regenerates the speaking face.
- [Chanjing-Avatar V2V 1.3B](https://huggingface.co/cicada-ai/Chanjing-Avatar-V2V-1.3B): a lighter video-to-video model for audio-driven face animation.

The checkpoint contains audio modules, input projection, and LoRA adapters in
BF16. The Wan2.1 base model and Wav2Vec audio encoder are required separately.

```text
Chanjing-Avatar-14B/
|-- config.json
`-- diffusion_pytorch_model.safetensors
```

```bash
hf download cicada-ai/Chanjing-Avatar-14B \
  --local-dir models/Chanjing-Avatar-14B
```

Users are responsible for obtaining consent for source images and voices and
for clearly disclosing synthetic media.

