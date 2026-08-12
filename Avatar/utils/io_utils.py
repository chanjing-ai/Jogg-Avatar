import hashlib
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager

import imageio
import numpy as np
import soundfile as sf
import torch
from einops import rearrange
from safetensors import safe_open

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def split_model_paths(value):
    """Split comma-separated model paths produced by CLI or folded YAML."""
    return [path.strip() for path in value.split(",") if path.strip()]


@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers=False):
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(
                module._parameters[name].to(device), **kwargs
            )

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    constructors = {}
    if include_buffers:
        constructors = {
            name: getattr(torch, name) for name in ["empty", "zeros", "ones", "full"]
        }

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for name in constructors:
            setattr(torch, name, patch_tensor_constructor(getattr(torch, name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for name, old_constructor in constructors.items():
            setattr(torch, name, old_constructor)


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as file:
        for key in file.keys():
            value = file.get_tensor(key)
            state_dict[key] = value.to(torch_dtype) if torch_dtype is not None else value
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                state_dict[key] = value.to(torch_dtype)
    return state_dict


def smart_load_weights(model, checkpoint):
    compatible = {}
    for name, parameter in model.state_dict().items():
        if name not in checkpoint:
            continue
        checkpoint_parameter = checkpoint[name]
        if parameter.shape == checkpoint_parameter.shape:
            compatible[name] = checkpoint_parameter
        elif all(current >= saved for current, saved in zip(parameter.shape, checkpoint_parameter.shape)):
            expanded = parameter.clone()
            slices = tuple(slice(0, size) for size in checkpoint_parameter.shape)
            expanded[slices] = checkpoint_parameter
            compatible[name] = expanded
            print(f"[Expand] {name}: checkpoint {checkpoint_parameter.shape} -> model {parameter.shape}")
        else:
            print(f"[Skip] {name}: checkpoint {checkpoint_parameter.shape} is larger than model {parameter.shape}")
    missing, unexpected = model.load_state_dict(compatible, assign=True, strict=False)
    return model, missing, unexpected


def save_wav(audio, audio_path):
    if isinstance(audio, torch.Tensor):
        audio = audio.float().detach().cpu().numpy()
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    sf.write(audio_path, audio.T, 16000)


def save_video_with_audio(
    video_batch,
    save_path,
    fps=25,
    prompt=None,
    prompt_path=None,
    audio_path=None,
    prefix="result",
):
    """Save generated RGB frames and optionally mux a driving audio track."""
    os.makedirs(save_path, exist_ok=True)
    videos = [video_batch] if isinstance(video_batch, list) else list(video_batch)
    output_paths = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for index, video in enumerate(videos):
            chunks = video if isinstance(video, list) else [video]
            suffix = f"_{index:03d}" if len(videos) > 1 else ""
            output_path = os.path.join(save_path, f"{prefix}{suffix}.mp4")
            silent_path = os.path.join(temp_dir, f"{prefix}{suffix}.mp4")
            with imageio.get_writer(silent_path, fps=fps) as writer:
                for chunk in chunks:
                    frames = chunk[0] if chunk.ndim == 5 else chunk
                    for frame in frames:
                        frame = rearrange(frame, "c h w -> h w c")
                        frame = (frame.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                        writer.append_data(frame)
            if audio_path:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        silent_path,
                        "-i",
                        audio_path,
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-shortest",
                        output_path,
                        "-y",
                    ],
                    check=True,
                )
            else:
                shutil.copy2(silent_path, output_path)
            output_paths.append(output_path)
            print(f"Saved result video to: {output_path}")
    if prompt is not None and prompt_path is not None:
        with open(prompt_path, "w") as file:
            file.write(prompt)
    return output_paths


def hash_state_dict_keys(state_dict, with_shape=True):
    keys = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    return hashlib.md5(keys.encode("UTF-8")).hexdigest()


def split_state_dict_with_prefix(state_dict):
    groups = {}
    for key in sorted(key for key in state_dict if isinstance(key, str)):
        prefix = key if "." not in key else key.split(".")[0]
        groups.setdefault(prefix, []).append(key)
    return [{key: state_dict[key] for key in keys} for keys in groups.values()]


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, torch.Tensor):
            if with_shape:
                shape = "_".join(map(str, value.shape))
                keys.append(f"{key}:{shape}")
            keys.append(key)
        elif isinstance(value, dict):
            nested = convert_state_dict_keys_to_single_str(value, with_shape=with_shape)
            keys.append(f"{key}|{nested}")
    return ",".join(sorted(keys))
