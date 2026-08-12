import os

import soundfile as sf
from PIL import Image, UnidentifiedImageError

from .io_utils import split_model_paths


def compute_window_frames(max_tokens, height, width):
    """Convert the token budget to a valid Wan video frame count."""
    if max_tokens <= 0 or height <= 0 or width <= 0:
        raise ValueError("max_tokens, height, and width must be positive.")
    frames = int(max_tokens * 16 * 16 * 4 / height / width)
    if frames < 1:
        raise ValueError(
            f"max_tokens={max_tokens} is too small for output size {height}x{width}."
        )
    return frames // 4 * 4 + 1 if frames % 4 else frames - 3


def validate_runtime_config(args):
    """Validate model paths and distributed settings before loading 14B weights."""
    errors = []
    dit_paths = split_model_paths(args.dit_path)
    if not dit_paths:
        errors.append("dit_path does not contain any model files")

    required_files = [
        *dit_paths,
        args.text_encoder_path,
        args.vae_path,
        os.path.join(args.exp_path, "diffusion_pytorch_model.safetensors"),
    ]
    for path in required_files:
        if not os.path.isfile(path):
            errors.append(f"model file not found: {path}")

    if getattr(args, "use_audio", False) and not os.path.isdir(args.wav2vec_path):
        errors.append(f"Wav2Vec directory not found: {args.wav2vec_path}")
    if args.sp_size != args.world_size:
        errors.append(
            f"sp_size ({args.sp_size}) must equal torchrun world size ({args.world_size})"
        )
    if args.overlap_frame < 1 or args.overlap_frame % 4 != 1:
        errors.append("overlap_frame must equal 1 + 4*n")
    if args.num_steps < 1:
        errors.append("num_steps must be at least 1")
    if args.max_hw not in (720, 1280):
        errors.append("max_hw must be 720 or 1280")

    if errors:
        raise ValueError("Invalid inference configuration:\n- " + "\n- ".join(errors))


def validate_media_paths(image_path, audio_path, require_image, require_audio):
    """Decode input headers early so invalid media fails before inference."""
    if require_image and not image_path:
        raise ValueError("A reference image is required via --image_path or the input file.")
    if image_path:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Reference image not found: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Invalid reference image: {image_path}") from exc

    if require_audio and not audio_path:
        raise ValueError("Driving audio is required via --audio_path or the input file.")
    if audio_path:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Driving audio not found: {audio_path}")
        try:
            info = sf.info(audio_path)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Invalid driving audio: {audio_path}") from exc
        if info.frames <= 0 or info.samplerate <= 0:
            raise ValueError(f"Driving audio is empty: {audio_path}")
