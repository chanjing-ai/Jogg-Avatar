import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from Avatar.utils.inference_validation import (
    compute_window_frames,
    validate_media_paths,
    validate_runtime_config,
)


ROOT = Path(__file__).resolve().parents[1]


class InferenceValidationTest(unittest.TestCase):
    def test_tracked_release_contains_no_legacy_brand(self):
        legacy_brand = "".join(("jo", "gg"))
        paths = subprocess.check_output(
            ["git", "ls-files", "-z"], cwd=ROOT
        ).decode().split("\0")
        for relative in filter(None, paths):
            path = ROOT / relative
            if path.suffix in {".png", ".jpg", ".mp4", ".pth", ".safetensors"}:
                continue
            self.assertNotIn(
                legacy_brand,
                path.read_text(encoding="utf-8", errors="ignore").lower(),
                relative,
            )

    def test_window_frames_follow_output_resolution(self):
        self.assertEqual(compute_window_frames(30000, 720, 1280), 33)
        self.assertEqual(compute_window_frames(30000, 1280, 720), 33)
        self.assertEqual(compute_window_frames(320, 128, 128), 17)

    def test_invalid_media_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad_image = root / "bad.png"
            bad_audio = root / "bad.wav"
            bad_image.write_text("not an image", encoding="utf-8")
            bad_audio.write_bytes(b"")

            with self.assertRaisesRegex(ValueError, "Invalid reference image"):
                validate_media_paths(str(bad_image), None, True, False)
            image = root / "image.png"
            Image.new("RGB", (8, 8)).save(image)
            with self.assertRaisesRegex(ValueError, "Invalid driving audio"):
                validate_media_paths(str(image), str(bad_audio), True, True)

    def test_runtime_validation_reports_all_configuration_errors(self):
        args = SimpleNamespace(
            dit_path="missing-1.safetensors, missing-2.safetensors",
            text_encoder_path="missing-text.pth",
            vae_path="missing-vae.pth",
            exp_path="missing-avatar",
            wav2vec_path="missing-wav2vec",
            use_audio=True,
            sp_size=2,
            world_size=1,
            overlap_frame=12,
            num_steps=0,
            max_hw=999,
        )

        with self.assertRaises(ValueError) as context:
            validate_runtime_config(args)

        message = str(context.exception)
        self.assertIn("missing-1.safetensors", message)
        self.assertIn("sp_size (2) must equal torchrun world size (1)", message)
        self.assertIn("overlap_frame must equal 1 + 4*n", message)
        self.assertIn("num_steps must be at least 1", message)
        self.assertIn("max_hw must be 720 or 1280", message)


if __name__ == "__main__":
    unittest.main()
