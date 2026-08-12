import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from Avatar.models import wan_video_dit
from Avatar.utils.io_utils import split_model_paths
from examples.wanvideo import train_wan_avatar as training


class TrainingFlowSmokeTest(unittest.TestCase):
    def test_folded_yaml_model_paths_are_trimmed(self):
        self.assertEqual(
            split_model_paths("first.safetensors, second.safetensors,\n third.safetensors"),
            ["first.safetensors", "second.safetensors", "third.safetensors"],
        )

    def test_tensor_dataset_uses_preprocessed_prompt_embedding(self):
        dataset = object.__new__(training.TensorDataset)
        dataset.path = ["clip.mp4"]
        dataset.frame_interval = 1
        video = torch.randn(3, 121, 8, 8)
        audio = torch.randn(1, 121, 8)
        prompt = {"context": torch.randn(1, 3, 4)}
        dataset.load_frames_using_imageio = mock.Mock(
            return_value=(video, torch.tensor(0), audio, prompt)
        )

        sample = dataset[0]

        self.assertIs(sample["prompt_emb"], prompt)
        self.assertNotIn("prompt_id", sample)
        self.assertEqual(sample["audio_emb"].shape, audio.shape)
        dataset.load_frames_using_imageio.assert_called_once_with(
            "clip.mp4", 1, "clip.mp4.tensors.vae2.2.pth"
        )

    def test_tensor_dataset_fails_instead_of_looping_forever(self):
        dataset = object.__new__(training.TensorDataset)
        dataset.path = ["broken.mp4"]
        dataset.frame_interval = 1
        dataset.load_frames_using_imageio = mock.Mock(return_value=None)

        with self.assertRaisesRegex(RuntimeError, "No valid training sample"):
            dataset[0]

        self.assertEqual(dataset.load_frames_using_imageio.call_count, 2)

    def test_data_process_orchestration_reaches_test(self):
        calls = {}

        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, base_path, metadata_path, **kwargs):
                calls["dataset"] = (base_path, metadata_path, kwargs)

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {"index": index}

        class DummyModel:
            def __init__(self, **kwargs):
                calls["model"] = kwargs

        class DummyTrainer:
            def __init__(self, **kwargs):
                calls["trainer"] = kwargs

            def test(self, model, dataloader):
                calls["test"] = (model, next(iter(dataloader)))

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            output_dir = Path(tmpdir) / "output"
            dataset_dir.mkdir()
            metadata = dataset_dir / "hallo3_videos_clip_all.csv"
            metadata.write_text("file_name,text\nclip.mp4,hello\n", encoding="utf-8")
            args = SimpleNamespace(
                dataset_path=str(dataset_dir),
                output_path=str(output_dir),
                num_frames=17,
                height=128,
                width=128,
                image_encoder_path=None,
                text_encoder_path="text.pth",
                vae_path="vae.pth",
                wav2vec_path="wav2vec",
                tiled=True,
                tile_size_height=8,
                tile_size_width=8,
                tile_stride_height=4,
                tile_stride_width=4,
                dataloader_num_workers=0,
            )
            with (
                mock.patch.object(training, "TextVideoDataset", DummyDataset),
                mock.patch.object(training, "LightningModelForDataProcess", DummyModel),
                mock.patch.object(training.pl, "Trainer", DummyTrainer),
            ):
                training.data_process(args)

            self.assertIn("test", calls)
            self.assertEqual(calls["dataset"][1], str(metadata))
            self.assertEqual(calls["dataset"][2]["num_frames"], 17)
            self.assertEqual(calls["model"]["tile_size"], (8, 8))
            self.assertEqual(calls["trainer"]["accelerator"], "gpu")

    def test_train_orchestration_reaches_fit(self):
        calls = {}

        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, base_path, metadata_path, steps_per_epoch):
                calls["dataset"] = (base_path, metadata_path, steps_per_epoch)

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return {"index": index}

        class DummyModel:
            def __init__(self, **kwargs):
                calls["model"] = kwargs

        class DummyTrainer:
            def __init__(self, **kwargs):
                calls["trainer"] = kwargs

            def fit(self, model, dataloader):
                calls["fit"] = (model, next(iter(dataloader)))

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "dataset"
            output_dir = Path(tmpdir) / "output"
            dataset_dir.mkdir()
            (dataset_dir / "all.csv").write_text("file_name\nclip.mp4\n", encoding="utf-8")
            args = SimpleNamespace(
                dataset_path=str(dataset_dir),
                output_path=str(output_dir),
                steps_per_epoch=1,
                dataloader_num_workers=0,
                dit_path="base.safetensors",
                text_encoder_path="text.pth",
                vae_path="vae.pth",
                learning_rate=1e-4,
                train_architecture="lora",
                lora_rank=4,
                lora_alpha=4.0,
                lora_target_modules="q,k,v,o,ffn.0,ffn.2",
                init_lora_weights="kaiming",
                use_gradient_checkpointing=True,
                use_gradient_checkpointing_offload=False,
                pretrained_lora_path=None,
                use_swanlab=False,
                max_epochs=1,
                training_strategy="auto",
                accumulate_grad_batches=1,
            )
            with (
                mock.patch.object(training, "TensorDataset", DummyDataset),
                mock.patch.object(training, "LightningModelForTrain", DummyModel),
                mock.patch.object(training.pl, "Trainer", DummyTrainer),
            ):
                training.train(args)

            self.assertIn("fit", calls)
            self.assertEqual(calls["dataset"][2], 1)
            self.assertEqual(calls["model"]["lora_rank"], 4)
            self.assertEqual(calls["trainer"]["max_epochs"], 1)
            loss_files = list((output_dir / "loss").glob("*_loss_values.csv"))
            self.assertEqual(len(loss_files), 1)
            self.assertEqual(loss_files[0].read_text(encoding="utf-8"), "train_loss\n")

    def test_tiny_audio_conditioned_dit_forward_and_backward(self):
        previous_args = wan_video_dit.args
        wan_video_dit.args = SimpleNamespace(use_audio=True, sp_size=1)
        try:
            model = wan_video_dit.WanModel(
                dim=32,
                in_dim=33,
                ffn_dim=64,
                out_dim=16,
                text_dim=16,
                freq_dim=16,
                eps=1e-6,
                patch_size=(1, 2, 2),
                num_heads=4,
                num_layers=6,
                has_image_input=False,
                audio_hidden_size=8,
            )
            model.train()
            latents = torch.randn(1, 16, 2, 4, 4)
            image_condition = torch.randn(1, 17, 2, 4, 4)
            audio = torch.randn(1, 5, 10752)
            output = model(
                latents,
                timestep=torch.tensor([10.0]),
                context=torch.randn(1, 3, 16),
                y=image_condition,
                audio_emb=audio,
                lat_h=4,
                lat_w=4,
                use_gradient_checkpointing=True,
            )
            self.assertEqual(output.shape, latents.shape)
            output.square().mean().backward()
            self.assertIsNotNone(model.patch_embedding.weight.grad)
            self.assertIsNotNone(model.audio_proj.proj.weight.grad)
        finally:
            wan_video_dit.args = previous_args


if __name__ == "__main__":
    unittest.main()
