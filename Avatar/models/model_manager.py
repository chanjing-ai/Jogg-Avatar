import os

import torch
import torch.nn as nn

from ..configs.model_config import model_loader_configs
from ..utils.io_utils import (
    hash_state_dict_keys,
    init_weights_on_device,
    load_state_dict,
    smart_load_weights,
    split_state_dict_with_prefix,
)


def load_model_from_single_file(
    state_dict,
    model_names,
    model_classes,
    model_resource,
    torch_dtype,
    device,
    infer,
):
    loaded_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        print(f"    model_name: {model_name} model_class: {model_class.__name__}")
        converter = model_class.state_dict_converter()
        if model_resource == "civitai":
            converted = converter.from_civitai(state_dict)
        elif model_resource == "diffusers":
            converted = converter.from_diffusers(state_dict)
        else:
            raise ValueError(f"Unsupported model format: {model_resource}")

        if isinstance(converted, tuple):
            model_state_dict, extra_kwargs = converted
            print(f"        Initializing with extra kwargs: {extra_kwargs}")
        else:
            model_state_dict, extra_kwargs = converted, {}

        model_dtype = torch.float32 if extra_kwargs.get("upcast_to_float32") else torch_dtype
        with init_weights_on_device():
            model = model_class(**extra_kwargs)
        model = model.eval().to_empty(device=device)
        if not infer:
            for parameter in model.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter, gain=0.05)
                else:
                    nn.init.zeros_(parameter)
        model, _, _ = smart_load_weights(model, model_state_dict)
        loaded_names.append(model_name)
        loaded_models.append(model.to(dtype=model_dtype, device=device))
    return loaded_names, loaded_models


class ModelDetectorFromSingleFile:
    def __init__(self, loader_configs):
        self.keys_hash_with_shape = {}
        self.keys_hash = {}
        for metadata in loader_configs:
            self.add_model_metadata(*metadata)

    def add_model_metadata(
        self, keys_hash, keys_hash_with_shape, model_names, model_classes, model_resource
    ):
        metadata = (model_names, model_classes, model_resource)
        self.keys_hash_with_shape[keys_hash_with_shape] = metadata
        if keys_hash is not None:
            self.keys_hash[keys_hash] = metadata

    def _metadata(self, state_dict):
        strict_hash = hash_state_dict_keys(state_dict, with_shape=True)
        if strict_hash in self.keys_hash_with_shape:
            return self.keys_hash_with_shape[strict_hash]
        loose_hash = hash_state_dict_keys(state_dict, with_shape=False)
        return self.keys_hash.get(loose_hash)

    def match(self, file_path="", state_dict=None):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if state_dict is None:
            state_dict = load_state_dict(file_path)
        return self._metadata(state_dict) is not None

    def load(
        self,
        file_path="",
        state_dict=None,
        device="cuda",
        torch_dtype=torch.float16,
        infer=False,
        allowed_model_names=None,
        **kwargs,
    ):
        if state_dict is None:
            state_dict = load_state_dict(file_path)
        metadata = self._metadata(state_dict)
        if metadata is None:
            return [], []
        model_names, model_classes, model_resource = metadata
        if allowed_model_names:
            selected = [
                (name, model_class)
                for name, model_class in zip(model_names, model_classes)
                if name in allowed_model_names
            ]
            model_names = [item[0] for item in selected]
            model_classes = [item[1] for item in selected]
        return load_model_from_single_file(
            state_dict,
            model_names,
            model_classes,
            model_resource,
            torch_dtype,
            device,
            infer,
        )


class ModelDetectorFromSplitSingleFile(ModelDetectorFromSingleFile):
    def _matching_state_dicts(self, state_dict):
        base_match = super().match
        return [
            part
            for part in split_state_dict_with_prefix(state_dict)
            if base_match(state_dict=part)
        ]

    def match(self, file_path="", state_dict=None):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if state_dict is None:
            state_dict = load_state_dict(file_path)
        return bool(self._matching_state_dicts(state_dict))

    def load(self, file_path="", state_dict=None, **kwargs):
        if state_dict is None:
            state_dict = load_state_dict(file_path)
        loaded_names, loaded_models = [], []
        for part in self._matching_state_dicts(state_dict):
            names, models = super().load(state_dict=part, **kwargs)
            loaded_names.extend(names)
            loaded_models.extend(models)
        return loaded_names, loaded_models


class ModelManager:
    def __init__(
        self,
        torch_dtype=torch.float16,
        device="cuda",
        file_path_list=None,
        infer: bool = False,
    ):
        self.torch_dtype = torch_dtype
        self.device = device
        self.model = []
        self.model_path = []
        self.model_name = []
        self.infer = infer
        self.model_detector = [
            ModelDetectorFromSingleFile(model_loader_configs),
            ModelDetectorFromSplitSingleFile(model_loader_configs),
        ]
        self.load_models(file_path_list or [])

    def load_model(self, file_path, model_names=None, device=None, torch_dtype=None):
        print(f"Loading models from: {file_path}")
        device = self.device if device is None else device
        torch_dtype = self.torch_dtype if torch_dtype is None else torch_dtype

        paths = file_path if isinstance(file_path, list) else [file_path]
        missing_paths = [path for path in paths if not os.path.isfile(path)]
        if missing_paths:
            raise FileNotFoundError(f"Model file not found: {missing_paths[0]}")
        state_dict = {}
        for path in paths:
            state_dict.update(load_state_dict(path))

        for detector in self.model_detector:
            if not detector.match(file_path, state_dict):
                continue
            loaded_names, loaded_models = detector.load(
                file_path,
                state_dict,
                device=device,
                torch_dtype=torch_dtype,
                allowed_model_names=model_names,
                infer=self.infer,
            )
            self.model.extend(loaded_models)
            self.model_path.extend([file_path] * len(loaded_models))
            self.model_name.extend(loaded_names)
            print(f"    The following models are loaded: {loaded_names}.")
            return
        raise ValueError(f"Cannot detect model type from: {file_path}")

    def load_models(self, file_path_list, model_names=None, device=None, torch_dtype=None):
        for file_path in file_path_list:
            self.load_model(file_path, model_names, device=device, torch_dtype=torch_dtype)

    def fetch_model(self, model_name, file_path=None, require_model_path=False):
        matches = [
            (model, path)
            for model, path, name in zip(self.model, self.model_path, self.model_name)
            if name == model_name and (file_path is None or path == file_path)
        ]
        if not matches:
            print(f"No {model_name} models available.")
            return None
        if len(matches) > 1:
            print(f"More than one {model_name} model is loaded; using {matches[0][1]}.")
        else:
            print(f"Using {model_name} from {matches[0][1]}.")
        return matches[0] if require_model_path else matches[0][0]

    def to(self, device):
        for model in self.model:
            model.to(device)
