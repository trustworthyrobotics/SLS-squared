"""Shared loaders and native runtime for the PushT diffusion policy."""

from __future__ import annotations

import json
import math
import sys
import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from safetensors.torch import load_file
from torch import Tensor, nn


ACTION = "action"
OBS_IMAGES = "observation.images"
OBS_STATE = "observation.state"


@dataclass(frozen=True)
class NativePolicyFeature:
    type: str
    shape: tuple[int, ...]


@dataclass
class NativeDiffusionConfig:
    n_obs_steps: int
    input_features: dict[str, NativePolicyFeature]
    output_features: dict[str, NativePolicyFeature]
    device: str
    horizon: int
    n_action_steps: int
    normalization_mapping: dict[str, str]
    vision_backbone: str
    resize_shape: tuple[int, int] | None
    crop_shape: tuple[int, int] | None
    crop_is_random: bool
    pretrained_backbone_weights: str | None
    use_group_norm: bool
    spatial_softmax_num_keypoints: int
    use_separate_rgb_encoder_per_camera: bool
    down_dims: tuple[int, ...]
    kernel_size: int
    n_groups: int
    diffusion_step_embed_dim: int
    use_film_scale_modulation: bool
    noise_scheduler_type: str
    num_train_timesteps: int
    beta_schedule: str
    beta_start: float
    beta_end: float
    prediction_type: str
    clip_sample: bool
    clip_sample_range: float
    num_inference_steps: int | None

    @classmethod
    def from_json(cls, path: Path) -> NativeDiffusionConfig:
        with open(path) as f:
            raw = json.load(f)

        def features(name: str) -> dict[str, NativePolicyFeature]:
            return {
                key: NativePolicyFeature(type=value["type"], shape=tuple(value["shape"]))
                for key, value in raw[name].items()
            }

        return cls(
            n_obs_steps=raw["n_obs_steps"],
            input_features=features("input_features"),
            output_features=features("output_features"),
            device=raw.get("device", "cpu"),
            horizon=raw["horizon"],
            n_action_steps=raw["n_action_steps"],
            normalization_mapping=dict(raw["normalization_mapping"]),
            vision_backbone=raw["vision_backbone"],
            resize_shape=tuple(raw["resize_shape"]) if raw.get("resize_shape") is not None else None,
            crop_shape=tuple(raw["crop_shape"]) if raw.get("crop_shape") is not None else None,
            crop_is_random=raw["crop_is_random"],
            pretrained_backbone_weights=raw["pretrained_backbone_weights"],
            use_group_norm=raw["use_group_norm"],
            spatial_softmax_num_keypoints=raw["spatial_softmax_num_keypoints"],
            use_separate_rgb_encoder_per_camera=raw["use_separate_rgb_encoder_per_camera"],
            down_dims=tuple(raw["down_dims"]),
            kernel_size=raw["kernel_size"],
            n_groups=raw["n_groups"],
            diffusion_step_embed_dim=raw["diffusion_step_embed_dim"],
            use_film_scale_modulation=raw["use_film_scale_modulation"],
            noise_scheduler_type=raw["noise_scheduler_type"],
            num_train_timesteps=raw["num_train_timesteps"],
            beta_schedule=raw["beta_schedule"],
            beta_start=raw["beta_start"],
            beta_end=raw["beta_end"],
            prediction_type=raw["prediction_type"],
            clip_sample=raw["clip_sample"],
            clip_sample_range=raw["clip_sample_range"],
            num_inference_steps=raw["num_inference_steps"],
        )

    @property
    def image_features(self) -> dict[str, NativePolicyFeature]:
        return {key: value for key, value in self.input_features.items() if value.type == "VISUAL"}

    @property
    def robot_state_feature(self) -> NativePolicyFeature:
        return self.input_features[OBS_STATE]

    @property
    def action_feature(self) -> NativePolicyFeature:
        return self.output_features[ACTION]


class NativePolicyPreprocessor:
    def __init__(self, stats_path: Path, device: torch.device, eps: float = 1e-8):
        self.stats = _load_nested_stats(stats_path, device)
        self.device = device
        self.eps = eps

    def __call__(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        image = batch["observation.image"].to(self.device, dtype=torch.float32)
        state = batch[OBS_STATE].to(self.device, dtype=torch.float32)
        return {
            "observation.image": self._mean_std(image, "observation.image"),
            OBS_STATE: self._min_max(state, OBS_STATE),
        }

    def _mean_std(self, tensor: Tensor, key: str) -> Tensor:
        stats = self.stats[key]
        return (tensor - stats["mean"]) / (stats["std"] + self.eps)

    def _min_max(self, tensor: Tensor, key: str) -> Tensor:
        stats = self.stats[key]
        min_val = stats["min"]
        max_val = stats["max"]
        denom = torch.where(max_val == min_val, torch.full_like(max_val, self.eps), max_val - min_val)
        return 2 * (tensor - min_val) / denom - 1


class NativePolicyPostprocessor:
    def __init__(self, stats_path: Path, eps: float = 1e-8):
        self.stats = _load_nested_stats(stats_path, torch.device("cpu"))
        self.eps = eps

    def __call__(self, action: Tensor) -> Tensor:
        action = action.detach().cpu()
        stats = self.stats[ACTION]
        min_val = stats["min"]
        max_val = stats["max"]
        denom = torch.where(max_val == min_val, torch.full_like(max_val, self.eps), max_val - min_val)
        return (action + 1) / 2 * denom + min_val


def _load_nested_stats(path: Path, device: torch.device) -> dict[str, dict[str, Tensor]]:
    flat = load_file(path, device=str(device))
    nested: dict[str, dict[str, Tensor]] = {}
    for flat_key, value in flat.items():
        key, stat_name = flat_key.rsplit(".", 1)
        nested.setdefault(key, {})[stat_name] = value.to(device=device, dtype=torch.float32)
    return nested


class NativeDiffusionPolicy(nn.Module):
    def __init__(self, config: NativeDiffusionConfig):
        super().__init__()
        self.config = config
        self.diffusion = DiffusionModel(config)
        self._queues: dict[str, deque[Tensor]] = {}
        self.reset()

    def reset(self) -> None:
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
            OBS_IMAGES: deque(maxlen=self.config.n_obs_steps),
        }

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        image = batch["observation.image"]
        state = batch[OBS_STATE]
        images = torch.stack([image], dim=-4)
        self._populate_queue(OBS_STATE, state)
        self._populate_queue(OBS_IMAGES, images)

        if len(self._queues[ACTION]) == 0:
            sequence_batch = {
                OBS_STATE: torch.stack(list(self._queues[OBS_STATE]), dim=1),
                OBS_IMAGES: torch.stack(list(self._queues[OBS_IMAGES]), dim=1),
            }
            actions = self.diffusion.generate_actions(sequence_batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def _populate_queue(self, key: str, value: Tensor) -> None:
        if len(self._queues[key]) == 0:
            for _ in range(self.config.n_obs_steps):
                self._queues[key].append(value)
        else:
            self._queues[key].append(value)


class DiffusionModel(nn.Module):
    def __init__(self, config: NativeDiffusionConfig):
        super().__init__()
        self.config = config
        global_cond_dim = config.robot_state_feature.shape[0]
        self.rgb_encoder = DiffusionRgbEncoder(config)
        global_cond_dim += self.rgb_encoder.feature_dim * len(config.image_features)
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
        if config.noise_scheduler_type != "DDPM":
            raise ValueError(f"Native runtime only supports DDPM checkpoints, got {config.noise_scheduler_type}.")
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = config.num_inference_steps or self.noise_scheduler.config.num_train_timesteps

    @torch.no_grad()
    def conditional_sample(self, batch_size: int, global_cond: Tensor, noise: Tensor | None = None) -> Tensor:
        param = next(self.parameters())
        sample = noise
        if sample is None:
            sample = torch.randn(
                (batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=param.dtype,
                device=param.device,
            )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            timestep = torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device)
            model_output = self.unet(sample, timestep, global_cond=global_cond)
            sample = self.noise_scheduler.step(model_output, t, sample).prev_sample
        return sample

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(f"Expected {self.config.n_obs_steps} observation steps, got {n_obs_steps}.")
        global_cond = self._prepare_global_conditioning(batch)
        actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]
        images = batch[OBS_IMAGES]
        flat_images = images.reshape(batch_size * n_obs_steps * images.shape[2], *images.shape[3:])
        img_features = self.rgb_encoder(flat_images)
        img_features = img_features.reshape(batch_size, n_obs_steps, -1)
        global_cond_feats.append(img_features)
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)


class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], num_kp: int | None = None):
        super().__init__()
        self._in_c, self._in_h, self._in_w = input_shape
        if num_kp is not None:
            self.nets = nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features, dim=-1)
        expected_xy = attention @ self.pos_grid
        return expected_xy.view(-1, self._out_c, 2)


class DiffusionRgbEncoder(nn.Module):
    def __init__(self, config: NativeDiffusionConfig):
        super().__init__()
        self.resize = torchvision.transforms.Resize(config.resize_shape) if config.resize_shape is not None else None
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(config.crop_shape) if config.crop_is_random else self.center_crop
            )
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(weights=config.pretrained_backbone_weights)
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            self.backbone = _replace_submodules(
                self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        image_shape = next(iter(config.image_features.values())).shape
        dummy_h_w = config.crop_shape or config.resize_shape or image_shape[1:]
        dummy = torch.zeros((1, image_shape[0], *dummy_h_w))
        with torch.no_grad():
            feature_map_shape = tuple(self.backbone(dummy).shape[1:])

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.resize is not None:
            x = self.resize(x)
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        return self.relu(self.out(x))


def _replace_submodules(root_module: nn.Module, predicate, func) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [name.split(".") for name, module in root_module.named_modules() if predicate(module)]
    for *parents, key in replace_list:
        parent = root_module.get_submodule(".".join(parents)) if parents else root_module
        new_module = func(parent[int(key)] if isinstance(parent, nn.Sequential) else getattr(parent, key))
        if isinstance(parent, nn.Sequential):
            parent[int(key)] = new_module
        else:
            setattr(parent, key, new_module)
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class DiffusionConv1dBlock(nn.Module):
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DiffusionConditionalResidualBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int,
        n_groups: int,
        use_film_scale_modulation: bool,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels
        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        return out + self.residual_conv(x)


class DiffusionConditionalUnet1d(nn.Module):
    def __init__(self, config: NativeDiffusionConfig, global_cond_dim: int):
        super().__init__()
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:])
        )
        block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }

        self.down_modules = nn.ModuleList([])
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **block_kwargs),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **block_kwargs),
                DiffusionConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **block_kwargs),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for index, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = index >= len(in_out) - 1
            self.up_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **block_kwargs),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond: Tensor | None = None) -> Tensor:
        x = x.transpose(1, 2)
        timestep_embed = self.diffusion_step_encoder(timestep)
        global_feature = torch.cat([timestep_embed, global_cond], dim=-1) if global_cond is not None else timestep_embed

        skips: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            skips.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        return self.final_conv(x).transpose(1, 2)


def load_native_diffusion_policy_safetensors_bundle(model_dir: str | Path, device: torch.device):
    model_dir = Path(model_dir)
    config = NativeDiffusionConfig.from_json(model_dir / "config.json")
    policy = NativeDiffusionPolicy(config).to(device)
    state_dict = load_file(model_dir / "model.safetensors", device=str(device))
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    preprocessor = NativePolicyPreprocessor(
        model_dir / "policy_preprocessor_step_3_normalizer_processor.safetensors",
        device=device,
    )
    postprocessor = NativePolicyPostprocessor(
        model_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    )
    return policy, preprocessor, postprocessor


def _nested_stats_from_state_dict(flat: dict[str, Tensor], *, device: torch.device) -> dict[str, dict[str, Tensor]]:
    key_prefix_map = {
        "normalize_inputs.buffer_observation_image.": "observation.image.",
        "normalize_inputs.buffer_observation_state.": f"{OBS_STATE}.",
        "normalize_targets.buffer_action.": f"{ACTION}.",
        "unnormalize_outputs.buffer_action.": f"{ACTION}.",
    }
    nested: dict[str, dict[str, Tensor]] = {}
    for flat_key, value in flat.items():
        remapped = None
        for prefix, target_prefix in key_prefix_map.items():
            if flat_key.startswith(prefix):
                remapped = f"{target_prefix}{flat_key[len(prefix):]}"
                break
        if remapped is None:
            continue
        key, stat_name = remapped.rsplit(".", 1)
        nested.setdefault(key, {})[stat_name] = value.to(device=device, dtype=torch.float32)
    return nested


class NativePolicyPreprocessorFromStateDict(NativePolicyPreprocessor):
    def __init__(self, stats_state_dict: dict[str, Tensor], device: torch.device, eps: float = 1e-8):
        self.stats = _nested_stats_from_state_dict(stats_state_dict, device=device)
        self.device = device
        self.eps = eps


class NativePolicyPostprocessorFromStateDict(NativePolicyPostprocessor):
    def __init__(self, stats_state_dict: dict[str, Tensor], eps: float = 1e-8):
        self.stats = _nested_stats_from_state_dict(stats_state_dict, device=torch.device("cpu"))
        self.eps = eps


def _register_legacy_pickle_aliases() -> None:
    """Map old checkpoint pickle paths to this shared runtime module."""
    current_module = sys.modules[__name__]
    test_package = sys.modules.get("pusht.test")
    if test_package is None:
        test_package = types.ModuleType("pusht.test")
        test_package.__path__ = []
        sys.modules["pusht.test"] = test_package

    sys.modules.setdefault("pusht.test.native_diffusion_policy", current_module)
    setattr(test_package, "native_diffusion_policy", current_module)


def load_native_diffusion_policy_ckpt_bundle(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    _register_legacy_pickle_aliases()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict checkpoint at {checkpoint_path}, got {type(checkpoint).__name__}.")
    if "policy" not in checkpoint or "normalization_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} must contain 'policy' and 'normalization_state_dict' keys.")

    policy = checkpoint["policy"].to(device)
    policy.eval()
    policy.requires_grad_(False)
    normalization_state_dict = checkpoint["normalization_state_dict"]
    if not isinstance(normalization_state_dict, dict):
        raise TypeError(
            f"Checkpoint {checkpoint_path} has invalid normalization_state_dict type "
            f"{type(normalization_state_dict).__name__}."
        )

    preprocessor = NativePolicyPreprocessorFromStateDict(normalization_state_dict, device=device)
    postprocessor = NativePolicyPostprocessorFromStateDict(normalization_state_dict)
    return policy, preprocessor, postprocessor


def load_native_diffusion_policy_bundle(model_dir: str | Path, device: torch.device):
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / "model.ckpt"
    if checkpoint_path.is_file():
        return load_native_diffusion_policy_ckpt_bundle(checkpoint_path, device)
    return load_native_diffusion_policy_safetensors_bundle(model_dir, device)
