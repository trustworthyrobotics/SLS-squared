#!/usr/bin/env python3
"""Generate augmented one-step Rope prediction errors from MuJoCo rollouts.

For each selected HDF5 frame, the script restores the recorded MuJoCo state,
samples a clipped normalized warmup action, steps the environment to create a
synthetic state, then samples another clipped normalized action for the one-step
transition used to compute the model residual.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MESA_SHADER_CACHE_DIR", "/tmp/mesa_shader_cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import h5py
import mujoco
import numpy as np
import torch
from tqdm.auto import tqdm

from rope.data.rope_data_gen import render_rgb_frame
from rope.shared.lab_env import LabEnv, TaskState
from rope.train.mlpdyn_train import LeWMRopeDataset, build_markov_state, preprocess_pixels, required_markov_history


DEFAULT_DATASET_PATH = "rope/data/test_data_noshadow/rope_random_cubic_spline.h5"
DEFAULT_MODEL_DIR = "rope/models/mlpdyn_noshadow_ft"
DEFAULT_BASE_ERROR_PATH = "rope/eval/rope_one_step_error_data.pt"
DEFAULT_OUT_FILE = "rope/eval/rope_one_step_error_data_augmented.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--base-error-path", type=Path, default=DEFAULT_BASE_ERROR_PATH)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument(
        "--triples-out-file",
        type=Path,
        default=None,
        help="Optional output path for synthetic (x_t, a_t, x_next) triples. Defaults next to --out-file.",
    )
    parser.add_argument(
        "--save-triples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save synthetic state/action/next-state triples in a separate .pt file.",
    )
    parser.add_argument("--append-base-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--samples-per-frame", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-std", type=float, default=2.5)
    parser.add_argument("--action-clip", type=float, default=5.0)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=1,
        help="CPU thread count used by PyTorch/OpenMP-style kernels in the main process.",
    )
    parser.add_argument("--sim-batch-size", type=int, default=128)
    parser.add_argument("--camera", default=None)
    parser.add_argument("--disable-shadows", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def sample_normalized_actions(
    rng: np.random.Generator,
    count: int,
    action_dim: int,
    *,
    std: float,
    clip: float,
) -> np.ndarray:
    actions = rng.normal(loc=0.0, scale=std, size=(count, action_dim)).astype(np.float32)
    return np.clip(actions, -clip, clip).astype(np.float32)


def denormalize_actions(actions_norm: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray) -> np.ndarray:
    return (actions_norm * action_std + action_mean).astype(np.float32)


def restore_env_state(env: LabEnv, qpos: np.ndarray, qvel: np.ndarray, control: np.ndarray, task_target: np.ndarray) -> None:
    env.task_controller.set_target(TaskState.from_array(task_target))
    env.joint_controller.set_target(control)
    env.data.qpos[: qpos.shape[0]] = np.asarray(qpos, dtype=np.float64)
    env.data.qvel[: qvel.shape[0]] = np.asarray(qvel, dtype=np.float64)
    env.data.ctrl[: control.shape[0]] = np.asarray(control, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)


def apply_raw_task_delta(env: LabEnv, action_raw: np.ndarray, control_decimation: int) -> None:
    env.apply_task_delta(np.asarray(action_raw, dtype=np.float64))
    env.step(control_decimation)


@torch.no_grad()
def encode_pixels(
    model: torch.nn.Module,
    pixels: np.ndarray,
    device: torch.device,
    img_size: int,
    batch_size: int,
    *,
    desc: str,
    show_progress: bool = True,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(pixels)).permute(0, 3, 1, 2).contiguous()
    tensor = preprocess_pixels(tensor.unsqueeze(0), img_size)[0].to(device)
    latents = []
    starts = range(0, tensor.shape[0], batch_size)
    iterator = tqdm(
        starts,
        desc=desc,
        unit="batch",
        total=(tensor.shape[0] + batch_size - 1) // batch_size,
        disable=not show_progress,
    )
    for start in iterator:
        chunk = tensor[start : start + batch_size]
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        latents.append(model.projector(output.last_hidden_state[:, 0]))
    return torch.cat(latents, dim=0)


@torch.no_grad()
def encode_h5_pixels(
    model: torch.nn.Module,
    h5: h5py.File,
    indices: np.ndarray,
    device: torch.device,
    img_size: int,
    batch_size: int,
) -> torch.Tensor:
    latents = []
    starts = range(0, int(indices.shape[0]), batch_size)
    for start in tqdm(
        starts,
        desc="Encoding parent H5 pixels",
        unit="batch",
        total=(int(indices.shape[0]) + batch_size - 1) // batch_size,
    ):
        stop = min(start + batch_size, int(indices.shape[0]))
        pixels = np.asarray(h5["pixels"][indices[start:stop]], dtype=np.uint8)
        latents.append(
            encode_pixels(
                model,
                pixels,
                device,
                img_size,
                batch_size,
                desc="Encoding parent pixel chunk",
                show_progress=False,
            ).cpu()
        )
    return torch.cat(latents, dim=0)


def build_augmented_markov_states(
    parent_latents: torch.Tensor,
    synthetic_latents: torch.Tensor,
    next_latents: torch.Tensor,
    parent_indices: np.ndarray,
    markov_deriv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if markov_deriv != 1:
        raise ValueError(
            "This augmentation currently supports markov_deriv=1 because each synthetic branch has only "
            "one restored parent state before the generated transition."
        )
    context_len = required_markov_history(markov_deriv)
    states = []
    targets = []
    for row, parent_index in enumerate(tqdm(parent_indices.tolist(), desc="Building Markov latent triples", unit="triple")):
        parent_history_start = max(0, int(parent_index) - context_len + 2)
        parent_history = parent_latents[parent_history_start : int(parent_index) + 1]
        if parent_history.shape[0] < context_len - 1:
            padding_amt = context_len - 1 - parent_history.shape[0]
            parent_history = torch.cat((parent_history[:1].repeat(padding_amt, 1), parent_history), dim=0)
        current_history = torch.cat((parent_history, synthetic_latents[row : row + 1]), dim=0)
        next_history = torch.cat((synthetic_latents[row : row + 1], next_latents[row : row + 1]), dim=0)
        states.append(build_markov_state(current_history.unsqueeze(0), markov_deriv)[0])
        targets.append(build_markov_state(next_history.unsqueeze(0), markov_deriv)[0])
    return torch.stack(states, dim=0), torch.stack(targets, dim=0)


def build_augmented_markov_states_chunk(
    parent_latents: torch.Tensor,
    synthetic_latents: torch.Tensor,
    next_latents: torch.Tensor,
    local_parent_indices: np.ndarray,
    markov_deriv: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if markov_deriv != 1:
        raise ValueError("Synthetic MuJoCo augmentation currently expects markov_deriv=1.")
    context_len = required_markov_history(markov_deriv)
    states = []
    targets = []
    for row, parent_index in enumerate(local_parent_indices.tolist()):
        parent_history_start = max(0, int(parent_index) - context_len + 2)
        parent_history = parent_latents[parent_history_start : int(parent_index) + 1]
        if parent_history.shape[0] < context_len - 1:
            padding_amt = context_len - 1 - parent_history.shape[0]
            parent_history = torch.cat((parent_history[:1].repeat(padding_amt, 1), parent_history), dim=0)
        current_history = torch.cat((parent_history, synthetic_latents[row : row + 1]), dim=0)
        next_history = torch.cat((synthetic_latents[row : row + 1], next_latents[row : row + 1]), dim=0)
        states.append(build_markov_state(current_history.unsqueeze(0), markov_deriv)[0])
        targets.append(build_markov_state(next_history.unsqueeze(0), markov_deriv)[0])
    return torch.stack(states, dim=0), torch.stack(targets, dim=0)


@torch.no_grad()
def predict_next_states(
    model: torch.nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    device: torch.device,
    batch_size: int,
    *,
    show_progress: bool = True,
) -> torch.Tensor:
    preds = []
    starts = range(0, states.shape[0], batch_size)
    iterator = tqdm(
        starts,
        desc="Predicting next latent states",
        unit="batch",
        total=(states.shape[0] + batch_size - 1) // batch_size,
        disable=not show_progress,
    )
    for start in iterator:
        state_batch = states[start : start + batch_size].to(device)
        action_batch = actions[start : start + batch_size].to(device)
        act_emb = model.action_encoder(action_batch.unsqueeze(1))
        preds.append(model.predict(state_batch.unsqueeze(1), act_emb)[:, 0].cpu())
    return torch.cat(preds, dim=0)


def render_transition_chunk(
    *,
    h5: h5py.File,
    env: LabEnv,
    renderer: mujoco.Renderer,
    camera_id: int,
    start: int,
    parent_indices: np.ndarray,
    warmup_raw: np.ndarray,
    one_step_raw: np.ndarray,
    control_decimation: int,
    height: int,
    width: int,
    disable_shadows: bool,
) -> tuple[int, np.ndarray, np.ndarray]:
    synthetic_pixels = np.empty((parent_indices.shape[0], height, width, 3), dtype=np.uint8)
    next_pixels = np.empty((parent_indices.shape[0], height, width, 3), dtype=np.uint8)
    for local_idx, parent_index in enumerate(parent_indices.tolist()):
        restore_env_state(
            env,
            np.asarray(h5["qpos"][int(parent_index)], dtype=np.float32),
            np.asarray(h5["qvel"][int(parent_index)], dtype=np.float32),
            np.asarray(h5["control"][int(parent_index)], dtype=np.float32),
            np.asarray(h5["task_target"][int(parent_index)], dtype=np.float32),
        )
        apply_raw_task_delta(env, warmup_raw[local_idx], control_decimation)
        synthetic_pixels[local_idx] = render_rgb_frame(
            renderer,
            env,
            camera_id,
            disable_shadows=disable_shadows,
        )
        apply_raw_task_delta(env, one_step_raw[local_idx], control_decimation)
        next_pixels[local_idx] = render_rgb_frame(
            renderer,
            env,
            camera_id,
            disable_shadows=disable_shadows,
        )
    return start, synthetic_pixels, next_pixels


def iter_simulated_chunks(
    *,
    h5: h5py.File,
    parent_indices: np.ndarray,
    warmup_raw: np.ndarray,
    one_step_raw: np.ndarray,
    camera_name: str,
    control_decimation: int,
    height: int,
    width: int,
    disable_shadows: bool,
    sim_batch_size: int,
):
    total = int(parent_indices.shape[0])
    chunk_starts = list(range(0, total, sim_batch_size))
    env = LabEnv()
    camera_id = env.model.camera(camera_name).id
    with mujoco.Renderer(env.model, height=height, width=width) as renderer:
        for start in chunk_starts:
            stop = min(start + sim_batch_size, total)
            yield render_transition_chunk(
                h5=h5,
                env=env,
                renderer=renderer,
                camera_id=camera_id,
                start=start,
                parent_indices=parent_indices[start:stop],
                warmup_raw=warmup_raw[start:stop],
                one_step_raw=one_step_raw[start:stop],
                control_decimation=control_decimation,
                height=height,
                width=width,
                disable_shadows=disable_shadows,
            )


def main() -> None:
    args = parse_args()
    if args.samples_per_frame < 1:
        raise ValueError("--samples-per-frame must be positive.")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive when provided.")
    if args.action_std <= 0.0:
        raise ValueError("--action-std must be positive.")
    if args.action_clip <= 0.0:
        raise ValueError("--action-clip must be positive.")
    if args.sim_batch_size < 1:
        raise ValueError("--sim-batch-size must be positive.")
    if args.torch_num_threads < 1:
        raise ValueError("--torch-num-threads must be positive.")
    torch.set_num_threads(int(args.torch_num_threads))
    torch.set_num_interop_threads(max(1, min(2, int(args.torch_num_threads))))

    with open(args.model_dir / "config.json", "r") as f:
        config = json.load(f)
    markov_deriv = int(config.get("markov_deriv", 1))
    frameskip = int(config.get("frameskip", 1))
    img_size = int(config.get("img_size", 224))
    action_dim = int(config.get("action_dim", 3))
    state_dim = int(config.get("markov_state_dim", (markov_deriv + 1) * int(config.get("embed_dim", 12))))
    if frameskip != 1:
        raise ValueError("Synthetic MuJoCo augmentation currently expects frameskip=1.")
    if markov_deriv != 1:
        raise ValueError("Synthetic MuJoCo augmentation currently expects markov_deriv=1.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(
        f"Using torch device: {device}; pid={os.getpid()}; "
        f"sim_batch_size={int(args.sim_batch_size)}; torch_num_threads={torch.get_num_threads()}"
    )
    model = torch.load(latest_object_checkpoint(args.model_dir), map_location=device, weights_only=False).eval()

    stats_dataset = LeWMRopeDataset(
        str(args.dataset_path),
        markov_deriv=markov_deriv,
        num_preds=1,
        frameskip=frameskip,
        img_size=img_size,
        action_dim=action_dim,
    )
    action_mean = stats_dataset.action_mean.astype(np.float32)
    action_std = stats_dataset.action_std.astype(np.float32)

    with h5py.File(args.dataset_path, "r") as h5:
        finite_action = ~np.isnan(np.asarray(h5["action"][:], dtype=np.float32)).any(axis=1)
        candidate_indices = np.flatnonzero(finite_action)
        if args.max_frames is not None:
            candidate_indices = candidate_indices[: int(args.max_frames)]
        total_augmented = candidate_indices.shape[0] * int(args.samples_per_frame)
        print(
            f"Generating {total_augmented} synthetic transitions "
            f"from {candidate_indices.shape[0]} H5 frames in a single MuJoCo process."
        )
        parent_latents = encode_h5_pixels(model, h5, candidate_indices, device, img_size, args.frame_batch_size).cpu()

        camera_name = args.camera or str(h5.attrs.get("camera", "video_cam"))
        control_decimation = int(h5.attrs.get("control_decimation", 25))
        height = int(h5["pixels"].shape[1])
        width = int(h5["pixels"].shape[2])
        rng = np.random.default_rng(args.seed)
        parent_indices = np.repeat(candidate_indices.astype(np.int64), int(args.samples_per_frame))
        local_parent_indices = np.repeat(np.arange(candidate_indices.shape[0], dtype=np.int64), int(args.samples_per_frame))
        warmup_norm = sample_normalized_actions(
            rng,
            int(total_augmented),
            action_dim,
            std=args.action_std,
            clip=args.action_clip,
        )
        synthetic_actions = sample_normalized_actions(
            rng,
            int(total_augmented),
            action_dim,
            std=args.action_std,
            clip=args.action_clip,
        )
        warmup_raw = denormalize_actions(warmup_norm, action_mean, action_std)
        one_step_raw = denormalize_actions(synthetic_actions, action_mean, action_std)

        state_chunks = []
        target_chunks = []
        action_chunks = []
        error_chunks = []
        parent_index_chunks = []
        sim_iterator = iter_simulated_chunks(
            h5=h5,
            parent_indices=parent_indices,
            warmup_raw=warmup_raw,
            one_step_raw=one_step_raw,
            camera_name=camera_name,
            control_decimation=control_decimation,
            height=height,
            width=width,
            disable_shadows=bool(args.disable_shadows),
            sim_batch_size=int(args.sim_batch_size),
        )
        progress_desc = "Simulating, encoding, and scoring augmented transitions"
        with tqdm(total=int(total_augmented), desc=progress_desc, unit="transition") as progress:
            for start, synthetic_pixels, next_pixels in sim_iterator:
                stop = start + int(synthetic_pixels.shape[0])
                synthetic_latents = encode_pixels(
                    model,
                    synthetic_pixels,
                    device,
                    img_size,
                    args.frame_batch_size,
                    desc="Encoding synthetic chunk",
                    show_progress=False,
                ).cpu()
                next_latents = encode_pixels(
                    model,
                    next_pixels,
                    device,
                    img_size,
                    args.frame_batch_size,
                    desc="Encoding next chunk",
                    show_progress=False,
                ).cpu()
                states_chunk, targets_chunk = build_augmented_markov_states_chunk(
                    parent_latents,
                    synthetic_latents,
                    next_latents,
                    local_parent_indices[start:stop],
                    markov_deriv,
                )
                actions_chunk = torch.from_numpy(synthetic_actions[start:stop].reshape(stop - start, -1)).float()
                predicted_chunk = predict_next_states(
                    model,
                    states_chunk,
                    actions_chunk,
                    device,
                    batch_size=4096,
                    show_progress=False,
                )
                errors_chunk = (targets_chunk - predicted_chunk).cpu()
                state_chunks.append(states_chunk.float())
                target_chunks.append(targets_chunk.float())
                action_chunks.append(actions_chunk.float())
                error_chunks.append(errors_chunk.float())
                parent_index_chunks.append(torch.from_numpy(parent_indices[start:stop].astype(np.int64)))
                progress.update(stop - start)

    states = torch.cat(state_chunks, dim=0)
    targets = torch.cat(target_chunks, dim=0)
    actions = torch.cat(action_chunks, dim=0)
    errors = torch.cat(error_chunks, dim=0)
    parent_h5_indices = torch.cat(parent_index_chunks, dim=0)

    if states.shape[1] != state_dim or errors.shape[1] != state_dim:
        raise ValueError(f"Expected state/error dim {state_dim}, got {states.shape[1]} and {errors.shape[1]}.")

    payload = {
        "x_t": states.float(),
        "a_t": actions.float(),
        "error": errors.float(),
        "source": "synthetic_mujoco_augmented",
        "parent_h5_index": parent_h5_indices,
    }
    num_base = 0
    if args.append_base_errors:
        base = torch.load(args.base_error_path.expanduser(), map_location="cpu")
        payload["x_t"] = torch.cat((base["x_t"].float(), payload["x_t"]), dim=0)
        payload["a_t"] = torch.cat((base["a_t"].float(), payload["a_t"]), dim=0)
        payload["error"] = torch.cat((base["error"].float(), payload["error"]), dim=0)
        num_base = int(base["error"].shape[0])

    out_file = args.out_file.expanduser()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving error dataset to {out_file}...")
    torch.save(payload, out_file)

    triples_out_file = None
    if args.save_triples:
        triples_out_file = (
            args.triples_out_file.expanduser()
            if args.triples_out_file is not None
            else out_file.with_name(f"{out_file.stem}_triples.pt")
        )
        triples_out_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving synthetic triples to {triples_out_file}...")
        torch.save(
            {
                "x_t": states.float(),
                "a_t": actions.float(),
                "x_next": targets.float(),
                "error": errors.float(),
                "parent_h5_index": parent_h5_indices,
                "action_convention": "world-model normalized action: (raw_action - action_mean) / action_std",
                "source": "synthetic_mujoco_augmented",
            },
            triples_out_file,
        )

    summary = {
        "dataset_path": str(args.dataset_path),
        "model_dir": str(args.model_dir),
        "out_file": str(out_file),
        "triples_out_file": str(triples_out_file) if triples_out_file is not None else None,
        "base_error_path": str(args.base_error_path) if args.append_base_errors else None,
        "num_base_errors": num_base,
        "num_augmented_errors": int(errors.shape[0]),
        "num_total_errors": int(payload["error"].shape[0]),
        "samples_per_frame": int(args.samples_per_frame),
        "sim_batch_size": int(args.sim_batch_size),
        "num_sim_workers": 0,
        "action_std_normalized": float(args.action_std),
        "action_clip_normalized": float(args.action_clip),
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "markov_deriv": int(markov_deriv),
        "frameskip": int(frameskip),
        "action_mean": action_mean.reshape(-1).tolist(),
        "action_std": action_std.reshape(-1).tolist(),
    }
    with out_file.with_suffix(".json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
