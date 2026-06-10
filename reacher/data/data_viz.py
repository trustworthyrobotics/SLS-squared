#!/usr/bin/env python3
"""Save a random trajectory from a Reacher HDF5 dataset as an MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None

import h5py
import imageio.v2 as imageio
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = REPO_ROOT / "reacher/data/train_data_noisy.h5"
DEFAULT_FPS = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_path",
        type=Path,
        nargs="?",
        default=DEFAULT_DATASET_PATH,
        help=f"Path to a Reacher HDF5 dataset. Defaults to {DEFAULT_DATASET_PATH}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output MP4 path. Defaults to <dataset_stem>_random_traj_<episode>.mp4.",
    )
    parser.add_argument(
        "--episode-idx",
        type=int,
        default=None,
        help="Optional episode index. If omitted, sample one uniformly at random.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS.")
    return parser.parse_args()


def resolve_output_path(dataset_path: Path, output_path: Path | None, episode_idx: int) -> Path:
    if output_path is not None:
        return output_path.expanduser().resolve()
    return dataset_path.with_name(f"{dataset_path.stem}_random_traj_{episode_idx:05d}.mp4")


def load_episode_frames(dataset_path: Path, episode_idx: int | None) -> tuple[np.ndarray, int, float]:
    with h5py.File(dataset_path, "r") as h5:
        required_keys = ("ep_len", "ep_offset", "pixels")
        missing = [key for key in required_keys if key not in h5]
        if missing:
            missing_str = ", ".join(missing)
            raise KeyError(f"Dataset is missing required keys: {missing_str}")

        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        if ep_len.ndim != 1 or ep_offset.ndim != 1 or ep_len.shape != ep_offset.shape:
            raise ValueError(
                f"Expected ep_len and ep_offset to be 1D arrays with the same shape, "
                f"got {ep_len.shape} and {ep_offset.shape}."
            )
        if len(ep_len) == 0:
            raise ValueError(f"No episodes found in {dataset_path}.")

        if episode_idx is None:
            rng = np.random.default_rng()
            selected_idx = int(rng.integers(0, len(ep_len)))
        else:
            selected_idx = int(episode_idx)

        if selected_idx < 0 or selected_idx >= len(ep_len):
            raise IndexError(f"episode_idx={selected_idx} is out of range for {len(ep_len)} episodes.")

        length = int(ep_len[selected_idx])
        offset = int(ep_offset[selected_idx])
        if length <= 0:
            raise ValueError(f"Episode {selected_idx} has non-positive length {length}.")

        rows = slice(offset, offset + length)
        pixels_ds = h5["pixels"]
        try:
            frames = np.asarray(pixels_ds[rows], dtype=np.uint8)
        except OSError as exc:
            plist = pixels_ds.id.get_create_plist()
            filters = [plist.get_filter(i) for i in range(plist.get_nfilters())]
            filter_names = [flt[3].decode("utf-8", errors="replace") for flt in filters]
            plugin_hint = ""
            if "blosc" in filter_names:
                plugin_hint = (
                    " This dataset uses the Blosc HDF5 filter. Install `hdf5plugin` in the active "
                    "environment, for example: `pip install hdf5plugin`."
                )
            raise RuntimeError(
                "Failed to read `pixels` from the HDF5 dataset. "
                f"Dataset: {dataset_path}. Compression: {pixels_ds.compression!r}. "
                f"Filters: {filter_names!r}."
                "If this file was copied from another machine, it may require an HDF5 plugin/filter "
                f"that is not available in this environment.{plugin_hint}"
            ) from exc
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Expected RGB frames with shape (T, H, W, 3), got {frames.shape}.")

        fps = float(h5.attrs.get("video_fps", h5.attrs.get("physics_freq_hz", DEFAULT_FPS)))
        return frames, selected_idx, fps


def save_video(frames: np.ndarray, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        output_path,
        [np.ascontiguousarray(frame) for frame in frames],
        fps=fps,
        quality=8,
        macro_block_size=1,
    )


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frames, episode_idx, dataset_fps = load_episode_frames(dataset_path, args.episode_idx)
    fps = float(args.fps) if args.fps is not None else dataset_fps
    if fps <= 0.0:
        raise ValueError(f"FPS must be positive, got {fps}.")

    output_path = resolve_output_path(dataset_path, args.output, episode_idx)
    save_video(frames, output_path, fps)
    print(f"Saved episode {episode_idx} with {len(frames)} frames to {output_path}")


if __name__ == "__main__":
    main()
