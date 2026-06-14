# SLS2: Safe Latent Space Planning

This repository contains code for **SLS2**, a framework for safe feedback planning from pixels with learned latent world models, conformal calibration, and robust MPC via System Level Synthesis (SLS).

**Links:** [Website](https://trustworthyrobotics.github.io/SLS-squared/) | [Paper](https://trustworthyrobotics.github.io/SLS-squared/assets/paper/SLS2_paper.pdf) | [Video](https://www.youtube.com/watch?v=3sYYNSQqwSQ)

The method follows the paper draft _Pixels to Proofs: Probabilistically-Safe Control in Latent World Models via Conformalized Robust MPC_. At a high level, SLS2:

1. Learns compact Markov latent world models from image-action trajectories.
2. Calibrates latent prediction error with conformal prediction.
3. Fits in-distribution latent support regions and task-specific latent safety classifiers.
4. Plans in latent space using nominal iLQR/MPPI or robust SLS MPC with reachable tubes.

## Repository Layout

```text
.
|-- pusht/             # PushT planar manipulation experiments
|-- reacher/           # DeepMind Control Suite Reacher experiments
|-- rope/              # Bimanual MuJoCo rope manipulation experiments
|-- ogbench_cube/      # OGBench cube manipulation experiments
|-- error_calib/       # Shared conformal error-model code
|-- third_party/       # Local third-party dependencies and SLS solver code
|-- pyproject.toml
`-- requirements.txt
```

Each task folder follows the same pattern:

```text
<task>/
|-- data/      # Dataset generation and visualization scripts
|-- train/     # Latent world-model training and fine-tuning
|-- eval/      # Rollout evaluation, error extraction, calibration artifacts
|-- plan/      # iLQR, MPPI, constrained iLQR, and SLS2 planners
`-- shared/    # Environment wrappers, model definitions, task utilities
```

## Method-to-Code Map

**Latent world model.** The world model is a JEPA-style image encoder plus an MLP latent dynamics predictor. It builds Markov latent states from the current embedding and finite-difference latent features.

Relevant files:

```text
*/train/mlpdyn_train.py
*/train/mlpdyn_ft.py
*/shared/models.py
```

**Prediction-error calibration.** One-step prediction errors are generated from held-out rollouts, then used to train/calibrate learned error predictors or fixed covariance bounds.

Relevant files:

```text
*/eval/generate_error*.py
*/eval/compute_fixed_error_covariance.py
*/eval/conformal_calibration.py
error_calib/error_model/
```

**Latent in-distribution constraints.** Calibration rollouts are used to fit ellipsoidal latent support regions, which can be enforced during robust planning.

Relevant files:

```text
*/eval/fit_latent_ellipsoid.py
rope/eval/latent_domain_stats.py
```

**Latent obstacle/safety classifiers.** Task-specific scripts collect safe/unsafe examples, train latent classifiers on frozen world-model embeddings, and conformalize classifier thresholds.

Relevant files:

```text
rope/plan/obs_data_collect.py
rope/plan/obs_start_goal_sampling.py
rope/plan/obs_ellipsoid.py
rope/plan/obstacle_net.py
reacher/plan/obs_data_collect_new.py
reacher/plan/obstacle_net.py
ogbench_cube/plan/obs_data_collect_height.py
ogbench_cube/plan/obs_start_goal_sampling_height.py
ogbench_cube/plan/obstacle_net_height.py
pusht/plan/plot_tsne_latent_obs.py
```

**Planning.** Nominal planners optimize in learned latent space. SLS2 planners use MPPI warm starts, GPU SLS tube propagation, conformal error bounds, obstacle classifiers, and optional latent support constraints.

Relevant files:

```text
*/plan/plan_ilqr_mpc.py
pusht/plan/plan_mppi.py
pusht/plan/plan_mppi_ilqr_track.py
rope/plan/plan_sls_mpc_mppi.py
rope/plan/plan_sls_mpc_mppi_save_tubes.py
rope/plan/plan_constrained_ilqr_mpc_mppi.py
reacher/plan/plan_sls_mpc_mppi.py
reacher/plan/plan_sls_mpc_mppi_save_tubes.py
reacher/plan/plan_constrained_ilqr_mpc_mppi.py
ogbench_cube/plan/plan_sls_mpc_mppi_fixed_grasp.py
ogbench_cube/plan/plan_sls_mpc_mppi_fixed_grasp_save_tubes.py
ogbench_cube/plan/plan_constrained_ilqr_mppi_fixed_grasp.py
```

## Setup

Create and activate an environment, then install the package and dependencies:

```bash
pip install -r requirements.txt
pip install -e .
pip install -e third_party/gpu_sls
```

Some experiments use local third-party checkouts under `third_party/`, including `gpu_sls`, `dinov2`, `dm_control`, `ogbench`, `le-wm`, and related world-model tooling. If a script imports one of these packages directly, install the corresponding third-party package in editable mode as needed.

For MuJoCo headless rendering, many scripts set:

```bash
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

On a local desktop, `MUJOCO_GL=glfw` may be more appropriate.

## Typical Workflow

Run commands from the repository root.

### 1. Generate or provide datasets

Each task expects HDF5 datasets with image observations, actions, episode lengths, and episode offsets. Examples:

```bash
python -m reacher.data.reacher_data_gen
python -m rope.data.rope_data_gen
python -m ogbench_cube.data.ogbench_cube_data_gen
python -m pusht.data.pusht_data_gen
```

The default paths inside scripts point to experiment-specific data locations. Override them with `--dataset-path`, `--out`, or `--outdir` when running new experiments.

### 2. Train or fine-tune latent world models

```bash
python -m reacher.train.mlpdyn_train --dataset-path <train.h5> --run-dir reacher/models/mlpdyn
python -m rope.train.mlpdyn_train --dataset-path <train.h5> --run-dir rope/models/mlpdyn
python -m ogbench_cube.train.mlpdyn_train --dataset-path <train.h5> --run-dir ogbench_cube/models/mlpdyn
python -m pusht.train.mlpdyn_train --dataset-path <train.h5> --run-dir pusht/models/mlpdyn
```

Fine-tuning entry points are available as `*/train/mlpdyn_ft.py`.

### 3. Evaluate latent rollouts

```bash
python -m reacher.eval.mlpdyn_eval --model-dir <model_dir> --dataset-path <eval.h5>
python -m rope.eval.mlpdyn_eval --model-dir <model_dir> --dataset-path <eval.h5>
python -m ogbench_cube.eval.generate_error_markov --model-dir <model_dir> --dataset-path <eval.h5>
python -m pusht.eval.mlpdyn_eval --model-dir <model_dir> --dataset-path <eval.h5>
```

### 4. Build conformal calibration artifacts

Generate one-step latent prediction errors:

```bash
python -m reacher.eval.generate_error_markov --model-dir <model_dir> --dataset-path <calib.h5>
python -m rope.eval.generate_error --model-dir <model_dir> --dataset-path <calib.h5>
python -m ogbench_cube.eval.generate_error_markov --model-dir <model_dir> --dataset-path <calib.h5>
```

Train/calibrate learned error models:

```bash
python -m reacher.eval.conformal_calibration --data-path <one_step_error_data.pt>
python -m rope.eval.train_error
python -m ogbench_cube.eval.conformal_calibration
```

Compute fixed covariance and latent support constraints:

```bash
python -m reacher.eval.compute_fixed_error_covariance --data-path <one_step_error_data.pt>
python -m rope.eval.compute_fixed_error_covariance --data-path <one_step_error_data.pt>
python -m ogbench_cube.eval.compute_fixed_error_covariance --data-path <one_step_error_data.pt>

python -m reacher.eval.fit_latent_ellipsoid --model-dir <model_dir> --dataset-path <calib.h5>
python -m rope.eval.fit_latent_ellipsoid --model-dir <model_dir> --dataset-path <calib.h5>
python -m ogbench_cube.eval.fit_latent_ellipsoid --model-dir <model_dir> --dataset-path <calib.h5>
```

### 5. Train task safety classifiers

Examples:

```bash
python -m reacher.plan.obs_data_collect_new
python -m reacher.plan.obstacle_net --model-dir <model_dir>

python -m rope.plan.obs_data_collect
python -m rope.plan.obstacle_net --model-dir <model_dir>

python -m ogbench_cube.plan.obs_data_collect_height
python -m ogbench_cube.plan.obstacle_net_height --model-dir <model_dir>
```

These scripts produce classifier datasets and model artifacts used by the constrained planners.

### 6. Run planners

Nominal latent iLQR:

```bash
python -m reacher.plan.plan_ilqr_mpc --model-dir <model_dir> --dataset-path <eval.h5>
python -m rope.plan.plan_ilqr_mpc --model-dir <model_dir> --dataset-path <eval.h5>
python -m ogbench_cube.plan.plan_ilqr_mpc --model-dir <model_dir> --dataset-path <eval.h5>
python -m pusht.plan.plan_ilqr_mpc --model-dir <model_dir> --dataset-path <eval.h5>
```

Robust SLS2 planners:

```bash
python -m reacher.plan.plan_sls_mpc_mppi --config_path reacher/plan/sample_config_sls_mppi.yaml
python -m rope.plan.plan_sls_mpc_mppi --config_path rope/plan/sample_config_mppi.yaml
python -m ogbench_cube.plan.plan_sls_mpc_mppi_fixed_grasp --config_path ogbench_cube/plan/sample_config_mppi.yaml
```

Tube-saving variants write `tube_data.npz`, `executed_states.npz`, and related artifacts for plotting:

```bash
python -m reacher.plan.plan_sls_mpc_mppi_save_tubes --config_path reacher/plan/sample_config_sls_mppi_save_tubes.yaml
python -m rope.plan.plan_sls_mpc_mppi_save_tubes --config_path rope/plan/sample_config_mppi_save_tubes.yaml
python -m ogbench_cube.plan.plan_sls_mpc_mppi_fixed_grasp_save_tubes --config_path ogbench_cube/plan/sample_config_mppi_save_tubes.yaml
```

Plot saved latent tubes:

```bash
python -m reacher.plan.plot_saved_latent_tubes <run_dir_or_tube_data.npz>
python -m ogbench_cube.plan.plot_saved_latent_tubes <run_dir_or_tube_data.npz>
```

## Task Notes

**Reacher.** Uses DeepMind Control Suite Reacher. The robust setting constrains joint configurations and uses latent obstacle classifiers trained from sampled joint boxes.

**OGBench Cube.** Uses `cube-single-v0` style manipulation. The fixed-grasp planners isolate post-grasp cube transport and can enforce height-threshold constraints.

**Rope.** Uses a custom MuJoCo bimanual rope environment with two KUKA iiwa arms. The robust setting can enforce learned obstacle or ellipsoidal constraints while tracking a latent/task goal.

**PushT.** Includes nominal latent planners, diffusion-policy baselines, MPPI, and iLQR/MPPI tracking experiments. PushT support is less focused on SLS tube constraints than the Reacher, Cube, and Rope code paths.

## Outputs and Artifacts

Common outputs include:

```text
models/                  # World-model checkpoints and config.json files
eval/*error*.pt          # One-step latent error datasets and covariance artifacts
eval/latent_ellipsoid*/  # Calibrated latent support sets
plan/*/trajectory_summary.json
plan/*/metrics.json
plan/*/rollout.mp4
plan/*/executed_states.npz
plan/*/executed_actions.npz
plan/*/tube_data.npz
```

Datasets, checkpoints, generated videos, and run directories can be large and are generally treated as experiment artifacts rather than source code.

## Development Notes

- Run scripts from the repository root so relative paths resolve correctly.
- Many defaults encode paths from the original experiment runs. Prefer passing explicit `--model-dir`, `--dataset-path`, `--data-path`, and `--out-dir` values for new runs.
- Shared conformal error-model code lives under `error_calib/error_model`.
- The robust SLS planner depends on `third_party/gpu_sls`; install it editable before running SLS2 planners.
- The cleaned repository intentionally omits old scratch folders, archived run outputs, and personal-path names.
