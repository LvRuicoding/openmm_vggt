# Repository Guidelines

## Project Structure & Module Organization
`openmm_vggt/` is the main package. Use `datasets/` for dataset adapters (`VKITTI1SequenceDataset`, KITTI loaders), `models/` for model assembly (`VGGT_decoder_global`, `Aggregator`), `heads/` for prediction heads, `layers/` for reusable transformer blocks, and `utils/` for geometry, pose, and visualization helpers. Training entry points live in `tools/`, and mmengine configs live in `configs/`. The repository also exposes local symlinks such as `data/`, `ckpt/`, and `results/`; treat them as environment-specific inputs, not source files.

## Build, Test, and Development Commands
Install the package in editable mode with `pip install -e .`.
Start the current training workflow with `python tools/train_vkitti.py configs/vkitti_depth.py`.
Run distributed training with `torchrun --nproc_per_node=4 tools/train_vkitti.py configs/vkitti_depth.py`.
Do a fast syntax pass before pushing with `python -m compileall openmm_vggt tools configs`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, module filenames in `snake_case.py`, functions in `snake_case`, and classes in `CamelCase` where possible. Match surrounding code when editing older modules that already use names like `VGGT_decoder_global`. Keep imports explicit, prefer small helper functions over repeated tensor logic, and preserve mmengine registration patterns by importing new datasets/models through `openmm_vggt/__init__.py` and the relevant subpackage `__init__.py`.

## Testing Guidelines
There is no dedicated `tests/` directory yet, so every change should include at least one executable check. Minimum bar: run `python -m compileall openmm_vggt tools configs`. For model or dataset changes, add a small smoke test locally: one config load, one dataset sample, or one forward pass with a tiny batch. When adding formal tests later, place them under `tests/` and name files `test_<feature>.py`.

## Commit & Pull Request Guidelines
This checkout does not include local `.git` history, so no repository-specific commit convention could be verified here. Use short imperative subjects such as `add vkitti depth smoke test` or `fix camera token reshape`. In pull requests, include the config used, any dataset/checkpoint path assumptions, exact commands run, and before/after metrics or qualitative outputs for model behavior changes.

## Configuration & Data Paths
Prefer CLI overrides such as `--checkpoint`, `--output-dir`, `--epochs`, `--batch-size`, and `--lr` instead of hardcoding machine-specific paths in configs. Keep absolute dataset and checkpoint locations local when possible, and avoid committing secrets or private filesystem paths unless the repo intentionally standardizes them.
