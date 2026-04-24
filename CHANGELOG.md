# Changelog

All notable changes to FLICK will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - 2026-04-24

### Added
- `flick_urban` installable Python package (`pip install -e .`)
- Proper package structure: `flick_urban.preprocess`, `flick_urban.nn`, `flick_urban.postprocess`
- `flick_urban.preprocess.geo4cfd.ansa` submodule with all ANSA meshing automation scripts
- `pyproject.toml` with optional dependency groups: `gpu`, `hpc`, `dev`, `docs`
- Split requirements files: `requirements_gpu.txt`, `requirements_hpc.txt`, `requirements_dev.txt`, `requirements_docs.txt`
- Sphinx documentation scaffold with autodoc API reference
- `CHANGELOG.md`
- `MANIFEST.in` for source distribution packaging
- `Examples/` directory with example scripts and `grid_of_cubes.stl` test geometry
- `scripts/` directory with workflow and HPC scripts
- `scripts/hpc/` with MareNostrum 5 SLURM templates (p2, p3 partitions)
- `Testsuite/` (renamed from `tests/`)
- CI/CD: `docs.yml` workflow for Sphinx build validation
- CI/CD: updated `python-tests.yml` — installs via `pip install -e .[dev]`, checks out submodules

### Changed
- City4CFD and pyqvarsi converted from embedded copied code to proper git submodules (~700 MB reduction)
- `new-inference-script.py` → `flick_urban/nn/inference.py` (canonical inference; added docstrings)
- `compile_tools.sh` now runs `git submodule update --init --recursive` before building
- `pytest.ini` testpaths updated to `Testsuite/`

### Removed
- `output_sanjeronimo.stl` (16 MB) — too large for release distribution
- `wind-nn/inference-script.py` (old inference version superseded by `new-inference-script.py`)
- `pre-process/signal_process/` (empty stub)
- Debug dump artifacts from `pre-process/geo4CFD/`
