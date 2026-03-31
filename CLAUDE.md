# FLICK_webappdev – Claude Context
**Project**: FLICK (FMUC – Fast Modeling Urban Climate)
Urban wind modeling with Neural Networks. Collaboration: Barcelona Supercomputing Center (BSC) + UPC.
**Contact**: fabian.hernandez@bsc.es
**Active branch**: `webapp-dev`

---

## Quick Start

```bash
# Run webapp (from repo root)
cd webapp
uvicorn backend.main:app --reload --port 8000
# Open http://localhost:8000

# Run tests
pytest -v
```

---

## Directory Layout

```
pre-process/
  STL2GeoTool.py        – main pre-process entry (MPI + GPU/CPU)
  STL2GeoTool_loop.py   – batch variant for multiple angles
  gmtry_utils.py        – CPU geometry utilities (bbox, rotation, wall distance)
  opt_gpu_gmtry_utils.py – GPU (CuPy) extractor, extra fields (PRMBIN, PRMANG, etc.)
  gpu_gmtry_utils.py    – older GPU utilities
  utils.py              – MPI/pyQvarsi field averaging, meteo helpers
  ADD_FEAT.py           – add U/V features to existing H5 files
  UTILS/                – STL_from_BIM.py, simpli_STL.py
  geo4CFD/              – City4CFD integration (LAZ clipping, OSM buildings)
wind-nn/
  Models.py             – Generator2D (GAN, ACTIVE MODEL) + Discriminator
  Unet_model.py         – UNet_wind (alternative, used in older tests)
  inference-script.py   – original standalone inference (UNet, old)
  new-inference-script.py – updated inference (Generator2D)
  weights/              – model.pt lives here (NOT in git — obtain separately)
webapp/
  backend/
    main.py             – FastAPI app (all API endpoints)
    pipeline.py         – 6-stage background pipeline orchestration
    config.py           – JOBS_DIR, MODEL_LOADING_PATH, N_POINTS=256
    coord_utils.py      – get_stl_bbox, compute_step_size, generate_heatmap_png
    run_inference.py    – subprocess wrapper called by pipeline
  frontend/
    index.html          – single-page app (drag-drop STL, controls, Three.js canvas)
    app.js              – Three.js scene, API orchestration, overlays, animation
    style.css           – dark theme (#1a1d23 bg, #4f9cf9 accent), 300px sidebar
  jobs/                 – per-job dirs (uuid/), generated at runtime
post-process/
  PP_functions.py       – overlap stitching, CSV I/O, velocity magnitude/direction
  overlap.py            – main post-proc orchestration
  OverlapNInterp.py     – simplified overlap+interp workflow
  analysis_functions/geolocate.py – pixel → real-world coords
tests/
  test_gmtry_utils.py, test_stl2geotool.py, test_inference_script.py, test_overlap.py
City4CFD/               – C++ geometry generator (CGAL, submodule)
pyqvarsi/               – pyQvarsi post-processing library (submodule)
grid_of_cubes.stl       – default test geometry (390×390×30 m)
```

---

## Webapp API Reference

```
POST /api/upload                         → { job_id }
POST /api/jobs/{id}/process              → starts background pipeline
     ?wind_angle=N       (0-360°)
     ?px_resolution=N   (m/px, omit for auto)
     ?ref_height=N       (m, default 10)
     ?ref_velocity=N     (m/s, default 1.0)

GET  /api/jobs/{id}/status               → { status, progress_msg }
GET  /api/jobs/{id}/log                  → streaming pipeline log
GET  /api/jobs/{id}/results              → { wind_angle, step_size, stl_bbox, ... }
GET  /api/jobs/{id}/stl                  → input .stl file
GET  /api/jobs/{id}/heatmap?field=UMAG   → PNG (field: UMAG | UGT | VGT)
GET  /api/jobs/{id}/wind_data            → { u_grid, v_grid } JSON for Three.js overlays
GET  /api/jobs/{id}/download/csv         → ZIP of CSVs + results.json
GET  /api/jobs/{id}/download/geotiff     → GeoTIFF (local metre coords, rasterio)
POST /api/jobs/{id}/timeseries           → upload anemometer time series
GET  /api/jobs/{id}/timeseries           → retrieve stored time series
```

Frontend SPA served as static files at `/`.

---

## Pipeline Stages (`pipeline.py` → `run_pipeline()`)

| Stage | Description |
|-------|-------------|
| 1 | **Geometry Analysis** – STL bbox → `step_size` (half XY diagonal) |
| 2 | **Pre-processing** – subprocess `STL2GeoTool.py` → HDF5 (`output{angle}-{name}/`) |
| 3 | **NN Inference** – subprocess `run_inference.py` → UGT/VGT/UMAG CSV per tile |
| 4 | **Velocity Scaling** – multiply all grids by `ref_velocity` |
| 5 | **Heatmap Generation** – 3 PNGs (UMAG, UGT, VGT), Spectral_r colormap, transparent solid |
| 6 | **Metadata Export** – write `results.json` |

**Job directory** (`jobs/{uuid}/`): `input.stl`, `status.json`, `pipeline.log`, `preprocess_output/`, `infer_output/`, `heatmap_UMAG.png`, `heatmap_UGT.png`, `heatmap_VGT.png`, `results.json`, `timeseries.json`

---

## Neural Network: Generator2D (`wind-nn/Models.py`)

- **Type**: GAN-based encoder-decoder with 32 residual blocks
- **Input**: 3-channel 256×256 tensor
  - MASK ÷ 120.0, HEGT ÷ 16.0, WDST ÷ 120.0
- **Architecture**: Conv1(3→64) → Down1(→128, stride 2) → 32×ResidualBlock2D(128) → Up1(→64, bilinear) → skip + Conv2 → Output(→2) + sigmoid
- **Output**: 2 channels (Ux, Uy), normalized 0-1 via sigmoid
- **UMAG**: computed as √(Ux² + Uy²) in `run_inference.py`
- **Solid masking**: MASK==0 regions → output zeroed
- **Weights**: `wind-nn/weights/model.pt` — NOT in git, must be obtained separately
- **Device**: CUDA if available, else CPU

> UNet_wind (`Unet_model.py`) is an alternative architecture used only in older tests.

---

## Pre-processing Details

- Rotates STL by `wind_angle` around Z axis (geometry rotates relative to fixed wind direction)
- `step_size` = half of XY bounding box diagonal → guarantees full coverage for any rotation angle
- `px_resolution=None` → auto-computed as `step_size / 256`
- Output fields per grid point:
  - **MASK**: 0=solid (inside building), 1=air
  - **HEGT**: building height at that point
  - **WDST**: distance to nearest building perimeter edge
- GPU path (`opt_gpu_gmtry_utils.py`): extra fields — PRMBIN, PRMANG, PRMSEGLEN, PRMSHAN, FRONTAL, ALIGN
- Output HDF5: `output{angle}-{basename}/{basename}-{idx}-geodata.h5`

---

## Frontend Architecture (`webapp/frontend/`)

### Three.js scene setup (`app.js`)
- Y-up camera; `worldGroup` rotated -π/2 on X to convert Z-up STL coords
- STL mesh: `STLLoader` → `MeshPhongMaterial` (color 0x8899aa)
- Heatmap overlay: textured plane at `ref_height` metres, rotated `-wind_angle` on Z (undoes CFD rotation)
- Wind arrows: `InstancedMesh`, density controlled by slider (BASE_STEP × arrowDensity)
- Streamlines: RK2 integration through U/V field with bilinear interpolation, cyan→white fade
- Colormap: Spectral_r (blue→red, matches matplotlib)

### Polling flow
1. Upload STL → POST `/api/upload`
2. Set params → POST `/api/jobs/{id}/process`
3. Poll `/status` every 1 s → update progress bar + log
4. On `status=done`: fetch `/results` + `/heatmap` + `/wind_data` → render
5. Optional: upload time series CSV → animate arrow colors over time

---

## Data Formats

| Format | Description |
|--------|-------------|
| HDF5 (`-geodata.h5`) | `/FIELD/VARIABLES/` → MASK, HEGT, WDST (input); U, V (training target) |
| CSV | 256×256 float matrices: `{name}-{idx}-UGT.csv`, `VGT.csv`, `UMAG.csv` |
| `results.json` | `{ wind_angle, step_size, px_resolution, stl_bbox, ref_height, ref_velocity, umag_range, ugt_range, vgt_range }` |
| GeoTIFF | Local metre CRS (not geographic), generated with rasterio |
| `timeseries.json` | Anemometer time series, stored per job |

---

## Key Gotchas

- **Heatmap colormap**: `Spectral_r` (NOT jet)
- **3 separate heatmaps** per run: UMAG (magnitude), UGT (Ux), VGT (Uy)
- **Velocity is normalized**: NN output is 0-1 (sigmoid); real velocity = output × `ref_velocity`
- **`rotation_matrix_around_z`**: the `convention` parameter ("dataset" vs "math") has no effect — both branches use identical formula
- **`test_rotation_matrix_identity`**: tests R(270°) == eye(3) — mathematically incorrect, encodes a domain convention
- **`generator2D` is the active model** — UNet_wind is used only in older `test_inference_script.py`
- **Model weights not in git** — `wind-nn/weights/model.pt` must be obtained separately
- **`px_resolution=None`** → auto = `step_size / N_POINTS` (e.g. 320 m domain → 1.25 m/px)
- **Solid regions** (MASK==0) are zeroed in wind output and transparent in PNG

---

## Tests & CI

```bash
pytest -v          # from repo root
```

- `pytest.ini`: `norecursedirs = pyqvarsi`
- GitHub Actions: `.github/workflows/python-tests.yml` (Python 3.10, numpy/torch/pytest)
- Tests mock MPI/pyQvarsi to allow running without HPC environment
