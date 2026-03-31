"""
FLICK webapp – FastAPI backend.

Start with:
    cd webapp
    uvicorn backend.main:app --reload --port 8000
"""
import glob
import io
import json
import os
import shutil
import uuid
import zipfile

import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from .config import JOBS_DIR
from .pipeline import run_pipeline

app = FastAPI(title='FLICK Wind Webapp')
os.makedirs(JOBS_DIR, exist_ok=True)


def _load_csv_fast(csv_path: str) -> np.ndarray:
    """Load a square wind-field CSV, using a .npy cache for speed."""
    npy_path = csv_path.replace('.csv', '.npy')
    if os.path.exists(npy_path):
        return np.load(npy_path)
    arr = np.genfromtxt(csv_path, delimiter=',', dtype=np.float32)
    np.save(npy_path, arr)
    return arr


# ---------------------------------------------------------------------------
# API routes  (must be registered before the static-file catch-all)
# ---------------------------------------------------------------------------

@app.post('/api/upload')
async def upload_stl(file: UploadFile = File(...)):
    """Accept an STL file, create a job directory, return job_id."""
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    stl_path = os.path.join(job_dir, 'input.stl')
    with open(stl_path, 'wb') as out:
        shutil.copyfileobj(file.file, out)

    with open(os.path.join(job_dir, 'status.json'), 'w') as f:
        json.dump({'status': 'pending', 'progress_msg': 'STL uploaded.'}, f)

    return {'job_id': job_id}


@app.post('/api/jobs/{job_id}/process')
async def process_job(
    job_id: str,
    wind_angle: float = 0.0,
    px_resolution: float | None = None,
    ref_height: float = 10.0,
    ref_velocity: float = 1.0,
    background_tasks: BackgroundTasks = None,
):
    """
    Start the full pre-process → inference pipeline in the background.
    Returns immediately; poll /status for progress.

    px_resolution: metres per pixel for the CFD grid (None = auto-fit geometry).
    ref_height:    height in metres at which the wind field is evaluated.
    ref_velocity:  reference wind speed (m/s) used to scale normalised NN output.
    """
    job_dir = os.path.join(JOBS_DIR, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail='Job not found.')

    stl_path = os.path.join(job_dir, 'input.stl')
    if not os.path.exists(stl_path):
        raise HTTPException(status_code=400, detail='STL file missing.')

    background_tasks.add_task(
        run_pipeline, job_id, stl_path, wind_angle,
        px_resolution, ref_height, ref_velocity,
    )
    return {'status': 'started', 'job_id': job_id, 'wind_angle': wind_angle,
            'px_resolution': px_resolution, 'ref_height': ref_height,
            'ref_velocity': ref_velocity}


@app.get('/api/jobs/{job_id}/status')
async def get_status(job_id: str):
    status_path = os.path.join(JOBS_DIR, job_id, 'status.json')
    if not os.path.exists(status_path):
        raise HTTPException(status_code=404, detail='Job not found.')
    with open(status_path) as f:
        return json.load(f)


@app.get('/api/jobs/{job_id}/results')
async def get_results(job_id: str):
    results_path = os.path.join(JOBS_DIR, job_id, 'results.json')
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail='Results not available yet.')
    with open(results_path) as f:
        return json.load(f)


@app.get('/api/jobs/{job_id}/stl')
async def get_stl(job_id: str):
    stl_path = os.path.join(JOBS_DIR, job_id, 'input.stl')
    if not os.path.exists(stl_path):
        raise HTTPException(status_code=404, detail='STL not found.')
    return FileResponse(stl_path, media_type='model/stl')


@app.get('/api/jobs/{job_id}/log')
async def get_log(job_id: str):
    log_path = os.path.join(JOBS_DIR, job_id, 'pipeline.log')
    if not os.path.exists(log_path):
        return JSONResponse({'log': ''})
    with open(log_path) as f:
        return JSONResponse({'log': f.read()})


_VALID_FIELDS = {'UMAG', 'UGT', 'VGT'}

@app.get('/api/jobs/{job_id}/heatmap')
async def get_heatmap(job_id: str, field: str = 'UMAG'):
    field = field.upper()
    if field not in _VALID_FIELDS:
        raise HTTPException(status_code=400,
                            detail=f'field must be one of {_VALID_FIELDS}')
    heatmap_path = os.path.join(JOBS_DIR, job_id, f'heatmap_{field}.png')
    if not os.path.exists(heatmap_path):
        raise HTTPException(status_code=404, detail='Heatmap not ready.')
    return FileResponse(heatmap_path, media_type='image/png')


# ---------------------------------------------------------------------------
# Download endpoints
# ---------------------------------------------------------------------------

@app.get('/api/jobs/{job_id}/download/csv')
async def download_csv(job_id: str):
    """Download all inference CSVs + results.json as a zip."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    infer_dir = os.path.join(job_dir, 'infer_output')
    if not os.path.isdir(infer_dir):
        raise HTTPException(status_code=404, detail='No inference output.')

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for csv_file in sorted(glob.glob(os.path.join(infer_dir, '*.csv'))):
            zf.write(csv_file, os.path.basename(csv_file))
        results_path = os.path.join(job_dir, 'results.json')
        if os.path.exists(results_path):
            zf.write(results_path, 'results.json')
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type='application/zip',
        headers={'Content-Disposition':
                 f'attachment; filename=flick_{job_id[:8]}_wind_data.zip'},
    )


@app.get('/api/jobs/{job_id}/download/geotiff')
async def download_geotiff(job_id: str, field: str = 'UMAG'):
    """Download a single wind field as a GeoTIFF (local metre coordinates)."""
    field = field.upper()
    if field not in _VALID_FIELDS:
        raise HTTPException(status_code=400,
                            detail=f'field must be one of {_VALID_FIELDS}')

    job_dir = os.path.join(JOBS_DIR, job_id)
    csv_files = sorted(glob.glob(
        os.path.join(job_dir, 'infer_output', f'input-*-{field}.csv')))
    if not csv_files:
        raise HTTPException(status_code=404, detail='Field data not found.')

    results_path = os.path.join(job_dir, 'results.json')
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail='Results not available.')
    with open(results_path) as f:
        results = json.load(f)

    data = _load_csv_fast(csv_files[0])
    n = data.shape[0]
    step_size = results['step_size']
    bbox = results['stl_bbox']
    px_size = 2.0 * step_size / n

    import rasterio
    from rasterio.transform import from_origin

    transform = from_origin(
        bbox['centre_x'] - step_size,   # west edge
        bbox['centre_y'] + step_size,   # north edge
        px_size, px_size,
    )

    buf = io.BytesIO()
    with rasterio.open(buf, 'w', driver='GTiff',
                       height=n, width=n, count=1,
                       dtype='float32', transform=transform,
                       nodata=0.0) as dst:
        dst.write(data, 1)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type='image/tiff',
        headers={'Content-Disposition':
                 f'attachment; filename=flick_{field}_{job_id[:8]}.tif'},
    )


# ---------------------------------------------------------------------------
# Wind vector data (JSON arrays for Three.js overlay)
# ---------------------------------------------------------------------------

@app.get('/api/jobs/{job_id}/wind_data')
async def get_wind_data(job_id: str):
    """Return the full U/V grids as JSON arrays for vector overlays."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    results_path = os.path.join(job_dir, 'results.json')
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail='Results not available.')

    ugt_files = sorted(glob.glob(
        os.path.join(job_dir, 'infer_output', 'input-*-UGT.csv')))
    vgt_files = sorted(glob.glob(
        os.path.join(job_dir, 'infer_output', 'input-*-VGT.csv')))
    if not ugt_files or not vgt_files:
        raise HTTPException(status_code=404, detail='Wind data not available.')

    umag_files = sorted(glob.glob(
        os.path.join(job_dir, 'infer_output', 'input-*-UMAG.csv')))

    u    = _load_csv_fast(ugt_files[0])
    v    = _load_csv_fast(vgt_files[0])
    umag = _load_csv_fast(umag_files[0]) if umag_files else None

    with open(results_path) as f:
        results = json.load(f)

    payload = {
        'n_points': results['n_points'],
        'step_size': results['step_size'],
        'u': np.round(u, 4).tolist(),
        'v': np.round(v, 4).tolist(),
    }
    if umag is not None:
        payload['umag'] = np.round(umag, 4).tolist()
    return JSONResponse(payload)


# ---------------------------------------------------------------------------
# Time series endpoints
# ---------------------------------------------------------------------------

@app.post('/api/jobs/{job_id}/timeseries')
async def upload_timeseries(job_id: str, file: UploadFile = File(None)):
    """Store a time series as [{t, v}, …] from a CSV file upload."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail='Job not found.')

    if file is not None:
        content = (await file.read()).decode('utf-8')
        rows = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.lower().startswith('time'):
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            try:
                rows.append({'t': float(parts[0]), 'v': float(parts[1])})
            except ValueError:
                continue
        if not rows:
            raise HTTPException(status_code=400, detail='No valid rows found in CSV.')
        data = rows
    else:
        raise HTTPException(status_code=400, detail='Provide a CSV file.')

    data.sort(key=lambda e: e['t'])
    ts_path = os.path.join(job_dir, 'timeseries.json')
    with open(ts_path, 'w') as f:
        json.dump(data, f)
    return {'count': len(data), 't_start': data[0]['t'], 't_end': data[-1]['t']}


@app.put('/api/jobs/{job_id}/timeseries')
async def save_timeseries_json(job_id: str, body: list):
    """Store a time series from a JSON array [{t, v}, …]."""
    job_dir = os.path.join(JOBS_DIR, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail='Job not found.')
    if len(body) < 2:
        raise HTTPException(status_code=400, detail='Need at least 2 entries.')
    data = sorted([{'t': float(e['t']), 'v': float(e['v'])} for e in body], key=lambda e: e['t'])
    with open(os.path.join(job_dir, 'timeseries.json'), 'w') as f:
        json.dump(data, f)
    return {'count': len(data)}


@app.get('/api/jobs/{job_id}/timeseries')
async def get_timeseries(job_id: str):
    ts_path = os.path.join(JOBS_DIR, job_id, 'timeseries.json')
    if not os.path.exists(ts_path):
        raise HTTPException(status_code=404, detail='No time series uploaded.')
    with open(ts_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Serve frontend static files at / (must be last)
# ---------------------------------------------------------------------------
_frontend_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app.mount('/', StaticFiles(directory=_frontend_dir, html=True), name='frontend')
