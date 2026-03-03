"""
FLICK webapp – FastAPI backend.

Start with:
    cd webapp
    uvicorn backend.main:app --reload --port 8000
"""
import json
import os
import shutil
import uuid

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import JOBS_DIR
from .pipeline import run_pipeline

app = FastAPI(title='FLICK Wind Webapp')
os.makedirs(JOBS_DIR, exist_ok=True)


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
    background_tasks: BackgroundTasks = None,
):
    """
    Start the full pre-process → inference pipeline in the background.
    Returns immediately; poll /status for progress.
    """
    job_dir = os.path.join(JOBS_DIR, job_id)
    if not os.path.isdir(job_dir):
        raise HTTPException(status_code=404, detail='Job not found.')

    stl_path = os.path.join(job_dir, 'input.stl')
    if not os.path.exists(stl_path):
        raise HTTPException(status_code=400, detail='STL file missing.')

    background_tasks.add_task(run_pipeline, job_id, stl_path, wind_angle)
    return {'status': 'started', 'job_id': job_id, 'wind_angle': wind_angle}


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


@app.get('/api/jobs/{job_id}/heatmap')
async def get_heatmap(job_id: str):
    heatmap_path = os.path.join(JOBS_DIR, job_id, 'heatmap.png')
    if not os.path.exists(heatmap_path):
        raise HTTPException(status_code=404, detail='Heatmap not ready.')
    return FileResponse(heatmap_path, media_type='image/png')


# ---------------------------------------------------------------------------
# Serve frontend static files at / (must be last)
# ---------------------------------------------------------------------------
_frontend_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'frontend'))
app.mount('/', StaticFiles(directory=_frontend_dir, html=True), name='frontend')
