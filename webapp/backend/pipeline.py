"""
Pipeline orchestration for the FLICK webapp.

run_pipeline() is a synchronous function intended to be called via
FastAPI BackgroundTasks (which runs sync tasks in a thread pool).
"""
import glob
import json
import os
import subprocess
import sys

from . import config
from . import coord_utils


def _update_status(job_dir: str, status: str, msg: str = '') -> None:
    with open(os.path.join(job_dir, 'status.json'), 'w') as f:
        json.dump({'status': status, 'progress_msg': msg}, f)


_SUBPROCESS_TIMEOUT = 600  # seconds; kill if a stage hangs longer than this

# Environment passed to every subprocess:
#   MPLBACKEND=Agg  → no X11 display needed
#   CUDA_VISIBLE_DEVICES=-1 → cupy/torch see no GPU and fail fast instead of hanging
_SUBPROCESS_ENV = {
    **os.environ,
    'MPLBACKEND': 'Agg',
    'CUDA_VISIBLE_DEVICES': '-1',
}


def _run(cmd: list, log_path: str) -> int:
    """Run a subprocess, streaming stdout+stderr to log_path in real time.

    Kills the process if it hasn't finished within _SUBPROCESS_TIMEOUT seconds.
    Returns returncode (negative if killed by timeout).
    """
    import threading

    with open(log_path, 'a') as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=_SUBPROCESS_ENV,
        )

        def _kill_on_timeout():
            logf.write(f'\n[webapp] Process timed out after {_SUBPROCESS_TIMEOUT}s – killed.\n')
            logf.flush()
            proc.kill()

        timer = threading.Timer(_SUBPROCESS_TIMEOUT, _kill_on_timeout)
        timer.start()
        try:
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
            proc.wait()
        finally:
            timer.cancel()

    return proc.returncode


def run_pipeline(job_id: str, stl_path: str, wind_angle: float) -> None:
    """
    Full pre-process → inference pipeline for one (STL, wind_angle) pair.

    Directory layout created under JOBS_DIR/{job_id}/:
        input.stl             – original uploaded STL (pre-existing)
        pipeline.log          – combined stdout/stderr from subprocesses
        preprocess_output/    – STL2GeoTool HDF5 output
        infer_output/         – UMAG CSV files from inference
        heatmap.png           – jet colourmap PNG for the frontend
        results.json          – metadata consumed by the frontend
        status.json           – {status, progress_msg} polled by frontend
    """
    job_dir = os.path.join(config.JOBS_DIR, job_id)
    log_path = os.path.join(job_dir, 'pipeline.log')

    try:
        # ------------------------------------------------------------------
        # Step 1: geometry analysis
        # ------------------------------------------------------------------
        _update_status(job_dir, 'running', 'Analysing STL geometry…')

        min_c, max_c, centre = coord_utils.get_stl_bbox(stl_path)
        raw_step = coord_utils.compute_step_size(min_c, max_c)

        # Fix N_POINTS = 256; derive px_resolution so the full STL fits.
        # STL2GeoTool computes: N_POINTS = int(STEP_SIZE / PX_RESOLUTION)
        # We want N_POINTS == config.N_POINTS exactly, so:
        #   PX_RESOLUTION = STEP_SIZE / N_POINTS
        n_points = config.N_POINTS  # 256
        # Use ceiling integer for step_size so we don't under-cover the STL
        step_size = max(int(raw_step) + 1, 1)
        px_resolution = step_size / n_points  # e.g. 250/256 ≈ 0.977 m

        preprocess_out = os.path.join(job_dir, 'preprocess_output')
        infer_out = os.path.join(job_dir, 'infer_output')
        os.makedirs(preprocess_out, exist_ok=True)
        os.makedirs(infer_out, exist_ok=True)

        stl_dir = os.path.dirname(stl_path) + os.sep

        # ------------------------------------------------------------------
        # Step 2: STL2GeoTool pre-processing
        # ------------------------------------------------------------------
        _update_status(job_dir, 'running', 'Running geometry pre-processing…')

        preprocess_cmd = [
            sys.executable,
            os.path.join(config.REPO_ROOT, 'pre-process', 'STL2GeoTool.py'),
            '-stl_dir', stl_dir,
            '-stl_basename', 'input',
            '-output_path', preprocess_out + os.sep,
            '-step_size', str(step_size),
            '-px_resolution', str(px_resolution),
            '-wind_direction', str(wind_angle),
        ]
        rc = _run(preprocess_cmd, log_path)
        if rc != 0:
            raise RuntimeError(f'STL2GeoTool.py exited with code {rc}. '
                               f'See {log_path} for details.')

        # STL2GeoTool writes HDF5 files to:
        #   preprocess_out/output{wind_angle}-input/
        preprocess_dir = os.path.join(
            preprocess_out, f'output{float(wind_angle)}-input')

        # ------------------------------------------------------------------
        # Step 3: NN inference
        # ------------------------------------------------------------------
        _update_status(job_dir, 'running', 'Running neural-network inference…')

        infer_cmd = [
            sys.executable,
            os.path.join(config.REPO_ROOT, 'webapp', 'backend', 'run_inference.py'),
            '--preprocess_dir', preprocess_dir + os.sep,
            '--basename', 'input',
            '--model_path', config.MODEL_LOADING_PATH,
            '--model_basename', config.MODEL_BASENAME,
            '--output_dir', infer_out + os.sep,
            '--n_points', str(n_points),
        ]
        rc = _run(infer_cmd, log_path)
        if rc != 0:
            raise RuntimeError(f'run_inference.py exited with code {rc}. '
                               f'See {log_path} for details.')

        # ------------------------------------------------------------------
        # Step 4: generate heatmap PNG from first UMAG CSV
        # ------------------------------------------------------------------
        _update_status(job_dir, 'running', 'Generating heatmap…')

        umag_files = sorted(glob.glob(os.path.join(infer_out, 'input-*-UMAG.csv')))
        if not umag_files:
            raise RuntimeError('No UMAG CSV found after inference.')
        umag_csv = umag_files[0]

        heatmap_png = os.path.join(job_dir, 'heatmap.png')
        vmin, vmax = coord_utils.generate_heatmap_png(umag_csv, heatmap_png)

        # ------------------------------------------------------------------
        # Step 5: write results.json
        # ------------------------------------------------------------------
        results = {
            'wind_angle': wind_angle,
            'step_size': step_size,
            'stl_bbox': {
                'centre_x': float(centre[0]),
                'centre_y': float(centre[1]),
                'min_z': float(min_c[2]),
                'max_z': float(max_c[2]),
            },
            'wind_speed': {
                'min': round(vmin, 4),
                'max': round(vmax, 4),
            },
        }
        with open(os.path.join(job_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        _update_status(job_dir, 'done', 'Pipeline complete.')

    except Exception as exc:
        _update_status(job_dir, 'error', str(exc))
        raise
