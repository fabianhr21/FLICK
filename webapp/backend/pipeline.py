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
import numpy as np

from . import config
from . import coord_utils


def _update_status(job_dir: str, status: str, msg: str = '') -> None:
    with open(os.path.join(job_dir, 'status.json'), 'w') as f:
        json.dump({'status': status, 'progress_msg': msg}, f)


_SUBPROCESS_TIMEOUT = 600  # seconds; kill if a stage hangs longer than this

# Environment passed to every subprocess:
#   MPLBACKEND=Agg  → no X11 display needed
_SUBPROCESS_ENV = {
    **os.environ,
    'MPLBACKEND': 'Agg',
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


def run_pipeline(job_id: str, stl_path: str, wind_angle: float,
                 user_px_resolution: float | None = None,
                 ref_height: float = 10.0,
                 ref_velocity: float = 1.0) -> None:
    """
    Full pre-process → inference pipeline for one (STL, wind_angle) pair.

    user_px_resolution: metres per pixel requested by the user.
        None  → auto: defaults to a px_resolution that yields 256 grid points.
        float → cell size is fixed at this value; N_POINTS is derived.

    ref_height:   height (m) at which the wind field is evaluated (heatmap Z).
    ref_velocity: reference wind speed (m/s) at ref_height.  The NN outputs
                  normalised velocities; they are scaled by this value.

    The domain (step_size) always covers the full STL geometry at any
    rotation angle (uses the XY diagonal + 10 % margin).
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

        # Domain always covers the full STL with a 10 % margin.
        step_size = max(int(raw_step) + 1, 1)

        if user_px_resolution is not None:
            px_resolution = user_px_resolution
        else:
            # Auto: target the default grid size (256 points).
            px_resolution = step_size / config.N_POINTS

        # STL2GeoTool computes: N_POINTS = int(STEP_SIZE / PX_RESOLUTION)
        n_points = max(int(step_size / px_resolution), 1)

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
        # Step 4: scale inference output by reference velocity
        # ------------------------------------------------------------------
        _update_status(job_dir, 'running', 'Scaling wind field…')

        umag_files = sorted(glob.glob(os.path.join(infer_out, 'input-*-UMAG.csv')))
        if not umag_files:
            raise RuntimeError('No UMAG CSV found after inference.')
        umag_csv = umag_files[0]

        if ref_velocity != 1.0:
            for csv_path in sorted(glob.glob(
                    os.path.join(infer_out, 'input-*-*.csv'))):
                data = np.loadtxt(csv_path, delimiter=',')
                data *= ref_velocity
                np.savetxt(csv_path, data, delimiter=',')

        # ------------------------------------------------------------------
        # Step 5: generate one heatmap PNG per wind field
        # ------------------------------------------------------------------
        _update_status(job_dir, 'running', 'Generating heatmaps…')

        # Map field key → (CSV glob suffix, colormap, human label)
        FIELD_SPECS = {
            'UMAG': ('UMAG', 'Spectral_r', 'Wind speed magnitude (m/s)'),
            'UGT':  ('UGT',  'Spectral_r', 'U component – X velocity (m/s)'),
            'VGT':  ('VGT',  'Spectral_r', 'V component – Y velocity (m/s)'),
        }

        wind_fields = {}
        for key, (suffix, cmap, label) in FIELD_SPECS.items():
            csv_files = sorted(glob.glob(
                os.path.join(infer_out, f'input-*-{suffix}.csv')))
            if not csv_files:
                continue
            png_path = os.path.join(job_dir, f'heatmap_{key}.png')
            vmin, vmax = coord_utils.generate_heatmap_png(
                csv_files[0], png_path, cmap_name=cmap)
            wind_fields[key] = {
                'label': label,
                'min': round(vmin, 4),
                'max': round(vmax, 4),
            }

        if not wind_fields:
            raise RuntimeError('No field CSVs found after inference.')

        # ------------------------------------------------------------------
        # Step 6: write results.json
        # ------------------------------------------------------------------
        results = {
            'wind_angle': wind_angle,
            'step_size': step_size,
            'px_resolution': round(px_resolution, 4),
            'n_points': n_points,
            'ref_height': ref_height,
            'ref_velocity': ref_velocity,
            'stl_bbox': {
                'centre_x': float(centre[0]),
                'centre_y': float(centre[1]),
                'min_z': float(min_c[2]),
                'max_z': float(max_c[2]),
            },
            'wind_fields': wind_fields,
        }
        with open(os.path.join(job_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        _update_status(job_dir, 'done', 'Pipeline complete.')

    except Exception as exc:
        _update_status(job_dir, 'error', str(exc))
        raise
