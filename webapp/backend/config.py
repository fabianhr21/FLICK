"""
Webapp configuration constants.
Override MODEL_LOADING_PATH and MODEL_BASENAME via environment variables
or edit this file directly.
"""
import os

# Root of the FLICK repository
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Path to directory containing <MODEL_BASENAME>.pt weights file
# Set the MODEL_LOADING_PATH env var on your server, e.g.:
#   export MODEL_LOADING_PATH=/path/to/weights/
MODEL_LOADING_PATH = os.environ.get(
    'MODEL_LOADING_PATH',
    os.path.join(REPO_ROOT, 'wind-nn', 'weights') + os.sep,
)

# Basename of the .pt weights file (without extension)
MODEL_BASENAME = os.environ.get('MODEL_BASENAME', 'model')

# Directory where per-job files are stored
JOBS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'jobs'))

# Fixed NN input/output dimension (UNet_wind is trained for 256×256)
N_POINTS = 256
