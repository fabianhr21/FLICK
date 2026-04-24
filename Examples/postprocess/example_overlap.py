"""Post-processing overlap example.

Stitches prediction tiles using exponential decay weighting.
Expects inference output files in the nn output directory.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

print("Run post-processing overlap with:")
print("  python -m flick_urban.postprocess.overlap --input_dir <nn_output_dir> --output_dir <final_output_dir>")
