"""Neural network inference example.

Requires model weights in 170625_weights/ (request from fabian.hernandez@bsc.es).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "../../170625_weights")

if not os.path.exists(WEIGHTS_PATH):
    print("Model weights not found.")
    print("Request from: fabian.hernandez@bsc.es")
    print(f"Expected path: {os.path.abspath(WEIGHTS_PATH)}")
    sys.exit(1)

print("Run inference with:")
print("  python -m flick_urban.nn.inference --data_sample_basename <name>")
