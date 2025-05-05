#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import osmnx as ox
from pyproj import CRS
import geopandas as gpd
from shapely.geometry import shape, Point, box
import laspy
import pdal
import json

def main():
    parser = argparse.ArgumentParser(description="Convert LAZ files to CityGML format.")
    parser.add_argument("input_file", help="Path to the input LAZ file.")
    parser.add_argument("output_file", help="Path to the output CityGML file.")
    parser.add_argument("--city_name", default="City", help="Name of the city for the CityGML file.")
    args = parser.parse_args()

    # Polyprep
    os.system("python ./polyprep/polyprep.py intput.geojson simplified.geojson 1.0 --remove_holes 2 --simplify_tol 0.1")
    python clipnbuildings_run.py -id data/ -o ./ -r 1000


