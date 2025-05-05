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

    # Read the LAZ file
    laz_file = laspy.read(args.input_file)
    
    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        laz_file.points,
        geometry=gpd.points_from_xy(laz_file.x, laz_file.y),
        crs=CRS.from_epsg(4326)  # Assuming WGS84
    )

    # Create a bounding box around the points
    minx, miny, maxx, maxy = gdf.total_bounds
    bbox = box(minx, miny, maxx, maxy)

    # Create a GeoDataFrame for the bounding box
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox], crs=gdf.crs)

    # Save the bounding box as a shapefile
    bbox_gdf.to_file(args.output_file.replace(".gml", ".shp"))

    # Convert to CityGML format (this is a placeholder; actual conversion would require a library or tool)
    citygml_data = {
        "city_name": args.city_name,
        "bounding_box": {
            "minx": minx,
            "miny": miny,
            "maxx": maxx,
            "maxy": maxy
        }
    }

    with open(args.output_file, 'w') as f:
        json.dump(citygml_data, f)