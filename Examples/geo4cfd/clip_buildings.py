#!/usr/bin/env python3
"""
Script: Clip a LAZ file to a defined area and extract OSM building footprints for that same extent.
Generates a CITY4CFD config.json file for downstream ETL processing.

Dependencies:
  - LAStools or PDAL
  - OSMnx
  - GeoPandas, Shapely, Fiona, pyproj, laspy
  - Polyprep (local module)
"""
import os
import sys
import argparse
import subprocess
import osmnx as ox
from pyproj import CRS, Transformer
import geopandas as gpd
from shapely.geometry import shape, Point, box
from shapely.ops import unary_union
import laspy
import pdal
import json
import math

from .polyprep import process_polygons

# Define default CRS
DEFAULT_CRS = 'EPSG:4326' 

# NOTE: Commented out for portability. valid PDAL environment usually handles this.
# os.environ['PDAL_DRIVER_PATH'] = '/home/fabianh/anaconda3/envs/qgis_env/bin/pdal'

def create_city4cfd_config(output_dir, center_point, radius,bounds, ground_laz, building_laz, polygon_geojson, output_name="mesh_mty_topo"):
    """
    Generates the config.json file for CITY4CFD using the paths and coordinates 
    derived from the processing steps.
    """
    
    # Ensure we use absolute paths for the ETL to avoid CWD issues
    abs_ground = os.path.abspath(ground_laz)
    abs_build = os.path.abspath(building_laz)
    abs_poly = os.path.abspath(polygon_geojson)
    
    # If radius wasn't provided (e.g. BBOX mode), default to 400 or calculate from bounds
    roi_radius = radius if radius else 400

    xmin, ymin, xmax, ymax = bounds
    poi = [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax]
    ]

    config = {
        "point_clouds": {
            "ground": abs_ground,
            "buildings": abs_build
        },
        "polygons": [
            {
                "type": "Building",
                "path": abs_poly,
                "unique_id": "gid",
                "height_attribute": "height",
                "floor_attribute": "num_floors",
                "floor_height": 3,
                "height_attribute_advantage": False,
                "avoid_bad_polys": True,
                "refine": False
            }
        ],
        "reconstruction_regions": [
            {
                "influence_region": poi,
                "lod": "1.2",
                "complexity_factor": 0.1,
                "lod13_step_height": 3,
                "validate": True,
                "enforce_validity": "lod1.2",
                "relative_alpha": 500,
                "relative_offset": 1200,
                "skip_gap_closing": False,
                "import_advantage": False
            }
        ],
        "point_of_interest": [center_point.x, center_point.y],
        "domain_bnd": None,
        "top_height": 300,
        "buffer_region": -20,
        "reconstruct_boundaries": False,
        "bnd_type_bpg": "Rectangle",
        "bpg_blockage_ratio": False,
        "flow_direction": [1, 0],
        "bpg_domain_size": [20, 30, 40, 20],
        "terrain_thinning": 80,
        "smooth_terrain": {
            "iterations": 1,
            "max_pts": 250000
        },
        "flat_terrain": True,
        "building_percentile": 90,
        "min_height": 3,
        "min_area": 50,
        "reconstruct_failed": False,
        "intersect_buildings_terrain": False,
        "edge_max_len": 5,
        "output_file_name": output_name,
        "output_format": "stl",
        "output_separately": True,
        "output_log": True,
        "log_file": "logFile.log"
    }

    config_path = os.path.join(output_dir, "config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"CITY4CFD configuration generated at: {config_path}")
    return config_path

def get_laz_crs(laz_path):
    las = laspy.read(laz_path)
    crs = las.header.parse_crs()
    if crs:
        pyproj_crs = crs.from_wkt(crs.to_wkt())
        epsg = pyproj_crs.to_epsg()
        if epsg is None:
            print(f"Use this to find your default EPSG code: {pyproj_crs.to_wkt()}")
            raise ValueError("No EPSG code found in LAZ header.")
        return epsg
    else:
        raise ValueError("No CRS found in LAZ header.")
    
def merge_laz_files(input_directory, output_laz):
    """
    Merge all LAZ files in the input directory into a single output LAZ file.
    If files have different CRS, reprojects all to the most common CRS before merging.
    """
    if os.path.exists(output_laz):
        os.remove(output_laz)

    laz_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory)
                 if (f.endswith('.laz') or f.endswith('.las')) and not f.endswith('.copc.las') and not f.endswith('.copc.laz')] 

    if not laz_files:
        raise ValueError("No LAZ files found in the input directory.")

    # Detect CRS of each file
    file_epsg = {}
    for f in laz_files:
        try:
            file_epsg[f] = get_laz_crs(f)
        except ValueError:
            file_epsg[f] = None

    valid_epsgs = [e for e in file_epsg.values() if e is not None]
    if not valid_epsgs:
        raise ValueError("No CRS found in any LAZ file.")

    # Use the most common EPSG as the merge target
    target_epsg = max(set(valid_epsgs), key=valid_epsgs.count)
    target_srs = f"EPSG:{target_epsg}"
    unique_crs = set(valid_epsgs)

    if len(unique_crs) > 1:
        print(f"CRS mismatch detected: {unique_crs}. Reprojecting all files to {target_srs} before merging.")
        # Build a tagged PDAL pipeline so each reader can be reprojected independently
        stages = []
        merge_inputs = []
        for i, f in enumerate(laz_files):
            reader_tag = f"reader_{i}"
            stages.append({"type": "readers.las", "filename": f, "tag": reader_tag})
            if file_epsg.get(f) != target_epsg:
                reproj_tag = f"reproj_{i}"
                stages.append({
                    "type": "filters.reprojection",
                    "out_srs": target_srs,
                    "inputs": [reader_tag],
                    "tag": reproj_tag
                })
                merge_inputs.append(reproj_tag)
            else:
                merge_inputs.append(reader_tag)
        stages.append({"type": "filters.merge", "inputs": merge_inputs})
        stages.append({"type": "writers.las", "filename": output_laz})
        pipeline = pdal.Pipeline(json.dumps({"pipeline": stages}))
    else:
        pipeline_steps = (
            [{"type": "readers.las", "filename": f} for f in laz_files]
            + [{"type": "filters.merge"}]
            + [{"type": "writers.las", "filename": output_laz}]
        )
        pipeline = pdal.Pipeline(json.dumps(pipeline_steps))

    count = pipeline.execute()
    print(f"Merged {len(laz_files)} files -> {output_laz} ({count:,} points)")

    # Cleanup copc files
    for f in laz_files:
        if f.endswith('.copc.las'):
            os.remove(f)

    return output_laz

def load_area_extent_geojson(geojson_path, target_crs):
    gdf = gpd.read_file(geojson_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf.total_bounds

def compute_circle_bounds(center, radius):
    x, y = center
    return (x - radius, y - radius, x + radius, y + radius)

def clip_laz_cli_or_pdal(bbox, input_laz, output_laz, poly=False):
    xmin, ymin, xmax, ymax = bbox
    bounds = f"([{xmin},{xmax}],[{ymin},{ymax}])"
    try:
        if poly:
            try:
                # PDAL crop expects WKT
                print(f'Attempting Polygon Crop with WKT...')
                pipeline_json = f"""
                [
                "{input_laz}",
                {{
                    "type": "filters.crop",
                    "polygon": "{poly}" 
                }},
                "{output_laz}"
                ]
                """
                # Validate JSON structure before running
                json.loads(pipeline_json) 
            except Exception as e:
                print(f'Error structuring polygon pipeline: {e}. Falling back to bounds.')
                pipeline_json = f"""
                [
                  "{input_laz}",
                  {{
                    "type": "filters.crop",
                    "bounds": "{bounds}"
                  }},
                  "{output_laz}"
                ]
                """
        else:
            pipeline_json = f"""
            [
              "{input_laz}",
              {{
                "type": "filters.crop",
                "bounds": "{bounds}"
              }},
              "{output_laz}"
            ]
            """

        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()
        print(f"Clipped LAZ written to: {output_laz}")

    except Exception as e:
        sys.stderr.write(f"PDAL clipping failed: {e}\n")

def fetch_osm_buildings(bbox, target_crs, circle=None):
    # Convert bbox to polygon and reproject to EPSG:4326 for OSM
    bbox_polygon = box(*bbox)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_polygon], crs=target_crs)
    bbox_gdf_wgs = bbox_gdf.to_crs(epsg=4326)
    west, south, east, north = bbox_gdf_wgs.total_bounds
    
    tags = {'building': True}
    gdf = ox.features_from_bbox((west, south, east, north), tags=tags)

    # Keep only Polygon and MultiPolygon geometries
    gdf = gdf[gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]

    # Reproject to target CRS
    gdf = gdf.to_crs(target_crs)
    gdf = gdf.reset_index() 

    # Optional spatial filtering by circle
    if circle:
        centroids = gdf.geometry.centroid
        gdf = gdf[centroids.within(circle)]

    # Clean and standardize properties
    gdf = gdf.reset_index(drop=True)
    gdf['fid'] = 1
    gdf['osm_id'] = gdf['id'].astype(str)
    gdf['osm_type'] = gdf['element']
    gdf['full_id'] = gdf['osm_type'].str[0] + gdf['osm_id']

    # Retain only selected fields
    keep_cols = ['fid', 'full_id', 'osm_id', 'osm_type', 'building', 'name', 'amenity', 'brand', 'wheelchair']
    for col in keep_cols:
        if col not in gdf.columns:
            gdf[col] = None
    gdf = gdf[keep_cols + ['geometry']]

    return gdf

def find_center_point_within_domain(gdf):
    if gdf.empty:
        return None
    unified = unary_union(gdf.geometry)
    center_point = unified.representative_point()
    center_point = Point(round(center_point.x, 2), round(center_point.y, 2))
    return center_point

def separate_laz_file(input_laz, output_dir=None):
    """
    Separate a LAZ file into ground (2) and building (6) classes.
    """
    las = laspy.read(input_laz)

    # Filter by classification
    ground_points = las.points[las.classification == 2]
    building_points = las.points[las.classification == 6]

    header = las.header
    base_name = os.path.splitext(os.path.basename(input_laz))[0]
    output_dir = output_dir or os.path.dirname(input_laz)

    ground_laz = os.path.join(output_dir, f"{base_name}_ground.laz")
    building_laz = os.path.join(output_dir, f"{base_name}_buildings.laz")

    # Save Ground
    with laspy.open(ground_laz, mode='w', header=header) as writer:
        writer.write_points(ground_points)
        print(f"Ground points saved to {ground_laz}")

    # Save Buildings
    with laspy.open(building_laz, mode='w', header=header) as writer:
        writer.write_points(building_points)
        print(f"Building points saved to {building_laz}")

    return ground_laz, building_laz

def main():
    DEFAULT_CRS = 'EPSG:25830'

    parser = argparse.ArgumentParser(description="Clip LAZ and fetch OSM buildings in same CRS")
    parser.add_argument('--input','-i', help="Input LAZ file")
    parser.add_argument('--input_dir','-id' ,help="Input directory with LAZ files")
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('xmin','ymin','xmax','ymax'), help="Bounding box (in target CRS)")
    parser.add_argument('--area_geojson', help="GeoJSON defining polygon area (any CRS)")
    parser.add_argument('--radius', '-r',type=float, help="Radius for circular clipping (map units of target CRS)")
    parser.add_argument('--bbox_bounding', type=float, default=750, help="Bounding box size for radius mode (default: 750)")
    parser.add_argument('--center', nargs=2, type=float, metavar=('x','y'), help="Center for circular clipping (in target CRS)")
    parser.add_argument('--center_latlon', type=str, help="Center in EPSG:4326 (lat, lon). Will be transformed to the LAZ CRS.")
    parser.add_argument('--crs', help=f"CRS for LAZ, defaults to header or {DEFAULT_CRS}")
    parser.add_argument('--output_dir','-o', required=True, help="Output directory")
    parser.add_argument('--output_filename', default='clipped', help="Output filename prefix")
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error('Specify --input or --input_dir')

    if not (args.bbox_bounding or args.area_geojson or args.radius):
        parser.error('Specify --bbox_bounding, --area_geojson, or --radius')

    # Create output directory
    os.makedirs(args.output_dir + '/output', exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, 'output')
    print(f"Output directory: {args.output_dir}")

    # Handle Input (Dir vs File)
    if args.input_dir:
        if os.path.isdir(args.input_dir):
            args.input = merge_laz_files(args.input_dir, os.path.join(args.output_dir, args.output_filename + '_merged.laz'))
        else:
            raise ValueError("Input directory does not exist.")
    
    # Determine CRS
    if args.crs:
        laz_crs = args.crs
    else:
        try:
            laz_crs = get_laz_crs(args.input)
        except ValueError:
            laz_crs = DEFAULT_CRS
            sys.stderr.write(f"Warning: using default CRS {DEFAULT_CRS}\n")
    print(f"LAZ CRS: {laz_crs}")

    # Transform center_latlon (EPSG:4326) to LAZ CRS if provided
    if args.center_latlon:
        lat, lon = map(float, args.center_latlon.split(','))
        transformer = Transformer.from_crs("EPSG:4326", laz_crs, always_xy=False)
        cx, cy = transformer.transform(lat, lon)
        args.center = [cx, cy]
        print(f"Transformed center ({lat}, {lon}) EPSG:4326 -> ({cx:.2f}, {cy:.2f}) {laz_crs}")
    # Determine bounds and optional circle
    circle = None
    circle_wkt = None

    # Set bbox to be a 1500 side square around center if radius mode and no bbox provided
    if args.bbox_bounding and not args.bbox:
        if args.center:
            center = tuple(args.center)
        else:
            las = laspy.read(args.input)
            dom = (las.header.min[0], las.header.min[1], las.header.max[0], las.header.max[1])
            center = ((dom[0]+dom[2]) / 2, (dom[1]+dom[3]) / 2)
        
        args.bbox = compute_circle_bounds(center, args.bbox_bounding) # 1500x1500 box
    
    if args.area_geojson:
        bounds = load_area_extent_geojson(args.area_geojson, laz_crs)
    elif args.bbox:
        bounds = args.bbox
        print(f"Using BBOX: {bounds}")
    else:
        # Radius Mode logic
        if args.center:
            center = tuple(args.center)
        else:
            if args.bbox:
                dom = args.bbox
            else:
                las = laspy.read(args.input)
                dom = (las.header.min[0], las.header.min[1], las.header.max[0], las.header.max[1])
            center = ((dom[0]+dom[2]) / 2, (dom[1]+dom[3]) / 2)
        
        bounds = compute_circle_bounds(center, args.radius)
        circle = Point(center).buffer(args.radius, resolution=8)
        circle_wkt = circle.wkt

    clipped_laz = os.path.join(args.output_dir, args.output_filename + '_merged.laz')
    buildings_geojson = os.path.join(args.output_dir, 'osm_buildings.geojson')
    polyprep_geojson = os.path.join(args.output_dir, 'osm_buildings_polyprep.geojson')

    # Clip LAZ
    clip_laz_cli_or_pdal(bounds, args.input, clipped_laz, circle_wkt)
    
    # Separate Ground/Building classes
    ground_path, building_path = separate_laz_file(clipped_laz, output_dir=args.output_dir)

    # Fetch & save OSM buildings
    buildings = fetch_osm_buildings(bounds, laz_crs, circle)
    buildings.to_file(buildings_geojson, driver='GeoJSON')
    print(f"OSM building footprints saved to {buildings_geojson}")

    # Find center point for Config
    center_point = find_center_point_within_domain(buildings)
    
    # Fallback if no buildings found to define center
    if center_point is None:
        cx = (bounds[0] + bounds[2]) / 2
        cy = (bounds[1] + bounds[3]) / 2
        center_point = Point(cx, cy)
        print(f"No buildings found. Using BBOX center: {center_point}")
    else:
        print(f"Center point derived from buildings: {center_point}")

    # Apply polyprep
    print("Applying polyprep to osm_buildings...")
    try:
        process_polygons(buildings_geojson, polyprep_geojson, buffer_size=3.0, apply_convex_hull=False, remove_holes=2, simplification_tol=1.5)
        print(f"Polyprep saved to {polyprep_geojson}")
    except Exception as e:
        print(f"Polyprep failed ({e}). Using raw OSM file for config.")
        polyprep_geojson = buildings_geojson

    # --- NEW: Generate CITY4CFD Configuration ---
    create_city4cfd_config(
        output_dir=args.output_dir,
        center_point=center_point,
        radius=args.radius,
        bounds=bounds,
        ground_laz=ground_path,
        building_laz=building_path,
        polygon_geojson=polyprep_geojson,
        output_name=args.output_filename
    )

if __name__ == '__main__':
    main()
