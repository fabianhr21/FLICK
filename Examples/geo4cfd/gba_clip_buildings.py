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
import gc
import argparse
import subprocess
import osmnx as ox
from pyproj import CRS, Transformer
import geopandas as gpd
import pyarrow.parquet as pq
from shapely.geometry import shape, Point, box
from shapely.ops import unary_union
import laspy
import pdal
import json
import math
import requests

# Add polyprep to path - ensure this folder exists relative to script execution
sys.path.append('./polyprep')
from polyprep import process_polygons

# Define default CRS
DEFAULT_CRS = 'EPSG:4326'

GBA_BASE_URL = "https://data.source.coop/tge-labs/globalbuildingatlas-lod1/"
GBA_CACHE_DIR = "/home/fabianh/GEO_CASES/GBA_cache/"

# NOTE: Commented out for portability. valid PDAL environment usually handles this.
# os.environ['PDAL_DRIVER_PATH'] = '/home/fabianh/anaconda3/envs/qgis_env/bin/pdal'

def create_city4cfd_config(output_dir, center_point, radius,bounds, ground_laz, building_laz, polygon_geojson, output_name="mesh_mty_topo", influence_polygon=None):
    """
    Generates the config.json file for CITY4CFD using the paths and coordinates
    derived from the processing steps.

    If ``influence_polygon`` is given (shapely Polygon in the working CRS) its
    exterior ring is used as the reconstruction influence region instead of the
    axis-aligned bounding box.
    """

    # Ensure we use absolute paths for the ETL to avoid CWD issues
    abs_ground = os.path.abspath(ground_laz)
    abs_build = os.path.abspath(building_laz)
    abs_poly = os.path.abspath(polygon_geojson)

    # If radius wasn't provided (e.g. BBOX mode), default to 400 or calculate from bounds
    roi_radius = radius if radius else 400

    if influence_polygon is not None:
        # Drop the closing vertex — City4CFD closes the ring itself
        coords = list(influence_polygon.exterior.coords)
        if len(coords) > 1 and coords[0] == coords[-1]:
            coords = coords[:-1]
        poi = [[float(x), float(y)] for x, y in coords]
    else:
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
                "complexity_factor": 0.25,
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
        "bpg_domain_size": [10, 10, 10, 10],
        # "bpg_domain_size": [20, 30, 40, 20],
        "terrain_thinning": 80,
        "smooth_terrain": {
            "iterations": 5,
            "max_pts": 250000
        },
        "flat_terrain": False,
        "building_percentile": 90,
        "min_height": 3,
        "min_area": 50,
        "reconstruct_failed": False,
        "intersect_buildings_terrain": True,
        "edge_max_len": 5,
        "output_file_name": output_name,
        "output_format": "stl",
        "output_separately": False,
        "output_log": True,
        "log_file": "logFile.log"
    }

    config_path = os.path.join(output_dir, "config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"CITY4CFD configuration generated at: {config_path}")
    return config_path

def _extract_projected_epsg(pyproj_crs):
    """For compound CRS, find the projected sub-CRS and return its EPSG."""
    # Compound CRS (e.g. PROJCRS + vertical): iterate sub-CRS list
    if pyproj_crs.is_compound:
        for sub in pyproj_crs.sub_crs_list:
            if sub.is_projected:
                epsg = sub.to_epsg()
                if epsg:
                    return epsg
    return pyproj_crs.to_epsg()

def get_laz_crs(laz_path, default_crs=DEFAULT_CRS):
    # Read the header only — laspy.read() would decompress the whole point cloud
    with laspy.open(laz_path) as fh:
        crs = fh.header.parse_crs()
    crs = crs if crs else CRS.from_user_input(default_crs)
    if crs:
        pyproj_crs = CRS.from_wkt(crs.to_wkt())
        epsg = _extract_projected_epsg(pyproj_crs)
        if epsg is None:
            print(f"Use this to find your default EPSG code: {pyproj_crs.to_wkt()}")
            raise ValueError("No EPSG code found in LAZ header.")
        return epsg
    else:
        raise ValueError("No CRS found in LAZ header.")
    
def list_laz_files(input_directory):
    """Return the non-COPC LAZ/LAS files in a directory."""
    return [os.path.join(input_directory, f) for f in sorted(os.listdir(input_directory))
            if (f.endswith('.laz') or f.endswith('.las'))
            and not f.endswith('.copc.las') and not f.endswith('.copc.laz')]


def detect_target_epsg(laz_files, default_crs=DEFAULT_CRS):
    """Probe each file's header and pick the most common EPSG as the merge target.

    Returns (target_epsg, file_epsg, readable_files).
    """
    file_epsg = {}
    bad_files = []
    for f in laz_files:
        try:
            file_epsg[f] = get_laz_crs(f)
        except ValueError:
            # If no CRS found, use default and warn
            sys.stderr.write(f"Warning: No CRS found in {f}. Defaulting to {default_crs} for this file.\n")
            file_epsg[f] = CRS.from_user_input(default_crs).to_epsg()
        except Exception as e:
            # Corrupted/truncated file — skip it
            sys.stderr.write(f"Warning: Cannot read {f} (corrupt/truncated): {e}. Skipping.\n")
            bad_files.append(f)

    laz_files = [f for f in laz_files if f not in bad_files]
    if not laz_files:
        raise ValueError("All LAZ files are corrupt or unreadable.")

    valid_epsgs = [e for e in file_epsg.values() if e is not None]
    if not valid_epsgs:
        raise ValueError("No CRS found in any LAZ file.")

    target_epsg = max(set(valid_epsgs), key=valid_epsgs.count)
    return target_epsg, file_epsg, laz_files


def merge_laz_files(input_directory, output_laz, default_crs=DEFAULT_CRS,
                    target_epsg=None, crop_wkt=None):
    """
    Merge all LAZ files in the input directory into a single output LAZ file.
    If files have different CRS, reprojects all to the most common CRS before merging.

    ``crop_wkt`` (WKT string or list of WKT, expressed in the target CRS) crops
    every tile before the merge, so only the clip zone is ever held in memory.
    """
    if os.path.exists(output_laz):
        os.remove(output_laz)

    laz_files = list_laz_files(input_directory)
    if not laz_files:
        raise ValueError("No LAZ files found in the input directory.")

    detected_epsg, file_epsg, laz_files = detect_target_epsg(laz_files, default_crs)
    if target_epsg is None:
        target_epsg = detected_epsg
    target_srs = f"EPSG:{target_epsg}"

    unique_crs = set(e for e in file_epsg.values() if e is not None)
    if len(unique_crs) > 1:
        print(f"CRS mismatch detected: {unique_crs}. Reprojecting all files to {target_srs} before merging.")

    # Tagged pipeline: reader -> [reprojection] -> [crop] -> merge -> writer
    stages = []
    merge_inputs = []
    for i, f in enumerate(laz_files):
        tag = f"reader_{i}"
        stages.append({"type": "readers.las", "filename": f, "tag": tag})

        if file_epsg.get(f) != target_epsg:
            prev, tag = tag, f"reproj_{i}"
            stages.append({"type": "filters.reprojection", "out_srs": target_srs,
                           "inputs": [prev], "tag": tag})

        if crop_wkt:
            polygons = crop_wkt if isinstance(crop_wkt, (list, tuple)) else [crop_wkt]
            prev, tag = tag, f"crop_{i}"
            stages.append({"type": "filters.crop", "polygon": list(polygons),
                           "inputs": [prev], "tag": tag})

        merge_inputs.append(tag)

    stages.append({"type": "filters.merge", "inputs": merge_inputs})
    stages.append({"type": "writers.las", "filename": output_laz})

    pipeline = pdal.Pipeline(json.dumps({"pipeline": stages}))
    count = pipeline.execute()
    scope = "cropped + merged" if crop_wkt else "merged"
    print(f"{scope.capitalize()} {len(laz_files)} files -> {output_laz} ({count:,} points)")

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


def load_area_polygon_geojson(geojson_path, target_crs):
    """Load a GeoJSON clip zone and return (polygon, bounds) in ``target_crs``.

    All (Multi)Polygon features are dissolved into a single geometry, so the
    clip zone may be an arbitrary shape (e.g. a rotated rectangle) rather than
    an axis-aligned bounding box.
    """
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None:
        raise ValueError(f"{geojson_path} has no CRS; cannot reproject to {target_crs}.")
    gdf = gdf.to_crs(target_crs)

    gdf = gdf[gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]
    if gdf.empty:
        raise ValueError(f"No (Multi)Polygon geometry found in {geojson_path}.")

    polygon = unary_union(gdf.geometry)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    return polygon, tuple(polygon.bounds)


def polygon_to_wkt_list(polygon):
    """Return a list of Polygon WKT strings for PDAL filters.crop."""
    geoms = list(polygon.geoms) if polygon.geom_type == 'MultiPolygon' else [polygon]
    return [g.wkt for g in geoms]


def compute_circle_bounds(center, radius):
    x, y = center
    return (x - radius, y - radius, x + radius, y + radius)

def _run_crop_pipeline(input_laz, output_laz, crop_stage):
    """Execute reader -> filters.crop -> writer and return the point count."""
    pipeline = pdal.Pipeline(json.dumps([
        {"type": "readers.las", "filename": input_laz},
        crop_stage,
        {"type": "writers.las", "filename": output_laz},
    ]))
    return pipeline.execute()


def clip_laz_cli_or_pdal(bbox, input_laz, output_laz, poly=False):
    """Clip a LAZ to ``poly`` (WKT string or list of WKT) if given, else to ``bbox``.

    Falls back to the bounding box if the polygon crop fails or keeps no points.
    """
    xmin, ymin, xmax, ymax = bbox
    bounds_stage = {
        "type": "filters.crop",
        "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])",
    }

    if poly:
        polygons = poly if isinstance(poly, (list, tuple)) else [poly]
        print(f"Attempting polygon crop with {len(polygons)} WKT polygon(s)...")
        try:
            count = _run_crop_pipeline(input_laz, output_laz, {
                "type": "filters.crop",
                "polygon": list(polygons),
            })
            if count:
                print(f"Clipped LAZ written to: {output_laz} ({count:,} points)")
                return
            sys.stderr.write("Polygon crop kept 0 points. Falling back to bounds.\n")
        except Exception as e:
            sys.stderr.write(f"Polygon crop failed: {e}. Falling back to bounds.\n")

    try:
        count = _run_crop_pipeline(input_laz, output_laz, bounds_stage)
        print(f"Clipped LAZ written to: {output_laz} ({count:,} points)")
    except Exception as e:
        sys.stderr.write(f"PDAL clipping failed: {e}\n")

def fetch_osm_buildings(bbox, target_crs, clip_polygon=None):
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

    # Optional spatial filtering by the clip polygon (circle or GeoJSON zone)
    if clip_polygon is not None:
        centroids = gdf.geometry.centroid
        gdf = gdf[centroids.within(clip_polygon)]

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

def get_gba_tile_names(geom_wgs84):
    """Calculate GBA tile filenames covering the given WGS84 geometry bounds."""
    minx, miny, maxx, maxy = geom_wgs84.bounds
    lon_start = math.floor(minx / 5) * 5
    lon_end   = math.floor(maxx / 5) * 5
    lat_start = math.floor(miny / 5) * 5
    lat_end   = math.floor(maxy / 5) * 5
    tiles = []
    for lon in range(lon_start, lon_end + 1, 5):
        for lat in range(lat_start, lat_end + 1, 5):
            s_prefix = 'n' if lat     >= 0 else 's'; s_val = abs(lat)
            n_prefix = 'n' if lat + 5 >= 0 else 's'; n_val = abs(lat + 5)
            w_prefix = 'e' if lon     >= 0 else 'w'; w_val = abs(lon)
            e_prefix = 'e' if lon + 5 >= 0 else 'w'; e_val = abs(lon + 5)
            tiles.append(
                f"{w_prefix}{w_val:03d}_{n_prefix}{n_val:02d}_"
                f"{e_prefix}{e_val:03d}_{s_prefix}{s_val:02d}.parquet"
            )
    return tiles


def download_gba_tile(tile_name, cache_dir):
    """Download a GBA parquet tile to cache_dir. Returns local path or None."""
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, tile_name)
    if os.path.exists(local_path):
        print(f"   GBA tile cached: {tile_name}")
        return local_path
    url = f"{GBA_BASE_URL}{tile_name}"
    print(f"   Downloading GBA tile: {tile_name} ...")
    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        tmp = local_path + ".tmp"
        size = 0
        with open(tmp, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                size += len(chunk)
        os.rename(tmp, local_path)
        print(f"   -> Cached ({size / 1024**2:.1f} MB): {local_path}")
        return local_path
    except Exception as e:
        print(f"   Warning: Failed to download GBA tile {tile_name}: {e}")
        for p in [local_path + ".tmp", local_path]:
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass
        return None


def fetch_gba_buildings(bbox, target_crs, cache_dir, clip_polygon=None):
    """Fetch buildings from GBA tiles for bbox (in target_crs).

    Returns a GeoDataFrame in target_crs with a 'height' column,
    or an empty GeoDataFrame on failure.
    """
    import geopandas as gpd

    empty = gpd.GeoDataFrame(crs=target_crs, geometry=[])

    # Convert bbox to WGS84 for tile selection
    bbox_poly = box(*bbox)
    bbox_gdf  = gpd.GeoDataFrame(geometry=[bbox_poly], crs=target_crs)
    poly_wgs84 = bbox_gdf.to_crs("EPSG:4326").geometry.iloc[0]

    tile_names = get_gba_tile_names(poly_wgs84)
    frames = []
    for tile_name in tile_names:
        local_path = download_gba_tile(tile_name, cache_dir)
        if local_path is None:
            continue
        try:
            minx, miny, maxx, maxy = poly_wgs84.bounds
            # Only load geometry + height — skip all other columns
            schema = pq.read_schema(local_path)
            cols = ['geometry'] + (['height'] if 'height' in schema.names else [])
            # Try bbox-aware read (geopandas >= 1.0); fall back to post-load filter
            try:
                gdf = gpd.read_parquet(local_path, columns=cols,
                                       bbox=(minx, miny, maxx, maxy))
            except (TypeError, ValueError):
                gdf = gpd.read_parquet(local_path, columns=cols)
                if not gdf.empty:
                    gdf = gdf.cx[minx:maxx, miny:maxy].copy()
            if not gdf.empty:
                frames.append(gdf)
            del gdf
            gc.collect()
        except Exception as e:
            print(f"   Warning: Failed to read GBA tile {tile_name}: {e}")

    if not frames:
        return empty

    bldgs = gpd.GeoDataFrame(
        gpd.pd.concat(frames, ignore_index=True), geometry='geometry'
    )
    if bldgs.crs is None:
        bldgs = bldgs.set_crs("EPSG:4326")
    bldgs = bldgs.to_crs(target_crs)
    bldgs = bldgs[bldgs.geom_type.isin(['Polygon', 'MultiPolygon'])]

    if clip_polygon is not None:
        bldgs = bldgs[bldgs.geometry.centroid.within(clip_polygon)]

    # Standardise: keep height, add gid
    bldgs = bldgs.reset_index(drop=True)
    bldgs['gid'] = bldgs.index + 1
    # 'height' column is native in GBA tiles
    if 'height' not in bldgs.columns:
        bldgs['height'] = None

    return bldgs if not bldgs.empty else empty


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
    parser = argparse.ArgumentParser(description="Clip LAZ and fetch OSM buildings in same CRS")
    parser.add_argument('--input','-i', help="Input LAZ file")
    parser.add_argument('--input_dir','-id' ,help="Input directory with LAZ files")
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('xmin','ymin','xmax','ymax'), help="Bounding box (in target CRS)")
    parser.add_argument('--area_geojson', help="GeoJSON defining the clip polygon (any CRS). Takes precedence over --bbox/--bbox_bounding/--radius; the LAZ, the buildings and the City4CFD influence region all follow this polygon.")
    parser.add_argument('--radius', '-r',type=float, help="Radius for circular clipping (map units of target CRS)")
    parser.add_argument('--bbox_bounding', type=float, default=None, help="Half-size of the square bbox around the center, in map units (default: 750 when no --area_geojson/--bbox/--radius given)")
    parser.add_argument('--center', nargs=2, type=float, metavar=('x','y'), help="Center for circular clipping (in target CRS)")
    parser.add_argument('--center_latlon', type=str, help="Center in EPSG:4326 (lat, lon). Will be transformed to the LAZ CRS.")
    parser.add_argument('--crs', help=f"CRS for LAZ, defaults to header or {DEFAULT_CRS}")
    parser.add_argument('--output_dir','-o', required=True, help="Output directory")
    parser.add_argument('--output_filename', default='clipped', help="Output filename prefix")
    args = parser.parse_args()

    fallback_crs = args.crs if args.crs else DEFAULT_CRS

    if not args.input and not args.input_dir:
        parser.error('Specify --input or --input_dir')

    if not (args.bbox_bounding or args.area_geojson or args.radius or args.bbox):
        print("No clip zone given — defaulting to --bbox_bounding 750")
        args.bbox_bounding = 750.0

    if args.area_geojson and (args.bbox_bounding or args.bbox or args.radius):
        print("--area_geojson given: ignoring --bbox_bounding/--bbox/--radius")
        args.bbox_bounding = None
        args.bbox = None
        args.radius = None

    # Create output directory
    os.makedirs(args.output_dir + '/output', exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, 'output')
    print(f"Output directory: {args.output_dir}")

    # Determine CRS from the headers BEFORE merging, so the clip polygon can be
    # loaded early and applied to every tile as it is read.
    detected_epsg = None
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            raise ValueError("Input directory does not exist.")
        dir_files = list_laz_files(args.input_dir)
        if not dir_files:
            raise ValueError("No LAZ files found in the input directory.")
        try:
            detected_epsg = detect_target_epsg(dir_files, fallback_crs)[0]
        except (ValueError, Exception) as e:
            sys.stderr.write(f"Warning: CRS detection failed ({e}).\n")
    elif args.input:
        try:
            detected_epsg = get_laz_crs(args.input, fallback_crs)
        except (ValueError, Exception):
            detected_epsg = None

    # Determine CRS: prefer --crs arg, then header, then DEFAULT_CRS
    if args.crs:
        laz_crs = args.crs
    elif detected_epsg is not None:
        laz_crs = detected_epsg
    else:
        laz_crs = fallback_crs
        sys.stderr.write(f"Warning: using default CRS {fallback_crs}\n")
    print(f"LAZ CRS: {laz_crs}")

    # Load the clip polygon up front (needs laz_crs) so it can crop during the merge
    clip_polygon = None      # shapely geometry in laz_crs, used to filter buildings
    clip_wkt = None          # WKT (or list of WKT) handed to PDAL filters.crop
    influence_polygon = None # exterior ring written into the City4CFD config
    area_bounds = None

    if args.area_geojson:
        clip_polygon, area_bounds = load_area_polygon_geojson(args.area_geojson, laz_crs)
        clip_wkt = polygon_to_wkt_list(clip_polygon)
        # City4CFD influence_region needs a simple ring; use the convex hull for MultiPolygons
        influence_polygon = (clip_polygon if clip_polygon.geom_type == 'Polygon'
                             else clip_polygon.convex_hull)
        print(f"Using clip polygon from {args.area_geojson}")
        print(f"  bounds: {area_bounds}")
        print(f"  area:   {clip_polygon.area / 1e6:.3f} km2 "
              f"({len(influence_polygon.exterior.coords) - 1} vertices)")

    # Only pre-crop when the polygon CRS matches the CRS the merge will output in
    merge_crop_wkt = clip_wkt
    if clip_wkt and args.crs and detected_epsg is not None:
        try:
            forced_epsg = CRS.from_user_input(args.crs).to_epsg()
        except Exception:
            forced_epsg = None
        if forced_epsg != detected_epsg:
            sys.stderr.write(
                f"Warning: --crs {args.crs} differs from the LAZ headers (EPSG:{detected_epsg}); "
                "skipping the pre-merge crop.\n")
            merge_crop_wkt = None

    # Handle Input (Dir vs File)
    already_cropped = False
    if args.input_dir:
        args.input = merge_laz_files(
            args.input_dir,
            os.path.join(args.output_dir, args.output_filename + '_merged.laz'),
            default_crs=fallback_crs,
            crop_wkt=merge_crop_wkt,
        )
        already_cropped = bool(merge_crop_wkt)

    # Transform center_latlon (EPSG:4326) to LAZ CRS if provided
    if args.center_latlon:
        lat, lon = map(float, args.center_latlon.split(','))
        transformer = Transformer.from_crs("EPSG:4326", laz_crs, always_xy=False)
        cx, cy = transformer.transform(lat, lon)
        args.center = [cx, cy]
        print(f"Transformed center ({lat}, {lon}) EPSG:4326 -> ({cx:.2f}, {cy:.2f}) {laz_crs}")
    # Determine bounds (the clip polygon, if any, was already loaded above)

    # Set bbox to a square of side 2*bbox_bounding around center if no bbox provided
    if args.bbox_bounding and not args.bbox:
        if args.center:
            center = tuple(args.center)
        else:
            las = laspy.read(args.input)
            dom = (las.header.min[0], las.header.min[1], las.header.max[0], las.header.max[1])
            center = ((dom[0]+dom[2]) / 2, (dom[1]+dom[3]) / 2)

        args.bbox = compute_circle_bounds(center, args.bbox_bounding)

    if area_bounds is not None:
        bounds = area_bounds
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
        clip_polygon = Point(center).buffer(args.radius, resolution=8)
        clip_wkt = clip_polygon.wkt

    clipped_laz = os.path.join(args.output_dir, args.output_filename + '_clipped.laz')
    buildings_geojson = os.path.join(args.output_dir, 'osm_buildings.geojson')
    polyprep_geojson = os.path.join(args.output_dir, 'osm_buildings_polyprep.geojson')

    # Clip LAZ (never in-place: the merged file is the reader input).
    # Skipped when the merge already cropped every tile to the same polygon.
    if already_cropped:
        print(f"Merged file already cropped to the clip polygon: {args.input}")
        clipped_laz = args.input
    else:
        clip_laz_cli_or_pdal(bounds, args.input, clipped_laz, clip_wkt)
    
    # Separate Ground/Building classes
    ground_path, building_path = separate_laz_file(clipped_laz, output_dir=args.output_dir)

    # Fetch buildings: GBA first, fall back to OSM
    # clip_polygon (if any) keeps only footprints whose centroid falls inside the
    # clip zone, so buildings straddling the border stay whole instead of being cut.
    print("Fetching buildings from GBA...")
    buildings = fetch_gba_buildings(bounds, laz_crs, GBA_CACHE_DIR, clip_polygon)
    if buildings.empty:
        print("GBA returned no buildings — falling back to OSM.")
        buildings = fetch_osm_buildings(bounds, laz_crs, clip_polygon)
        print(f"OSM building footprints fetched: {len(buildings)} buildings")
    else:
        print(f"GBA building footprints fetched: {len(buildings)} buildings")
    buildings.to_file(buildings_geojson, driver='GeoJSON')
    print(f"Building footprints saved to {buildings_geojson}")

    # Find center point for Config
    center_point = find_center_point_within_domain(buildings)
    del buildings  # free before polyprep reloads the same data from file
    gc.collect()

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
        output_name=args.output_filename,
        influence_polygon=influence_polygon
    )

if __name__ == '__main__':
    main()
