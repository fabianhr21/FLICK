#!/usr/bin/env python3
"""
Script: Clip a LAZ file to a defined area and extract OSM building footprints for that same extent,
ensuring both point cloud and footprints share the same CRS. Outputs footprints as GeoJSON.
Supports:
  - rectangular (bbox)
  - polygon GeoJSON
  - circular radius clipping (default center at domain midpoint, overrideable)

Default CRS: Mexico ITRF2008 / UTM zone 14N (EPSG:6369)

Dependencies:
  - QGIS (PyQGIS + Processing plugin)
  - LAStools (lasclip) or PDAL for LAZ clipping
  - OSMnx for downloading OSM footprints
  - GeoPandas, Shapely, Fiona, pyproj, laspy
"""
import os
import sys
sys.path.append('./polyprep')
from polyprep import process_polygons
import argparse
import subprocess

# Define default CRS and CLI binaries
DEFAULT_CRS = 'EPSG:4326'  # Mexico ITRF2008 / UTM 14N
PDAL_CMD = 'pdal'
os.environ['PDAL_DRIVER_PATH'] = '/home/fabianh/anaconda3/envs/qgis_env/bin/pdal'

# OSM and spatial libraries
import osmnx as ox
from pyproj import CRS
import geopandas as gpd
from shapely.geometry import shape, Point, box
import laspy
import pdal
import json

os.environ['PDAL_DRIVER_PATH'] = '/usr/lib/pdal'

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
    """
    # Erase output file if it exists
    if os.path.exists(output_laz):
        os.remove(output_laz)

    laz_files = [os.path.join(input_directory, f) for f in os.listdir(input_directory) if (f.endswith('.laz') or f.endswith('.las') and not ('.copc' in f))] 
    if not laz_files:
        raise ValueError("No LAZ files found in the input directory.")
    
    pipeline_steps = (
        [{"type": "readers.las", "filename": f} for f in laz_files]
        + [{"type": "filters.merge"}]      # concatenate the point streams
        + [{
            "type": "writers.las",
            "filename": output_laz
        }]
    )

    # ---- 3. run it ---------------------------------------------------------------
    pipeline = pdal.Pipeline(json.dumps(pipeline_steps))
    count = pipeline.execute()   # run! returns number of points written

    print(f"Merged {len(laz_files)} files → {output_laz} ({count:,} points)")
    # Remove .copc.las files if they exist
    for f in laz_files:
        if f.endswith('.copc.las'):
            os.remove(f)
    print(f"Removed .copc.las files from {input_directory}")    
        
    return output_laz


def load_area_extent_geojson(geojson_path, target_crs):
    """
    Loads a GeoJSON polygon via GeoPandas, reprojects to target_crs, returns bounds.
    """
    gdf = gpd.read_file(geojson_path)
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf.total_bounds

def compute_circle_bounds(center, radius):
    x, y = center
    return (x - radius, y - radius, x + radius, y + radius)


def clip_laz_with_processing(bbox, input_laz, output_laz):
    rect = QgsRectangle(*bbox)
    processing.run("lastools:lasclip", {
        'INPUT': input_laz,
        'BOX': rect,
        'OUTPUT': output_laz
    })


def clip_laz_cli_or_pdal(bbox, input_laz, output_laz, poly=False):
    xmin, ymin, xmax, ymax = bbox
    bounds = f"([{xmin},{xmax}],[{ymin},{ymax}])"
    try:
        if poly:
            try:
                print('Trying to create polygon')
                # Usamos un polígono WKT
                pipeline_json = f"""
                [
                "{input_laz}",
                {{
                    "type": "filters.crop",
                    "polygon": {repr(poly)}
                }},
                "{output_laz}"
                ]
                """
                print(json.loads(pipeline_json) )
                print('hasta aqui 2')
            except:
                print('Error creating polygon, using bounds instead')
                # Usamos un recorte rectangular con bounds
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
            # Usamos un recorte rectangular con bounds
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


# def fetch_osm_buildings(bbox, target_crs, circle=None):
#     """
#     bbox: (xmin, ymin, xmax, ymax) in **projected CRS** (e.g. EPSG:32614)
#     target_crs: final CRS for output buildings (usually same afrom shapely.geometry import boxs bbox)
#     circle: shapely.geometry.Polygon in same CRS as bbox (optional)
#     """
#     # Convert bbox to polygon and reproject to EPSG:4326
#     bbox_polygon = box(*bbox)
#     bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_polygon], crs=target_crs)
#     bbox_gdf_wgs = bbox_gdf.to_crs(epsg=4326)
#     # Extract reprojected bounds: (west, south, east, north)
#     west, south, east, north = bbox_gdf_wgs.total_bounds
#     bbox_tuple = (west, south, east, north)
#     # Download buildings from OSM
#     tags = {'building': True}
#     gdf = ox.features_from_bbox(bbox_tuple, tags=tags)
#     # Keep only polygon geometries
#     gdf = gdf[gdf.geom_type.isin(['Polygon'])]
#     # Reproject to target_crs
#     gdf = gdf.to_crs(target_crs)
#     # Optional: filter by circle
#     if circle:
#         centroids = gdf.geometry.centroid
#         gdf = gdf[centroids.within(circle)]
#     if gdf.empty:
#         print("Warning: No geometries found after OSM fetch and filtering.")

#     # Converto to MultiPolygon
#     # gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom if geom.geom_type == 'MultiPolygon' else shape(geom).buffer(0))
#     return gdf


def fetch_osm_buildings(bbox, target_crs, circle=None):
    """
    bbox: (xmin, ymin, xmax, ymax) in projected CRS (e.g. EPSG:32614)
    target_crs: output CRS (usually same as bbox)
    circle: shapely.geometry.Polygon to spatially filter buildings (optional)
    """
    # Convert bbox to polygon and reproject to EPSG:4326 for OSM
    bbox_polygon = box(*bbox)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_polygon], crs=target_crs)
    bbox_gdf_wgs = bbox_gdf.to_crs(epsg=4326)
    west, south, east, north = bbox_gdf_wgs.total_bounds
    bbox_tuple = (west, south, east, north)
    # Download building footprints from OSM
    tags = {'building': True}
    gdf = ox.features_from_bbox(bbox_tuple, tags=tags)

    # Keep only Polygon and MultiPolygon geometries
    gdf = gdf[gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]

    # Reproject to target CRS
    gdf = gdf.to_crs(target_crs)
    gdf = gdf.reset_index() 

    # Optional spatial filtering by circle
    if circle:
        centroids = gdf.geometry.centroid
        gdf = gdf[centroids.within(circle)]
    # print("Head: ", gdf.columns)
    # print("Head: ", gdf.head())

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

def separate_laz_file(input_laz, output_dir=None):
    """
    Separate a LAZ file into ground and building classes.
    Saves new LAZ files with filtered points.
    """

    las = laspy.read(input_laz)

    # Filtrar por clasificación
    ground_points = las.points[las.classification == 2]
    building_points = las.points[las.classification == 6]

    # Copiar el header original
    header = las.header

    # Determinar nombres de salida
    base_name = os.path.splitext(os.path.basename(input_laz))[0]
    output_dir = output_dir or os.path.dirname(input_laz)

    ground_laz = os.path.join(output_dir, f"{base_name}_ground.laz")
    building_laz = os.path.join(output_dir, f"{base_name}_buildings.laz")

    # Guardar puntos ground
    with laspy.open(ground_laz, mode='w', header=header) as writer:
        writer.write_points(ground_points)
        print(f"Ground points saved to {ground_laz}")

    # Guardar puntos buildings
    with laspy.open(building_laz, mode='w', header=header) as writer:
        writer.write_points(building_points)
        print(f"Building points saved to {building_laz}")

    return ground_laz, building_laz



def main():
    DEFAULT_CRS = 'EPSG:6369'  # Mexico ITRF2008 / UTM 14N

    parser = argparse.ArgumentParser(description="Clip LAZ and fetch OSM buildings in same CRS")
    parser.add_argument('--input','-i', help="Input LAZ file")
    parser.add_argument('--input_dir','-id' ,help="Input directory with LAZ files")
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('xmin','ymin','xmax','ymax'), help="Bounding box (in target CRS)")
    parser.add_argument('--area_geojson', help="GeoJSON defining polygon area (any CRS)")
    parser.add_argument('--radius', '-r',type=float, help="Radius for circular clipping (map units of target CRS)")
    parser.add_argument('--center', nargs=2, type=float, metavar=('x','y'), help="Center for circular clipping (in target CRS)")
    parser.add_argument('--crs', help=f"CRS for LAZ, defaults to header or {DEFAULT_CRS}")
    parser.add_argument('--output_dir','-o', required=True, help="Output directory")
    parser.add_argument('--output_filename', default='clipped', help="Output filename prefix")
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error('Specify --input or --input_dir')

    if not (args.bbox or args.area_geojson or args.radius):
        parser.error('Specify --bbox, --area_geojson, or --radius')

    # Create output directory
    os.makedirs(args.output_dir + '/output', exist_ok=True)
    args.output_dir = os.path.join(args.output_dir, 'output')
    print(f"Output directory: {args.output_dir}")

        # Check if input is a directory or file
    if args.input_dir:
        if os.path.isdir(args.input_dir):
            laz_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if (f.endswith('.laz') or f.endswith('.las'))]
            if not laz_files:
                raise ValueError("No LAZ files found in the input directory.")
            # Merge LAZ files
            args.input = merge_laz_files(args.input_dir, os.path.join(args.output_dir, args.output_filename + '_merged.laz'))
        else:
            raise ValueError("Input directory does not exist or is not a directory.")
    # args.input = os.path.join(args.output_dir, args.output_filename + '_merged.laz')


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

    # Determine bounds and optional circle
    circle = None
    if args.area_geojson:
        bounds = load_area_extent_geojson(args.area_geojson, laz_crs)
    elif args.bbox:
        bounds = args.bbox
    else:
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

    # Clip LAZ with fallback
    clip_laz_cli_or_pdal(bounds, args.input, clipped_laz, circle_wkt)
    separate_laz_file(clipped_laz, output_dir=args.output_dir)

    # Fetch & save OSM buildings
    buildings = fetch_osm_buildings(bounds, laz_crs, circle)
    buildings.to_file(buildings_geojson, driver='GeoJSON')
    print(f"OSM building footprints saved to {buildings_geojson}")

    # Applying polyprep to osm_buildings
    print("Applying polyprep to osm_buildings with arbitray parameters...")
    process_polygons(buildings_geojson, os.path.join(args.output_dir, 'osm_buildings_polyprep.geojson'), buffer_size=2.0, apply_convex_hull=False,remove_holes=2, simplification_tol=0.5)
    print(f"Polyprep applied to osm_buildings, saving to {os.path.join(args.output_dir, 'osm_buildings_polyprep.geojson')}")

if __name__ == '__main__':
    main()

