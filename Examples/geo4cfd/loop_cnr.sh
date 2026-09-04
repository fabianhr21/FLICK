#!/bin/bash
# Code executor for the workflow
shopt -s nullglob # skip case-folder loops cleanly instead of using a literal unmatched glob
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # FLICK repo root (Examples/geo4cfd -> FLICK)
CLIPNBUILDINGS="$SCRIPT_DIR/gba_clip_buildings.py"
DIR="${FLICK_DATA_DIR:-$REPO_ROOT}/"           # base dir holding the city subfolders; override with FLICK_DATA_DIR
CITIES=("test") # TODO: set to the city subfolder names under $DIR
CITY4CFD="$REPO_ROOT/City4CFD/build"
ansa_path="${ANSA_EXEC:-path/to/ansa/exec}"    # set ANSA_EXEC to your ANSA executable
working_directory="$REPO_ROOT"

# Directory where the p2 and p3 template folders live
templates_dir="$REPO_ROOT/flick_urban/geo4cfd/templates/"

bbox_bounding=500
CRS=""
AREA_GEOJSON=""

# Parse optional arguments
#   --crs EPSG:xxxxx    force the CRS of the LiDAR
#   --geojson FILE      clip zone polygon; overrides the dictionaries below
while [[ $# -gt 0 ]]; do
    case "$1" in
        --crs) CRS="$2"; shift 2 ;;
        --geojson) AREA_GEOJSON="$2"; shift 2 ;;
        *) shift ;;
    esac
done

# Clip-zone polygons. Key is either a case basename or a city name; a case key
# wins over a city key. When a case resolves to a polygon the LiDAR is cropped
# to that exact shape and --bbox_bounding is ignored.
declare -A geojson_dictionary=(
    ["BARCELONA"]="/path/to/polygon.geojson"
)
declare -A center_dictionary=(
    ["2-20"]="43.3054136,-3.00593894" # Example "CASE" - Coordinates
)

for city in "${CITIES[@]}"; do
    d="$DIR$city"
    echo "Processing directory: $d"
    for f in "$d"/*/; do
        echo "  Found subdirectory: $f"
        BASENAME=$(basename "$f")
        # Run clipnbuildings
        crs_arg=()
        [[ -n "$CRS" ]] && crs_arg=(--crs "$CRS")

        # Resolve the clip zone: CLI flag > per-case file > case key > city key
        area_geojson="$AREA_GEOJSON"
        if [[ -z "$area_geojson" && -f "${f}clip_zone.geojson" ]]; then
            area_geojson="${f}clip_zone.geojson"
        fi
        if [[ -z "$area_geojson" && -n "${geojson_dictionary[$BASENAME]}" ]]; then
            area_geojson="${geojson_dictionary[$BASENAME]}"
        fi
        if [[ -z "$area_geojson" && -n "${geojson_dictionary[$city]}" ]]; then
            area_geojson="${geojson_dictionary[$city]}"
        fi

        if [[ -n "$area_geojson" ]]; then
            if [[ ! -f "$area_geojson" ]]; then
                echo "  ERROR: clip zone not found: $area_geojson" >&2
                exit 1
            fi
            echo "  Using clip polygon: $area_geojson"
            python "$CLIPNBUILDINGS" -id "$f" --area_geojson "$area_geojson" -o "$f" --output_filename "$BASENAME" "${crs_arg[@]}"
        elif [[ -n "${center_dictionary[$BASENAME]}" ]]; then
            center_latlon="${center_dictionary[$BASENAME]}"
            echo "  Using center_latlon from dictionary: $center_latlon"
            python "$CLIPNBUILDINGS" -id "$f" --center_latlon $center_latlon --bbox_bounding $bbox_bounding -o "$f" --output_filename "$BASENAME" "${crs_arg[@]}"
        else
            python "$CLIPNBUILDINGS" -id "$f" --bbox_bounding $bbox_bounding -o "$f" --output_filename "$BASENAME" "${crs_arg[@]}"
        fi

    # Run City4CFD
    $CITY4CFD/city4cfd "$f/output/config.json" --output_dir "$f/output/"
    done
done
    # # Get average height / domain dimensions
    # python "$SCRIPT_DIR/witness.py" -stl "$f/output/${BASENAME}_Buildings.stl" -p "$f/output/" -o "$f/output/"

    # buildings_file="${f}output/${BASENAME}_Buildings.stl"
    # # # Copy and process .geo script
    # cp "${working_directory}script_gmsh_ParaPC_orden1.geo" "${f}output/${BASENAME}_Buildings.geo"
    # $ansa_path -nogui -noopencl \
    #     -execscript "${working_directory}args_check_flat.py|main('${buildings_file}','${working_directory}','${f}output/','${f}output/')"

    # python "$SCRIPT_DIR/replace_templates.py" "$f" "$BASENAME" "$templates_dir"