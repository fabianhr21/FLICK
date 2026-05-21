#!/bin/bash
# Code executor for the workflow
DIR="/home/fabianh/GEO_CASES/BARCELONA/test/"
CITY4CFD="/home/fabianh/City4CFDlocal/new_build/"
ansa_path=/home/fabianh/ANSA/BETA_CAE_Systems24.1/ansa_v24.1.2/ansa64.sh
scripts_final="ansa/"
working_directory="ansa/"

# Directory where the p2 and p3 template folders live
templates_dir="/home/fabianh/FLICK_untouched/pre-process/geo4CFD/MN5_TEMPLATES/"

bbox_bounding=750
CRS=""

# Parse optional --crs argument
while [[ $# -gt 0 ]]; do
    case "$1" in
        --crs) CRS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "Processing directory: $DIR"
for f in "$DIR"; do
    echo "  Found subdirectory: $f"
    BASENAME=$(basename "$f")
    # Run clipnbuildings
    crs_arg=()
    [[ -n "$CRS" ]] && crs_arg=(--crs "$CRS")

    if [[ -n "${center_dictionary[$BASENAME]}" ]]; then
        center_latlon="${center_dictionary[$BASENAME]}"
        echo "  Using center_latlon from dictionary: $center_latlon"
        python gba_clip_buildings.py -id "$f" --center_latlon $center_latlon --bbox_bounding $bbox_bounding -o "$f" --output_filename "$BASENAME" "${crs_arg[@]}"
    else
        python gba_clip_buildings.py -id "$f" --bbox_bounding $bbox_bounding -o "$f" --output_filename "$BASENAME" "${crs_arg[@]}"
    fi
    # Run City4CFD
    $CITY4CFD/city4cfd "$f/output/config.json" --output_dir "$f/output/"
    # Get average height / domain dimensions
    python witness.py -stl "$f/output/${BASENAME}_Buildings.stl" -p "$f/output/" -o "$f/output/"

    buildings_file="${f}output/${BASENAME}_Buildings.stl"
    # # Copy and process .geo script
    cp "${scripts_final}script_gmsh_ParaPC_orden1.geo" "${f}output/${BASENAME}_Buildings.geo"
    $ansa_path -nogui -noopencl \
        -execscript "${working_directory}args_check_flat.py|main('${buildings_file}','${working_directory}','${f}output/','${f}output/')"

    python replace_templates.py "$f" "$BASENAME" "$templates_dir"
done
