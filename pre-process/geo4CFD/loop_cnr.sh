#!/bin/bash
## Code to loop over a desired directory and apply clipnbuildings
DIR="/home/fabianh/GEO_CASES/"
CITIES=("LONDRES") # "BARCELONA" "MADRID"
CITY4CFD="/home/fabianh/City4CFDlocal/new_build/"
ansa_path=/home/fabianh/ANSA/BETA_CAE_Systems24.1/ansa_v24.1.2/ansa64.sh
scripts_final="/home/fabianh/FLICK_untouched/pre-process/geo4CFD/ANSA_SCRIPTS/"
working_directory="/home/fabianh/FLICK_untouched/pre-process/geo4CFD/ANSA_SCRIPTS/"

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
declare -A center_dictionary=(
    ["2-20"]="43.3054136,-3.00593894"
    ["3-34"]="41.39319987,2.11873603"
    ["6-41"]="41.3392511,2.13141541"
    ["4-121"]="41.40273385,2.19038727"
    ["5-75"]="41.37545579,2.15485058"
    ["14-35"]="43.25423964,-2.93054075"
    ["30-382"]="40.40612014,-3.68054419"
    ["28-229"]="40.40575832,-3.73946573"
    ["33-326"]="40.46903807,-3.70477111"
    ["31-481"]="40.46037414,-3.64570761"
    ["32-691"]="40.44290096,-3.53941629"
    ["29-292"]="40.43293198,-3.71618378"
    ["45-74"]="37.3962905,-5.97463131"
    ["46-127"]="37.40642025,-5.92985035"
    ["47-23"]="39.45720149,-0.40732426"
    ["48-40"]="39.45694188,-0.39571264"
    ["49-783"]="41.6460981,-0.8675578"
    ["50-136"]="41.61369487,-1.07265142"
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

        if [[ -n "${center_dictionary[$BASENAME]}" ]]; then
            center_latlon="${center_dictionary[$BASENAME]}"
            echo "  Using center_latlon from dictionary: $center_latlon"
            python GBA_clipnbuildings_run.py -id "$f" --center_latlon $center_latlon --bbox_bounding $bbox_bounding -o "$f" --output_filename "$BASENAME" "${crs_arg[@]}"
        else
            python GBA_clipnbuildings_run.py -id "$f" --bbox_bounding $bbox_bounding -o "$f" --output_filename "$BASENAME" "${crs_arg[@]}"
        fi

        # Run City4CFD
        $CITY4CFD/city4cfd "$f/output/config.json" --output_dir "$f/output/"

        # Get average height / domain dimensions
        python generate_witness.py -stl "$f/output/${BASENAME}_Buildings.stl" -p "$f/output/" -o "$f/output/"

        buildings_file="${f}output/${BASENAME}_Buildings.stl"
        echo "  Using buildings file: $buildings_file"

        # # Copy and process .geo script
        cp "${scripts_final}script_gmsh_ParaPC_orden1.geo" "${f}output/${BASENAME}_Buildings.geo"
        $ansa_path -nogui -noopencl \
            -execscript "${working_directory}args_check_flat.py|main('${buildings_file}','${working_directory}','${f}output/','${f}output/')"

        # Copy p2/p3 templates into <case>/MN5/ and substitute all placeholders
        # Reads domain_dimensions.txt from $f/output/domain_dimensions.txt automatically
        echo "  Generating MN5 directory for: $BASENAME"
        python replace_templates.py "$f" "$BASENAME" "$templates_dir"

        echo "  Done with: $BASENAME"
        echo ""
        # break # Remove this break to process all cases in the city
    done
done