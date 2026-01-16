#!/bin/bash
## COde to loop over a desrierd directory and apply clipnbuildings

DIR="/home/fabianh/ANSA/CASES_MESHES/"
CITIES=("MADRID" "BARCELONA"  "SEVILLA"  "VALENCIA"  "ZARAGOZA")
CITY4CFD="/home/fabianh/City4CFD/build/"
RADIUS=1000

for city in "${CITIES[@]}"; do
    d="$DIR$city"
    echo "Processing directory: $d"
    for f in $d/*/ ; do
        echo "  Found subdirectory: $f"
        python clipnbuildings_run.py -id $f -r 1000 -o $f --output_filename $(basename $f)
        $CITY4CFD/city4cfd $f/output/config.json  --output_dir $f/output/

    done
done    
