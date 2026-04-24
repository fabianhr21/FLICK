#!/bin/bash
## COde to loop over a desrierd directory and apply clipnbuildings

DIR="/home/fabianh/ANSA/CASES_MESHES/"
CITIES=("BARCELONA"  "MADRID"  "SEVILLA"  "VALENCIA"  "ZARAGOZA")
CITY4CFD="/home/fabianh/City4CFD/build/"
RADIUS=1000



for city in "${CITIES[@]}"; do
    d="$DIR$city"
    echo "Processing directory: $d"
    for f in $d/*/ ; do
        echo "  Found subdirectory: $f"
        $CITY4CFD/city4cfd $f/output/config.json  --output_dir $f/output/
    done   
done 

# /home/fabianh/ANSA/CASES_MESHES/MADRID/652-227/output/config.json