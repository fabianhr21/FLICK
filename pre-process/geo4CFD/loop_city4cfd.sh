#!/bin/bash
## COde to loop over a desrierd directory and apply clipnbuildings

DIR="/home/fabianh/GEO_CASES/"
CITIES=("BARCELONA")
CITY4CFD="/home/fabianh/City4CFDlocal/new_build/"
RADIUS=1000



for city in "${CITIES[@]}"; do
    d="$DIR$city"
    echo "Processing directory: $d"
    for f in $d/*/ ; do
        echo "  Found subdirectory: $f"
        $CITY4CFD/city4cfd $f/output/config.json  --output_dir $f/output/
    done   
done 