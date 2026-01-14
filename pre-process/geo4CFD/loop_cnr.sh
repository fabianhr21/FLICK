#!/bin/bash
## COde to loop over a desrierd directory and apply clipnbuildings

DIR="/home/fabianh/GEO_CASES/"
CITIES=("MADRID" "VALENCIA" "ZARAGOZA")
RADIUS=1000

for city in "${CITIES[@]}"; do
    d="$DIR$city"
    echo "Processing directory: $d"
    for f in $d/*/ ; do
        echo "  Found subdirectory: $f"
        python clipnbuildings_run.py -id $f -r 1000 -o $f --output_filename $(basename $f)

    done
done    
