#!/bin/bash
## Code to loop over a desired directory and apply clipnbuildings
DIR="/home/fabianh/ANSA/CASES_MESHES/"
CITIES=("BARCELONA") # "MADRID" "VALENCIA" "SEVILLA" "ZARAGOZA")
CITY4CFD="/home/fabianh/City4CFD/build/"
ansa_path=/apps/ANSA/24.1.2/ansa_v24.1.2/ansa64.sh
scripts_final="/home/fabianh/FLICK/pre-process/geo4CFD/ANSA_SCRIPTS/"
working_directory="/home/fabianh/FLICK/pre-process/geo4CFD/ANSA_SCRIPTS/"

# Directory where the p2 and p3 template folders live
templates_dir="/home/fabianh/FLICK/pre-process/geo4CFD/MN5_TEMPLATES/"

RADIUS=1500
for city in "${CITIES[@]}"; do
    d="$DIR$city"
    echo "Processing directory: $d"
    for f in "$d"/*/; do
        echo "  Found subdirectory: $f"
        
        BASENAME=$(basename "$f")

        # Run clipnbuildings
        python clipnbuildings_run.py -id "$f" -r $RADIUS -o "$f" --output_filename "$BASENAME"

        # Run City4CFD
        $CITY4CFD/city4cfd "$f/output/config.json" --output_dir "$f/output/"

        # Get average height / domain dimensions
        python generate_witness.py -stl "$f/output/${BASENAME}_Buildings.stl" -p "$f/output/"

        buildings_file="${f}output/${BASENAME}_Buildings.stl"
        echo "  Using buildings file: $buildings_file"

        # Copy and process .geo script
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
# #!/bin/bash
# ## COde to loop over a desrierd directory and apply clipnbuildings

# DIR="/home/fabianh/ANSA/CASES_MESHES/"
# CITIES=("BARCELONA" "MADRID" "VALENCIA" "SEVILLA" "ZARAGOZA")
# CITY4CFD="/home/fabianh/City4CFD/build/"
# ansa_path=/apps/ANSA/24.1.2/ansa_v24.1.2/ansa64.sh
# scripts_final="/home/fabianh/FLICK/pre-process/geo4CFD/ANSA_SCRIPTS/"
# working_directory="/home/fabianh/FLICK/pre-process/geo4CFD/ANSA_SCRIPTS/"
# RADIUS=1500

# for city in "${CITIES[@]}"; do
#     d="$DIR$city"
#     echo "Processing directory: $d"
#     for f in $d/*/ ; do
#         echo "  Found subdirectory: $f"
#         python clipnbuildings_run.py -id $f -r $RADIUS -o $f --output_filename $(basename $f)     # Run clipnbuildings
#         $CITY4CFD/city4cfd $f/output/config.json  --output_dir $f/output/                       # Run City4CFD
#         python generate_witness.py -stl $f/output/$(basename $f)_Buildings.stl -p $f/output/    # Get average height
#         buildings_file="${f}output/$(basename $f)_Buildings.stl"
#         echo "Using buildings file: $buildings_file"
#         cp "${scripts_final}script_gmsh_ParaPC_orden1.geo" "${f}output/$(basename $f)_Buildings.geo"
#         $ansa_path -nogui -noopencl -execscript "${working_directory}args_check_flat.py|main('${buildings_file}','${working_directory}','${f}output/','${f}output/')"

#     done
# done    
# # python clipnbuildings_run.py -id /home/fabianh/ANSA/CASES_MESHES/BARCELONA/275-76  -r 1000 -o /home/fabianh/ANSA/CASES_MESHES/BARCELONA/275-76/test/ --output_filename test 