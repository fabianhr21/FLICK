#!/bin/bash

input_dir="${ANSA_CASES_DIR:-/home/fabianh/ANSA/CASES_MESHES/}"
working_dir="${ANSA_SCRIPTS_DIR:-/home/fabianh/FLICK/pre-process/geo4CFD/ANSA_SCRIPTS/}"
cities=("BARCELONA")
scripts_final="./"
ansa_path="${ANSA_EXEC:-/apps/ANSA/24.1.2/ansa_v24.1.2/ansa64.sh}"

for city in "${cities[@]}"; do
    echo "Processing city: $city"
    for d in $input_dir$city/*/ ; do
        # Ignore __pycache__ directories
        [[ "$d" == *"__pycache__"* ]] && continue
        # echo "Processing directory: $d"
        dirname=$(basename "$d")
        buildings_file="${d}output/${dirname}_Buildings.stl"
        echo "Using buildings file: $buildings_file"
        # Copy geo file from scripts_final
        cp "${scripts_final}script_gmsh_ParaPC_orden1.geo" "${d}output/${dirname}_Buildings.geo"
        # echo "RUnning Path: '${d}output/'"
        $ansa_path -nogui -noopencl -execscript "args_check_flat.py|main('${buildings_file}','${working_dir}','${d}output/','${d}output/')"
        # wait until the process is done
        #wait
    done
done

# /apps/ANSA/24.1.2/ansa_v24.1.2/ansa64.sh -nogui -noopencl -execscript "args_check_flat.py|main('/home/fabianh/ANSA/CASES_MESHES/MADRID/652-227/output/652-227_Buildings.ansa','/home/fabianh/FLICK/pre-process/geo4CFD/ANSA_SCRIPTS/','/home/fabianh/ANSA/CASES_MESHES/MADRID/652-227/output/','/home/fabianh/ANSA/CASES_MESHES/MADRID/652-227/output/')"
