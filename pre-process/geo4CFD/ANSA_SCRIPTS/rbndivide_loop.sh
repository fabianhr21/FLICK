#!/bin/bash

input_dir="/home/fabianh/ANSA/CASES_MESHES/"
city="BARCELONA"
scripts_final="/home/fabianh/FLICK/pre-process/geo4CFD/ANSA_SCRIPTS/"

for d in $input_dir$city/*/ ; do
    # Ignore __pycache__ directories
    [[ "$d" == *"__pycache__"* ]] && continue
    # echo "Processing directory: $d"
    dirname=$(basename "$d")
    buildings_file="${d}output/${dirname}_Buildings.ansa"
    echo "Using buildings file: $buildings_file"
    # echo "RUnning Path: '${d}output/'"
    echo "Arguments: '${buildings_file}','${dirname}_Buildings','${d}output/'"
    /apps/ANSA/24.1.2/ansa_v24.1.2/ansa64.sh -nogui -noopencl -execscript "args_RbNDivide.py|main('${buildings_file}','${dirname}_Buildings','${d}output/')"
done

# FOr makinf the precursor and split2hexa
#~/ANSA/BETA_CAE_Systems24.1/ansa_v24.1.2/ansa64.sh -nogui -noopencl -execscript "RbNDivide.py|main('/home/fabianh/GEO_CASES/BARCELONA/267-43/output/267-43_Buildings.ansa','267-43_Buildings','/home/fabianh/GEO_CASES/BARCELONA/267-43/output/')"
