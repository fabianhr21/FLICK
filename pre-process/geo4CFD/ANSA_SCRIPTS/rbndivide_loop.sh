#!/bin/bash

input_dir="/home/fabianh/GEO_CASES/"
city="BARCELONA"
scripts_final="/home/fabianh/ANSA/Data/scripts_final/"

for d in $input_dir$city/*/ ; do
    # Ignore __pycache__ directories
    [[ "$d" == *"__pycache__"* ]] && continue
    # echo "Processing directory: $d"
    dirname=$(basename "$d")
    buildings_file="${d}output/${dirname}_Buildings.ansa"
    dimensions_file="${d}output/domain_dimensions.txt"
    echo "Using buildings file: $buildings_file"
    # echo "RUnning Path: '${d}output/'"
    ~/ANSA/BETA_CAE_Systems24.1/ansa_v24.1.2/ansa64.sh -nogui -noopencl -execscript "args_RbNGDivide.py|main('${buildings_file}','${dimensions_file}','${d}output/','${d}output/')"
done

# FOr makinf the precursor and split2hexa
#~/ANSA/BETA_CAE_Systems24.1/ansa_v24.1.2/ansa64.sh -nogui -noopencl -execscript "RbNDivide.py|main('/home/fabianh/GEO_CASES/BARCELONA/267-43/output/267-43_Buildings.ansa','267-43_Buildings','/home/fabianh/GEO_CASES/BARCELONA/267-43/output/')"
