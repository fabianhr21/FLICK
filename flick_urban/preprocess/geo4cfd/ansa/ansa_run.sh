#!/bin/bash

input_dir="/home/fabianh/GEO_CASES/"
city="BARCELONA"
scripts_final="/home/fabianh/ANSA/Data/scripts_final/"


~/ANSA/BETA_CAE_Systems24.1/ansa_v24.1.2/ansa64.sh -nogui -noopencl -execscript "args_check_flat.py|main('/home/fabianh/GEO_CASES/MADRID/652-227/652-227_Buildings.ansa','/home/fabianh/FLICK_untouched/pre-process/geo4CFD/ANSA_SCRIPTS/','/home/fabianh/GEO_CASES/MADRID/652-227/','/home/fabianh/GEO_CASES/MADRID/652-227/')"


# FOr makinf the precursor and split2hexa
#~/ANSA/BETA_CAE_Systems24.1/ansa_v24.1.2/ansa64.sh -nogui -noopencl -execscript "RbNDivide.py|main('/home/fabianh/GEO_CASES/BARCELONA/267-43/output/267-43_Buildings.ansa','267-43_Buildings','/home/fabianh/GEO_CASES/BARCELONA/267-43/output/')"
