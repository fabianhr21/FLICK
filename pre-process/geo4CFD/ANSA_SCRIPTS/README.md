# SCRIPTS for preparing StereoLithography geometry in ANSA for CFD 
## ALL FILES FROM THE REPO SHOULD BE IN THE SAME DIRECTORY BEFORE OPENING ANSA AND WHEN RUNNING ANY CODE. ##

## ORDER: check_flat >> RbNDivide >> script_gmsh
## Change CreateDomain.py from your ANSA>Scripts>CFD to add new features.

### FOR MESHING ###

After check flat you should add the meshing and volume scenarios and assign the PIDs of your domains to BATCH MESH.
Then revise the mesh and fix the mesh as necessary.

### IMPORTANT ###
After meshing and confirming quality, you should merge the bottom pids to the corresponding sides and the roofs to the pid1.


Finally run the RbNDivide
