# FUC
FUC - Fast modeling Urban Climate is a package developed in the Barcelona Supercomputing Center to model wind at urban scales using Neural Networks.
## Functionality
This repository serves as a guideline to generate urban climate wind analysis from Mesoscale Wheater Models and Urban Geometry.
## Pre-Process
In the Pre-Process directory, you can find scripts to prepare geometry from STL files to be processed by the Neural Network, the scripts are designed to obtain a georeferenced wind field and geometry at the end, so the best to use are BIM files which contain both geometry and georeference.

You can obtain BIM models from all of Catalunya on the following website:
https://geoportalcartografia.amb.cat/AppGeoportalCartografia2/index.html

## Wind-NN
The Wind-NN uses a surrogate model to predict wind behavior, the output is a normalized wind field by components. More info about the model in the wiki.

## Post-Process
In the Post-Process section you can find scripts to scale wind velocity and visualize.

