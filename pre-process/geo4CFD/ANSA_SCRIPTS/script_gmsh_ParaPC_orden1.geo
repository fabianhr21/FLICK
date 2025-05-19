Mesh.MshFileVersion = 2.2;
SetFactory("OpenCASCADE");

Mesh.CgnsConstructTopology = 1;

Mesh.CreateTopologyMsh2 = 1;

Mesh.SaveTopology = 1;

Merge "mesh_bcn_vol.cgns";

SetOrder 1;

Physical Surface("buildings",1) = {2};
Physical Surface("groundDomain",2) = {7};
Physical Surface("groundPrecursor",3) = {10};
Physical Surface("topDomain",4) = {5};
Physical Surface("topPrecursor",5) = {9};
Physical Surface("lateralDomainSouth",6) = {6};
Physical Surface("lateralDomainNorth",7) = {4};
Physical Surface("inlet",8) = {3};
Physical Surface("outlet",9) = {8};
Physical Surface("periodic",11) = {11,12,13,14};

Physical Volume("fluid",12) = {1};

Periodic Surface {13} = {11} Translate {-5000, 0, 0};
Periodic Surface {14} = {12} Translate {0, -1820.685888, 0};
