import pyAlya

CASESTR = 'cyl'
BASEDIR = 'examples/data/sod2d/'

mesh   = pyAlya.MeshSOD2D.read(CASESTR, basedir=BASEDIR)
bc,_,_ = mesh.extract_bc([1])
norms  = bc.computeNormals()
field  = pyAlya.FieldSOD2D(xyz=bc.xyz, ptable=bc.partition_table, normals=norms)

bc.write('out_norms')
field.write('out_norms')