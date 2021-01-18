import petsc4py, sys

from petsc4py import PETSc
# number of nodes in each direction
# excluding those at the boundary
n = 32
# grid spacing
h = 1.0/(n+1) 
A = PETSc.Mat().create()
A.setSizes([n**3, n**3])
A.setType('python')
shell = Del2Mat(n) # shell context
A.setPythonContext(shell)
A.setUp()