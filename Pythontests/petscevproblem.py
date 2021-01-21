from petsc4py import PETSc

n=32
h=1.0/(n+1)
A=PETSc.Mat().create()
A.setSizes([n**3, n**3])
A.setType("python")
A.setPythonContext()
A.setUp()