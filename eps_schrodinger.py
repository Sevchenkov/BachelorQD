import sys, slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
import numpy as np

opts=PETSc.Options()
n = opts.getInt('n', 30)
k = opts.getInt("k", 1)
V = opts.getInt('V', 1) #V=V(xn)

A = PETSc.Mat().create()
#A.create(PETSc.COMM_WORLD)
#A.setType('aij')
A.setSizes([n, n])
A.setFromOptions()
A.setUp()

rstart, rend = A.getOwnershipRange()

# Matrix Assembly
if rstart == 0:
    A[0, :2] = [2*k+V, -1*k]
    #rstart += 1
if rend == n:
    A[n-1, -2:] = [-1*k, 2*k+V]
    #rend -= 1
for i in range(rstart+1, rend-1):
    A[i, i-1:i+2] = [-1*k, 2*k+V, -1*k]


A.assemble()

E = SLEPc.EPS()
E.create()
E.setOperators(A)
#E.setProblemType(SLEPc.EPS.ProblemType.HEP)
E.setFromOptions()

E.solve()

Print = PETSc.Sys.Print

Print()
Print("******************************")
Print("********* Solution ***********")
Print("******************************")
Print()

its = E.getIterationNumber()
Print("Number of iterations of the method: %d" % its)

eps_type = E.getType()
Print("Solution method: %s" % eps_type)

nev, ncv, mpd = E.getDimensions()
Print("Number of requested eigenvalues: %d" % nev)

tol, maxit = E.getTolerances()
Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

nconv = E.getConverged()
Print("Number of converged eigenpairs %d" % nconv)

if nconv > 0:
    # Create the results vectors
    vr, wr = A.getVecs()
    vi, wi = A.getVecs()


    #
    Print()
    Print("        k          ||Ax-kx||/||kx|| ")
    Print("----------------- ------------------")
    for i in range(nconv):
        c = E.getEigenpair(i, vr, vi)
        error = E.computeError(i)
        if c.imag != 0.0:
            Print(" %9f%+9f j %12g" % (c.real, c.imag, error))
        else:
            Print(" %12f      %12g" % (c.real, error))
    Print()
    