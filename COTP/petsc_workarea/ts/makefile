#
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  SLEPc - Scalable Library for Eigenvalue Problem Computations
#  Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain
#
#  This file is part of SLEPc.
#  SLEPc is distributed under a 2-clause BSD license (see LICENSE).
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#
all: ex5

SLEPC_DIR=/usr/lib/slepcdir/slepc3.10/x86_64-linux-gnu-complex
PETSC_DIR=/usr/lib/petscdir/petsc3.10/x86_64-linux-gnu-complex

CFLAGS     =
FFLAGS     =
CPPFLAGS   =
FPPFLAGS   =
#LOCDIR     = 
EXAMPLESC  = ex5.c 
#EXAMPLESF  = ex1f.F ex1f90.F90 ex6f90.F90
#MANSEC     = EPS

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex5: ex5.o  chkopts
	-${CLINKER} -o ex5 ex5.o ${SLEPC_LIB}
	${RM} ex5.o

