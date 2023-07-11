import sys
import numpy
from pyscf import gto, scf, cc
from time import time

Rlist = numpy.arange(0.1, 6.1, 0.1)
ENERGY = numpy.zeros( (len(Rlist), 2) ) # HF, CCSD
for Ri,R in enumerate( Rlist ):
    mol = gto.M(
        atom = 'H 0 0 0; H 0 0 ' + str(R),  # in au
        basis = 'cc-pVQZ',
        unit = 'Bohr',
        symmetry = False,
    )
    mol.build()
    myHF   = scf.RHF(mol).run()
    myCCSD = cc.ccsd.CCSD( myHF ).run()
    ENERGY[Ri,0] = myHF.e_tot
    ENERGY[Ri,1] = myCCSD.e_tot

numpy.savetxt("GS_ENERGY.dat", numpy.c_[Rlist[:], ENERGY[:,:]], fmt="%1.6f", header="R\tHF\tCCSD [cc-pVQZ]")
