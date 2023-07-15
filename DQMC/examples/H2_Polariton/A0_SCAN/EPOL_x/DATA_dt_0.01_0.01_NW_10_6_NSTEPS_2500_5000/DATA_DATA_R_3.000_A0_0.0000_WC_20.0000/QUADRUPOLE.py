import numpy as np
from matplotlib import pyplot as plt

RGRID, WFN_X = np.loadtxt( "WAVEFUNCTION_d0_N2_INT_True_Production.dat" ).T
_,     WFN_Y = np.loadtxt( "WAVEFUNCTION_d1_N2_INT_True_Production.dat" ).T
_,     WFN_Z = np.loadtxt( "WAVEFUNCTION_d2_N2_INT_True_Production.dat" ).T
dR           = RGRID[1] - RGRID[0]

X, Y, Z = RGRID, RGRID, RGRID
WFN = np.array( [WFN_X, WFN_Y, WFN_Z] )

NORM = np.einsum( "dx,dx->d", WFN[:,:], WFN[:,:] ) * dR
print( NORM )

EL_DIP    = np.zeros( (3) )
EL_DIP[:] = np.einsum( "r,dr->d", RGRID[:],  WFN[:,:]**2 ) * dR
print(EL_DIP)

# Q_XX = \int dr rho(r) * ( 3 X**2 - delta_{xy} )
EL_QUAD    = np.zeros( (6) )
#EL_QUAD    = np.einsum( "xy,dx,dy->d", np.outer(RGRID,RGRID)[:,:], WFN[:,:], WFN[:,:] ) * dR
RHO     = WFN**2
RHO_xyz = np.einsum( "ax,->", RHO[:,:] )

for xi in range( len(X) ):
    for yi in range( len(Y) ):
        for zi in range( len(Z) ):
            WFN_xyz = WFN[0,]
            EL_QUAD[0] += WFN[] # XX

print(EL_QUAD)


