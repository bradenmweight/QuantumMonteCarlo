import numpy as np
from matplotlib import pyplot as plt

R_LIST  = np.arange( 0.4, 6.05, 0.05 )
A0_LIST = np.arange( 0.0, 1.1, 0.1 )
NA0 = len(A0_LIST)
E_RUBIO_U22 = np.zeros( (len(R_LIST),NA0) ) # RUBIO U22
E_RUBIO_U21 = np.zeros( (len(R_LIST),NA0) ) # RUBIO U21

for A0i,A0 in enumerate( A0_LIST ):
    A0 = round(A0,2)
    try:
        TMP = np.loadtxt( f"../scQED_CCSD_U22_S2/A0_{A0}/GS_ENERGY.dat" )[:,1]
        E_RUBIO_U22[:len(TMP),A0i] = TMP
        E_RUBIO_U22[len(TMP):,A0i] = np.zeros( (len(R_LIST)-len(TMP)) ) * float("nan")
    except OSError:
        E_RUBIO_U22[:,A0i] = np.zeros( (len(R_LIST)) ) * float("nan")

    TMP = np.loadtxt( f"A0_{A0}/GS_ENERGY.dat" )[:,1]
    E_RUBIO_U21[:len(TMP),A0i] = TMP
    E_RUBIO_U21[len(TMP):,A0i] = np.zeros( (len(R_LIST)-len(TMP)) ) * float("nan")

ENERGY_DQMC = np.loadtxt( f"../../../A0_SCAN/EPOL_x/DATA_dt_0.01_0.01_NW_10_6_NSTEPS_2500_5000/PES_Production_WC_20.0.dat" )
STD_DQMC    = np.loadtxt( f"../../../A0_SCAN/EPOL_x/DATA_dt_0.01_0.01_NW_10_6_NSTEPS_2500_5000/PES_Production_WC_20.0_ERROR.dat" )
R_DQMC      = ENERGY_DQMC[:,0]
ENERGY_DQMC = ENERGY_DQMC[:,1:]
STD_DQMC = STD_DQMC[:,1:]

plt.plot( A0_LIST, E_RUBIO_U22[20,:NA0] - E_RUBIO_U22[20,0], "-o", label=f"Rubio scQED-CCSD-U22-S2 {round(R_LIST[48],2)} a.u." )
plt.plot( A0_LIST, E_RUBIO_U21[20,:NA0] - E_RUBIO_U21[20,0], "-o", label=f"Rubio scQED-CCSD-U21-S1 {round(R_LIST[48],2)} a.u." )
plt.plot( A0_LIST, ENERGY_DQMC[6,:NA0] - ENERGY_DQMC[6,0], "-o", label=f"Braden DQMC {round(R_DQMC[13],2)} a.u." )
plt.legend()
plt.savefig("A0_SCAN.jpg",dpi=400)

print( STD_DQMC[6,:NA0] )
STD_OUT = np.array([ np.sqrt( STD**2 + STD_DQMC[6,0]**2 ) for STD in STD_DQMC[6,:NA0] ])
np.savetxt("ENERGY.dat", np.c_[ A0_LIST, ENERGY_DQMC[6,:NA0] - ENERGY_DQMC[6,0], STD_OUT, E_RUBIO_U22[20,:NA0] - E_RUBIO_U22[20,0], E_RUBIO_U21[20,:NA0] - E_RUBIO_U21[20,0] ])



