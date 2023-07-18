import numpy as np
from matplotlib import pyplot as plt

R_LIST  = np.arange( 0.4, 6.05, 0.05 )
A0_LIST = np.arange( 0.0, 0.6, 0.1 )
ENERGY = np.zeros( (len(R_LIST),len(A0_LIST),2) ) # RUBIO, DQMC

for A0i,A0 in enumerate( A0_LIST ):
    A0 = round(A0,2)
    TMP = np.loadtxt( f"A0_{A0}/GS_ENERGY.dat" )[:,1]
    ENERGY[:len(TMP),A0i,0] = TMP
    ENERGY[len(TMP):,A0i,0] = np.zeros( (len(R_LIST)-len(TMP)) ) * float("nan")

ENERGY_DQMC = np.loadtxt( f"../../../A0_SCAN/EPOL_x/DATA_dt_0.01_0.01_NW_10_6_NSTEPS_2500_5000/PES_Production_WC_20.0.dat" )
R_DQMC      = ENERGY_DQMC[:,0]
ENERGY_DQMC = ENERGY_DQMC[:,1:]
    

#plt.plot( A0_LIST, ENERGY[20,:,0] - ENERGY[20,0,0], "-o", label=f"Rubio {round(R_LIST[48],2)} a.u." )
#plt.plot( A0_LIST, (ENERGY_DQMC[6,:6] - ENERGY_DQMC[6,0]), "-o", label=f"DQMC {round(R_DQMC[13],2)} a.u." )
plt.plot( A0_LIST[1:], (ENERGY_DQMC[6,1:6] - ENERGY_DQMC[6,0]) / (ENERGY[20,1:,0] - ENERGY[20,0,0]), "-o", label=f"Rubio / DQMC a.u." )
plt.plot( A0_LIST[1:], A0_LIST[1:]*0 + np.sqrt(3)/2, "-o", label=f"Rubio / DQMC a.u." )
plt.legend()
plt.savefig("A0_SCAN.jpg",dpi=400)




