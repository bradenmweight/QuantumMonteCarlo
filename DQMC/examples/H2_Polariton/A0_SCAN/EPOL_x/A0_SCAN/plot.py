import numpy as np
from matplotlib import pyplot as plt

R_LIST = np.arange( 0.0, 1.0+0.01, 0.01 )

TYPE = "Production" # "Equilibration", "Production"

E   = np.zeros( (len(R_LIST)) )
STD = np.zeros( (len(R_LIST)) )

for Ri,R in enumerate( R_LIST ):
    try:
        # if ( round(R,2) < 1.0 ):
        #     TMP = open("DATA_%s0/E_AVE_VAR_STD_%s.dat" % (str(round(R,2))[1:], TYPE),"r").readlines()
        # else:
        #     TMP = open("DATA_%1.1f0/E_AVE_VAR_STD_%s.dat" % (R,TYPE),"r").readlines()
        TMP = open("DATA_%1.2f/E_AVE_VAR_STD_%s.dat" % (R,TYPE),"r").readlines()
    except FileNotFoundError:
        TMP = [float("Nan")]*4
    E[Ri]   = float(TMP[1])
    STD[Ri] = float(TMP[3])

plt.plot( R_LIST[:], E[:], "-o", c="black", label="$\langle E \\rangle$" )
plt.plot( R_LIST[:], E[:]-STD[:], "_", c="red", alpha=0.75, label="$\pm\sqrt{\langle (\Delta E)^2 \\rangle}$" )
plt.plot( R_LIST[:], E[:]+STD[:], "_", c="red", alpha=0.75 )
plt.legend()
#plt.xlim(R_LIST[0],R_LIST[-1])
plt.xlabel("Coupling Strength, A0 (a.u.)",fontsize=15)
plt.ylabel("Energy (a.u.)",fontsize=15)
plt.tight_layout()
plt.savefig(f"PES_{TYPE}.jpg",dpi=400)
plt.clf()