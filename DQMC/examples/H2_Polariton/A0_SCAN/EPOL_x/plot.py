import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

R_LIST  = np.arange( 0.2, 6.0+0.2, 0.2 )
A0_LIST = np.arange( 0.0, 1.0+0.1, 0.1 )

R_LIST = np.append( R_LIST, 30.0 )

TYPE = "Production" # "Equilibration", "Production"

E   = np.zeros( (len(R_LIST), len(A0_LIST)) )
STD = np.zeros( (len(R_LIST), len(A0_LIST)) )

for Ri,R in enumerate( R_LIST ):
    for A0i,A0 in enumerate( A0_LIST ):
        try:
            TMP = open("DATA_DATA_R_%1.3f_A0_%1.3f/E_AVE_VAR_STD_%s.dat" % (R,A0,TYPE),"r").readlines()
            #print(Ri,A0i)
        except FileNotFoundError:
            TMP = [float("Nan")]*4
        E[Ri,A0i]   = float(TMP[1])
        STD[Ri,A0i] = float(TMP[3])


# Cavity-free Dissociation Reference
WC = 3.0/27.2114
E_CAV  = 0.5 * WC
E_EL   = -1.000
E_DISS = R_LIST*0 + E_EL + E_CAV
plt.plot( R_LIST[:], E_DISS, "--", c="black", alpha=0.5, lw=1 )
plt.plot( R_LIST[:], E_DISS - E_CAV, "--", c="red", alpha=0.5, lw=1 )

for A0i,A0 in enumerate( A0_LIST ):
    plt.plot( R_LIST[:], E[:,A0i], "-o", label="A0 = %1.1f" % A0 )
    #plt.plot( R_LIST[:], E[:], "-o", c="black", label="$\langle E \\rangle$" )
    plt.plot( R_LIST[:], E[:,A0i]-STD[:,A0i], "_", c="black", alpha=0.75 )#, label="$\pm\sqrt{\langle (\Delta E)^2 \\rangle}$" )
    plt.plot( R_LIST[:], E[:,A0i]+STD[:,A0i], "_", c="black", alpha=0.75 )
plt.legend()
plt.xlim(R_LIST[0],R_LIST[-2])
plt.xlabel("H-H Bond Length, R (a.u.)",fontsize=15)
plt.ylabel("Energy (a.u.)",fontsize=15)
plt.tight_layout()
plt.savefig(f"PES_{TYPE}.jpg",dpi=400)
plt.clf()




for A0i,A0 in enumerate( A0_LIST ):
    #print( A0, E[-1,A0i], STD[-1,A0i] )
    plt.plot( R_LIST[:], E[:,A0i]-E[-1,A0i], "-o", label="A0 = %1.1f" % A0 )
    plt.plot( R_LIST[:], E[:,A0i]-E[-1,A0i]-(STD[:,A0i]+STD[-1,A0i]), "_", c="black", alpha=0.75 )
    plt.plot( R_LIST[:], E[:,A0i]-E[-1,A0i]+(STD[:,A0i]+STD[-1,A0i]), "_", c="black", alpha=0.75 )

plt.legend()
plt.xlim(R_LIST[0],R_LIST[-2])
plt.xlabel("H-H Bond Length, R (a.u.)",fontsize=15)
plt.ylabel("Energy, $E - E(R\\rightarrow\infty)$ (a.u.)",fontsize=15)
plt.tight_layout()
plt.savefig(f"PES_{TYPE}_RINF.jpg",dpi=400)
plt.clf()




### INTERPOLATE AND FIND LOCATION OF MINIMA OF PES ###
R_min = []
R_FINE = np.linspace( 1.0, 1.5, 10**2 )
for A0i,A0 in enumerate( A0_LIST ):
    f_interp = interp1d( R_LIST[3:-2], E[3:-2,A0i], kind="cubic" )
    R_min.append( R_FINE[ np.argmin( f_interp(R_FINE) ) ] )
    print( "A0, MIN", round(A0,3), round(R_min[-1],3) )

plt.plot( A0_LIST, R_min, "-o", c="black" )
plt.xlim(A0_LIST[0],A0_LIST[-1])
plt.xlabel("Coupling Strength, $A_0$ (a.u.)",fontsize=15)
plt.ylabel("PES Minimum, $R_\mathrm{MIN}$ (a.u.)",fontsize=15)
plt.tight_layout()
plt.savefig(f"R_MIN_{TYPE}.jpg",dpi=400)
plt.clf()
