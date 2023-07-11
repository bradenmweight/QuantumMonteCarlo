import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

R_LIST  = np.arange( 0.2, 6.0+0.2, 0.2 )
A0_LIST = np.arange( 0.0, 0.5+0.1, 0.1 )# np.arange( 0.0, 1.0+0.1, 0.1 )
WC_LIST = np.arange( 5.0, 20.0+5.0, 5.0 )

color_list = ["black", 'red', 'green', 'purple', 'blue', 'orange', 'gray']

R_LIST = np.append( R_LIST, 30.0 )

TYPE = "Production" # "Equilibration", "Production"

E       = np.zeros( (len(R_LIST), len(A0_LIST), len(WC_LIST)) )
VAR     = np.zeros( (len(R_LIST), len(A0_LIST), len(WC_LIST)) )
STD     = np.zeros( (len(R_LIST), len(A0_LIST), len(WC_LIST)) )
X_AVE   = np.zeros( (len(R_LIST), len(A0_LIST), len(WC_LIST)) )
X2_AVE  = np.zeros( (len(R_LIST), len(A0_LIST), len(WC_LIST)) )
Ld      = np.zeros( (len(R_LIST), len(A0_LIST), len(WC_LIST)) )

for Ri,R in enumerate( R_LIST ):
    for A0i,A0 in enumerate( A0_LIST ):
        for WCi,WC in enumerate( WC_LIST ):
            try:
                TMP1 = open("DATA_DATA_R_%1.3f_A0_%1.4f_WC_%1.4f/E_AVE_VAR_STD_%s.dat" % (R,A0,WC,TYPE),"r").readlines()
                TMP2 = np.loadtxt("DATA_DATA_R_%1.3f_A0_%1.4f_WC_%1.4f/X_AVE_VAR_STD_Ld_%s.dat" % (R,A0,WC,TYPE))
                #print(Ri,A0i,WCi)
            except FileNotFoundError:
                TMP1 = [float("Nan")]*4
                TMP2 = np.array([[float("Nan"),float("Nan")]*3])
                #print("DATA_DATA_R_%1.4f_A0_%1.4f_WC_%1.4f/E_AVE_VAR_STD_%s.dat" % (R,A0,WC,TYPE))
                #print( "NAN",Ri,A0i,WCi )
            E[Ri,A0i,WCi]      = float(TMP1[1])
            VAR[Ri,A0i,WCi]    = float(TMP1[2])
            STD[Ri,A0i,WCi]    = float(TMP1[3])

            X_AVE[Ri,A0i,WCi]  = float(TMP2[0,0])
            X2_AVE[Ri,A0i,WCi] = float(TMP2[0,1])
            Ld[Ri,A0i,WCi]     = float(TMP2[0,2])

for WCi,WC in enumerate( WC_LIST ):

    # Cavity-free Dissociation Reference
    WC_eV = WC
    WC    = WC/27.2114
    E_CAV  = 0.5 * WC
    E_EL   = -1.000 # Cavity-free H2 dissociation
    E_DISS = R_LIST*0 + E_EL + E_CAV # ZPE + cavity-free result
    plt.plot( R_LIST[:], E_DISS, "--", c="black", alpha=1, lw=1 )
    #plt.plot( R_LIST[:], E_DISS-E_CAV, "--", c="red", alpha=1, lw=1 )

    for A0i,A0 in enumerate( A0_LIST ):
        plt.plot( R_LIST[:], E[:,A0i,WCi], "-o", c=color_list[A0i], label="A0 = %1.1f" % A0 )
        plt.plot( R_LIST[:], E[:,A0i,WCi] - STD[:,A0i,WCi], "_", c=color_list[A0i], alpha=1 )#, label="$\pm\sqrt{\langle (\Delta E)^2 \\rangle}$" )
        plt.plot( R_LIST[:], E[:,A0i,WCi] + STD[:,A0i,WCi], "_", c=color_list[A0i], alpha=1 )
    plt.legend()
    plt.xlim(R_LIST[0],R_LIST[-2])
    #print( E_CAV, WCi )
    #plt.ylim(round(-1.25+E_CAV,1), round(E[-1,-1,WCi]+0.1,1))
    plt.xlabel("H-H Bond Length, R (a.u.)",fontsize=15)
    plt.ylabel("Energy (a.u.)",fontsize=15)
    plt.tight_layout()
    plt.savefig(f"PES_{TYPE}_WC_{WC_eV}.jpg",dpi=400)
    plt.clf()
    np.savetxt( f"PES_{TYPE}_WC_{WC_eV}.dat", np.c_[R_LIST[:], E[:,:,WCi]] )
    np.savetxt( f"PES_{TYPE}_WC_{WC_eV}_ERROR.dat", np.c_[R_LIST[:], STD[:,:,WCi]] )

    for A0i,A0 in enumerate( A0_LIST ):
        plt.plot( R_LIST[:], E[:,A0i,WCi] - E[:,0,WCi], "-o", c=color_list[A0i], label="A0 = %1.1f" % A0 )

        # STD_F = np.sqrt( STD_A**2 + STD_B**2 - 2*STD_AB ), STD_AB = 0 --> Uncorrelated A,B
        # VAR_F =          VAR_A    + VAR_B    - 2*VAR_AB  , STD_AB = 0 --> Uncorrelated A,B
        STD_F = np.sqrt( STD[:,A0i,WCi]**2 + STD[:,0,WCi]**2 )
        plt.plot( R_LIST[:], E[:,A0i,WCi] - E[:,0,WCi] - STD_F, "_", c=color_list[A0i], alpha=0.75 )#, label="$\pm\sqrt{\langle (\Delta E)^2 \\rangle}$" )
        plt.plot( R_LIST[:], E[:,A0i,WCi] - E[:,0,WCi] + STD_F, "_", c=color_list[A0i], alpha=0.75 )
    plt.legend()
    plt.xlim(R_LIST[0],R_LIST[-2])
    plt.xlabel("H-H Bond Length, R (a.u.)",fontsize=15)
    plt.ylabel("$E(A_0) - E(0)$, (a.u.)",fontsize=15)
    plt.tight_layout()
    plt.savefig(f"PES_DIFF_{TYPE}_WC_{WC_eV}.jpg",dpi=400)
    plt.clf()

    DIFF_PES       = np.zeros( E.shape ) 
    DIFF_PES_ERROR = np.zeros( E.shape ) 
    for A0i,A0 in enumerate( A0_LIST ):
        DIFF_PES[:,A0i,WCi]       = E[:,A0i,WCi] - E[:,0,WCi]
        DIFF_PES_ERROR[:,A0i,WCi] = np.sqrt( STD[:,A0i,WCi]**2 + STD[:,0,WCi]**2 )

    np.savetxt( f"PES_DIFF_{TYPE}_WC_{WC_eV}.dat", np.c_[R_LIST[:], DIFF_PES[:,:,WCi]] )
    np.savetxt( f"PES_DIFF_{TYPE}_WC_{WC_eV}_ERROR.dat", np.c_[R_LIST[:], DIFF_PES_ERROR[:,:,WCi]] )




    for A0i,A0 in enumerate( A0_LIST ):
        plt.plot( R_LIST[:], E[:,A0i,WCi]-E[-1,A0i,WCi], "-o", c=color_list[A0i], label="A0 = %1.1f" % A0 )

        # STD_F = np.sqrt( STD_A**2 + STD_B**2 - 2*STD_AB ), STD_AB = 0 --> Uncorrelated A,B
        # VAR_F =          VAR_A    + VAR_B    - 2*VAR_AB  , STD_AB = 0 --> Uncorrelated A,B
        STD_F = np.sqrt( STD[:,A0i,WCi]**2 + STD[-1,A0i,WCi]**2 )
        plt.plot( R_LIST[:], E[:,A0i,WCi]-E[-1,A0i,WCi]-STD_F, "_", c=color_list[A0i], alpha=0.75 )
        plt.plot( R_LIST[:], E[:,A0i,WCi]-E[-1,A0i,WCi]+STD_F, "_", c=color_list[A0i], alpha=0.75 )

    plt.legend()
    plt.xlim(0.75,2)
    plt.ylim(-0.24,-0.15)
    plt.xlabel("H-H Bond Length, R (a.u.)",fontsize=15)
    plt.ylabel("Energy, $E - E(R\\rightarrow\infty)$ (a.u.)",fontsize=15)
    plt.tight_layout()
    plt.savefig(f"PES_{TYPE}_RINF_WC_{WC_eV}.jpg",dpi=600)
    plt.clf()




### INTERPOLATE AND FIND LOCATION OF MINIMA OF PES ###

R_FINE = np.linspace( 1.0, 1.5, 10**2 )
for WCi,WC in enumerate( WC_LIST ):
    R_min = []
    STD_min = []
    for A0i,A0 in enumerate( A0_LIST ):
        f_interp   = interp1d( R_LIST[3:-2],  E[3:-2,A0i,WCi], kind="cubic" )
        STD_interp = interp1d( R_LIST[3:-2], STD[3:-2,A0i,WCi], kind="cubic" )
        R_min.append(   R_FINE[ np.argmin( f_interp(R_FINE) ) ] )
        STD_min.append( STD_interp( R_min[-1] ) )
        #print( "A0, MIN", round(A0,3), round(R_min[-1],3) ) # , round(STD_min[-1],3) )

    plt.plot( A0_LIST, R_min, "-o", c=color_list[WCi], label="$\omega_\mathrm{c}$ = "+"%1.0f eV"%(WC_LIST[WCi]) )
    plt.plot( A0_LIST, np.array(R_min) - np.array(STD_min), "_", ms=15, mew=1.5, c=color_list[WCi], alpha=1 )#, label="$\pm\sqrt{\langle (\Delta E)^2 \\rangle}$" )
    plt.plot( A0_LIST, np.array(R_min) + np.array(STD_min), "_", ms=15, mew=1.5, c=color_list[WCi], alpha=1 )#, label="$\pm\sqrt{\langle (\Delta E)^2 \\rangle}$" )

    np.savetxt(f"R_MIN_{TYPE}_WC_{WC}.dat", np.c_[ A0_LIST, R_min, STD_min ])

plt.legend()
plt.xlim(A0_LIST[0],A0_LIST[-1])
plt.xlabel("Coupling Strength, $A_0$ (a.u.)",fontsize=15)
plt.ylabel("PES Minimum, $R_\mathrm{MIN}$ (a.u.)",fontsize=15)
plt.tight_layout()
plt.savefig(f"R_MIN_{TYPE}_WC.jpg",dpi=400)
plt.clf()


Ri = ([ j for j in range(len(R_LIST)) if R_LIST[j] == 1.6 ])[0]
for WCi,WC in enumerate( WC_LIST ):
    E_ZPE = E[Ri,0,WCi] # 0.5 * WC/27.2114
    plt.plot( A0_LIST[:], E[Ri,:,WCi] - E_ZPE, "-o", c=color_list[WCi], label="WC = %2.1f eV" % WC )
    plt.plot( A0_LIST[:], E[Ri,:,WCi] - STD[Ri,:,WCi] - E_ZPE, "_", c=color_list[WCi], alpha=0.75 )#, label="$\pm\sqrt{\langle (\Delta E)^2 \\rangle}$" )
    plt.plot( A0_LIST[:], E[Ri,:,WCi] + STD[Ri,:,WCi] - E_ZPE, "_", c=color_list[WCi], alpha=0.75 )
plt.legend()
plt.xlim(A0_LIST[0],A0_LIST[-1])
plt.xlabel("Coupling Strength, $A_0$ (a.u.)",fontsize=15)
plt.ylabel("$E_{QED-CC}$ - $E_{ZPE}$ (a.u.)",fontsize=15)
plt.title("$R_{LiH}$"+" = %1.1f A" % R_LIST[Ri],fontsize=15)
plt.tight_layout()
plt.savefig(f"A0SCAN_{TYPE}_WC.jpg",dpi=400)
plt.clf()



for WCi,WC in enumerate( WC_LIST ):
    for A0i,A0 in enumerate( A0_LIST ):
        plt.plot( R_LIST[:], Ld[:,A0i,WCi], "-o", c=color_list[A0i], label="$A_0$ = %1.1f a.u." % A0 )
    plt.legend()
    #plt.xlim(A0_LIST[0],A0_LIST[-1])
    plt.xlabel("H-H Bond Length, R (a.u.)",fontsize=15)
    plt.ylabel("$L_d $ (a.u.)",fontsize=15)
    plt.tight_layout()
    plt.savefig(f"Ld_A0SCAN_{TYPE}_WC_{round(WC)}.jpg",dpi=400)
    plt.clf()

    np.savetxt( f"Ld_A0SCAN_{TYPE}_WC_{round(WC)}.dat", np.c_[ R_LIST, Ld[:,:,WCi] ] )


for WCi,WC in enumerate( WC_LIST ):
    for A0i,A0 in enumerate( A0_LIST ):
        plt.plot( R_LIST[:], Ld[:,A0i,WCi] - Ld[:,0,WCi], "-o", c=color_list[A0i], label="$A_0$ = %1.1f a.u." % A0 )
    plt.legend()
    #plt.xlim(A0_LIST[0],A0_LIST[-1])
    plt.xlabel("H-H Bond Length, R (a.u.)",fontsize=15)
    plt.ylabel("$L_d (A_0) - L_d(0) $ (a.u.)",fontsize=15)
    plt.tight_layout()
    plt.savefig(f"Ld_DIFF_A0SCAN_{TYPE}_WC_{round(WC)}.jpg",dpi=400)
    plt.clf()

    DIFF_Ld = np.zeros( Ld.shape ) 
    for A0i,A0 in enumerate( A0_LIST ):
        DIFF_Ld[:,A0i,WCi] = Ld[:,A0i,WCi] - Ld[:,0,WCi]
    np.savetxt( f"Ld_DIFF_A0SCAN_{TYPE}_WC_{round(WC)}.dat", np.c_[ R_LIST, DIFF_Ld[:,:,WCi] ] )