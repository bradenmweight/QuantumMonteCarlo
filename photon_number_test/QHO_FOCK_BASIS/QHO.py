import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.special import hermite

NM_MAX_LIST  = np.array([2,3,4,5,10])
NF_MAX       = 50
A0_LIST      = np.arange( 0.0, 5, 0.25 )

WM = 1.0
WC = 10.0

NQC  = 1000
QMIN = -5
QMAX = 5
E    = np.zeros(  (len(NM_MAX_LIST), len(A0_LIST)) )
N_EL = np.zeros(  (len(NM_MAX_LIST), len(A0_LIST)) )
N_PH = np.zeros(  (len(NM_MAX_LIST), len(A0_LIST)) )
WFN_PH_FOCK = np.zeros( (len(NM_MAX_LIST), len(A0_LIST), NF_MAX) )
WFN_PH_QC   = np.zeros( (len(NM_MAX_LIST), len(A0_LIST), NQC) )

a = np.zeros( (NF_MAX,NF_MAX) )
for n in range(1, NF_MAX ):
    a[n-1,n] = np.sqrt( n )

for NMi,NM_MAX in enumerate( NM_MAX_LIST ):
    print("NMi", NMi+1, "of", len(NM_MAX_LIST))

    def trace_matter( WFN ):
        cn  = np.zeros( NF_MAX )
        rho = np.zeros( (NF_MAX,NF_MAX) )
        for n in range( NF_MAX ):
            for alpha in range( NM_MAX ):
                polIND = alpha*NF_MAX + n # KRON( MATTER, PHOTON )
                cn[n] += WFN[polIND]

        #for n in range( NF_MAX ):
            #print( n, round(cn[n]**2,4) )
        print( "c0 =",round(cn[0],2), "NORM =",round(np.sum(cn**2),2) )
        return cn

    def get_Fock_Decomposition( WFN ):
        cn     = trace_matter( WFN )
        QGRID  = np.linspace(QMIN,QMAX,NQC)
        dq     = QGRID[1] - QGRID[0]
        WFN_QC = np.zeros(NQC)
        for n in range( NF_MAX ):
            H_n     = hermite( n )
            PHI_n   = WC**(1/4) * np.exp( -WC * QGRID**2 / 2) * H_n( WC**(1/2) * QGRID)
            PHI_n  /= np.sqrt( np.sum(PHI_n**2) * dq )
            WFN_QC += cn[n] * PHI_n
        return WFN_QC, cn

    b = np.zeros( (NM_MAX,NM_MAX) )
    for n in range(1, NM_MAX ):
        b[n-1,n] = np.sqrt( n )

    N_EL_OP = np.kron( b.T @ b, np.identity(NF_MAX) )
    N_PH_OP = np.kron( np.identity(NM_MAX), a.T @ a )

    H_EL  = np.kron( np.diag( np.arange( NM_MAX ) * WM ), np.identity(NF_MAX) )
    H_PH  = np.kron( np.identity(NM_MAX), np.diag( np.arange( NF_MAX ) * WC ) )

    for A0i,A0 in enumerate( A0_LIST ):
        print("\tA0i", A0i+1, "of", len(A0_LIST))
        H_INT = WC * A0    * np.kron( b.T + b, a.T + a )
        H_DSE = WC * A0**2 * np.kron( (b.T + b) @ (b.T + b), np.identity(NF_MAX) )

        H = H_EL + H_PH + H_INT + H_DSE

        Ei, Ui        = np.linalg.eigh( H )
        E[NMi,A0i]    = Ei[0]
        N_EL[NMi,A0i] = Ui[:,0] @ N_EL_OP @ Ui[:,0]
        N_PH[NMi,A0i] = Ui[:,0] @ N_PH_OP @ Ui[:,0]
        WFN_PH_QC[NMi,A0i,:], WFN_PH_FOCK[NMi,A0i,:] = get_Fock_Decomposition( Ui[:,0] )


plt.plot( A0_LIST, A0_LIST**2, "-", c="black", lw=10, alpha=0.5, label="$\sim A_0^2$" )
for NMi, NM_MAX in enumerate(NM_MAX_LIST):
    plt.plot( A0_LIST, N_PH[NMi,:], "-o", lw=3, markersize=8, label="$N_M = $"+f"{NM_MAX}" )
plt.legend()
#plt.ylim(0,np.ceil(np.max(N_PH)))
#plt.ylim(0,30)
plt.xlabel("Coupling Strength $A_0$ (a.u.)",fontsize=15)
plt.ylabel("Average Photon Number, $\langle \hat{a}^\dag\hat{a} \\rangle$",fontsize=15)
plt.title("Fixed Number of Fock States: $N_F = $"+f"{NF_MAX}",fontsize=15)
plt.tight_layout()
plt.savefig("Photon.jpg", dpi=300)
plt.clf()

A0i = -1
QGRID  = np.linspace(QMIN,QMAX,NQC)
dq     = QGRID[1] - QGRID[0]
for NMi, NM_MAX in enumerate(NM_MAX_LIST):
    if ( np.sum(WFN_PH_QC[NMi,A0i,:]) < 0 ):
        WFN_PH_QC[NMi,A0i,:] *= -1
    if ( np.sum(WFN_PH_QC[NMi,A0i,:len(QGRID)//2]**2)*dq < 0.5 ):
        print( "NORM/2",np.sum(WFN_PH_QC[NMi,A0i,:len(QGRID)//2]**2)*dq )
        WFN_PH_QC[NMi,A0i,:] = np.flip(WFN_PH_QC[NMi,A0i,:])
    plt.plot( QGRID, WFN_PH_QC[NMi,A0i,:], label=f"NM = {NM_MAX}")
plt.legend()
plt.xlabel("Photon Coordinate $q_c$ (a.u.)",fontsize=15)
plt.ylabel("Photon Wavefunction",fontsize=15)
plt.title(f"A0 = {A0_LIST[A0i]} a.u.",fontsize=15)
plt.tight_layout()
plt.savefig("WFN_PH_QC.jpg", dpi=300)
plt.clf()


A0i = -1
for NMi, NM_MAX in enumerate(NM_MAX_LIST):
    if ( WFN_PH_FOCK[NMi,A0i,0] < 0 ):
        WFN_PH_FOCK[NMi,A0i,:] *= -1
    plt.plot( np.arange(NF_MAX), WFN_PH_FOCK[NMi,A0i,:], "-o", lw=3, markersize=8, label=f"NM = {NM_MAX}")
plt.legend()
plt.xlim(0,NF_MAX)
plt.xlabel("Fock State, $n$",fontsize=15)
plt.ylabel("Photon Wavefunction",fontsize=15)
plt.title(f"A0 = {A0_LIST[A0i]} a.u.",fontsize=15)
plt.tight_layout()
plt.savefig("WFN_PH_FOCK.jpg", dpi=300)
plt.clf()
