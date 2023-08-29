import numpy as np
from matplotlib import pyplot as plt
import scipy
from scipy.special import hermite

A0_LIST = np.arange( 0.0, 5.25, 0.25 )
#A0_LIST = np.array([0.0])

# MATTER
NM_MAX_LIST  = np.array([2,3,4,5,10])
WM = 1.0

# PHOTON
QMIN   = -6
QMAX   = 6
NQC    = 200
QCGRID = np.linspace( QMIN,QMAX,NQC )
dQ     = QCGRID[1] - QCGRID[0]
WC     = 10.0

E = np.zeros(  (len(NM_MAX_LIST), len(A0_LIST)) )
U = np.zeros(  (len(NM_MAX_LIST), len(A0_LIST), NQC) )
N = np.zeros(  (len(NM_MAX_LIST), len(A0_LIST)) )
NF_MAX = 150
WFN_FOCK = np.zeros(  (len(NM_MAX_LIST), len(A0_LIST), NF_MAX) )

def pc2_nm(): # DVR Basis for photon kinetic energy
    pc2 = np.zeros( (NQC,NQC) )
    for n in range( NQC ):
        for m in range( NQC ):
            if ( n == m ):
                pc2[n,m] = np.pi**2 / 3 
            else:
                pc2[n,m] = 2 / (n-m)**2
            pc2[n,m] *= (-1) ** (n-m) / dQ**2
    return pc2 / 2

for NMi,NM_MAX in enumerate( NM_MAX_LIST ):
    print("NMi", NMi+1, "of", len(NM_MAX_LIST))

    def trace_matter( WFN ):
        cn  = np.zeros( NQC )
        rho = np.zeros( (NQC,NQC) )
        for n in range( NQC ):
            for alpha in range( NM_MAX ):
                polIND = alpha*NQC + n # KRON( MATTER, PHOTON )
                cn[n] += WFN[polIND]

        return cn / np.linalg.norm(cn)

    def get_Photon_Number( WFN ):
        OVLP   = np.zeros( (NF_MAX) )
        for n in range( NF_MAX ):
            H_n   = scipy.special.hermite( n )
            PHI_n = WC**(1/4) * np.exp( -WC * QCGRID**2 / 2) * H_n( WC**(1/2) * QCGRID)
            PHI_n /= np.sqrt( np.sum(PHI_n**2) * dQ )
            OVLP[n] = np.sum( WFN * PHI_n ) * dQ

        N_AVE = np.sum( np.arange(NF_MAX) * OVLP**2 ) / np.sum( OVLP**2 )
        print("QC --> Fock:", OVLP[0], "N =", N_AVE)
        return N_AVE, OVLP


    b = np.zeros( (NM_MAX,NM_MAX) )
    for n in range(1, NM_MAX ):
        b[n-1,n] = np.sqrt( n )

    H_EL  = np.kron( np.diag( np.arange( NM_MAX ) * WM ), np.identity(NQC) )
    H_Tph = np.kron( np.identity(NM_MAX), pc2_nm() )
    H_Vph = np.kron( np.identity(NM_MAX), np.diag(0.5 * WC**2 * QCGRID**2) )

    for A0i,A0 in enumerate( A0_LIST ):
        print("\tA0i", A0i+1, "of", len(A0_LIST))
        H_INT = np.sqrt(2 * WC**3) * A0 * np.kron( b.T + b, np.diag(QCGRID) )
        H_DSE = WC * A0**2 * np.kron( (b.T + b) @ (b.T + b), np.identity(NQC) )

        H = H_EL + H_Tph + H_Vph + H_INT + H_DSE

        Ei, Ui        = np.linalg.eigh( H )
        E[NMi,A0i]    = Ei[0]
        U[NMi,A0i,:]  = trace_matter( Ui[:,0] )
        N[NMi,A0i], WFN_FOCK[NMi,A0i,:] = get_Photon_Number( U[NMi,A0i,:] )


plt.plot( A0_LIST, A0_LIST**2, lw=8, c='black', alpha=0.5, label=f"$A_0^2$" )
for NMi in range( len(NM_MAX_LIST) ):
    plt.plot( A0_LIST, E[NMi,:] - E[NMi,0], lw=3, label=f"NM = {NM_MAX_LIST[NMi]}" )
plt.legend()
plt.xlabel("Coupling Strength, $A_0$ (a.u.)",fontsize=15)
plt.ylabel("Ground State Energy, $E_0$ (a.u.)",fontsize=15)
plt.savefig("E.jpg",dpi=300)
plt.clf()

plt.plot( A0_LIST, A0_LIST**2, lw=8, c='black', alpha=0.5, label=f"$A_0^2$" )
for NMi in range( len(NM_MAX_LIST) ):
    plt.plot( A0_LIST, N[NMi,:], lw=2, label=f"NM = {NM_MAX_LIST[NMi]}" )
plt.legend()
plt.xlabel("Coupling Strength, $A_0$ (a.u.)",fontsize=15)
plt.ylabel("Average Photon Number",fontsize=15)
plt.title(f"Fixed Number of Photon Grid Points: N = {NQC}",fontsize=15)
plt.savefig("PHOTON_NUMBER.jpg",dpi=300)
plt.clf()



A0i = -1


for NMi in range( len(NM_MAX_LIST) ):
    if ( np.sum(U[NMi,A0i,:]) < 0 ):
        U[NMi,A0i,:] *= -1
    plt.plot( QCGRID, U[NMi,A0i,:] / np.linalg.norm( U[NMi,A0i,:] ), lw=3, label=f"NM = {NM_MAX_LIST[NMi]}" )
plt.legend()
plt.title(f"A0 = {A0_LIST[A0i]} a.u.",fontsize=15)
plt.xlabel("Photonic Coordinate, $q_c$ (a.u.)",fontsize=15)
plt.ylabel("Wavefunction",fontsize=15)
plt.savefig("WFN_PHOTON_QC.jpg",dpi=300)
plt.clf()


for NMi in range( len(NM_MAX_LIST) ):
    plt.plot( np.arange(NF_MAX), WFN_FOCK[NMi,A0i,:], lw=3, label=f"NM = {NM_MAX_LIST[NMi]}" )
plt.legend()
plt.xlabel("Fock State, $n$",fontsize=15)
plt.ylabel("Wavefunction",fontsize=15)
plt.title(f"A0 = {A0_LIST[A0i]} a.u.",fontsize=15)
plt.savefig("WFN_PHOTON_FOCK.jpg",dpi=300)
plt.clf()