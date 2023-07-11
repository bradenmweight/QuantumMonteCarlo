import numpy as np
import scipy
from matplotlib import pyplot as plt
import subprocess as sp
from scipy.interpolate import interp1d

from potential import get_potential

def print_results( positions, energy_traj, PARAM ):
    # Print the results
    print("\n\tNumber of Walkers (Final Configuration):", len(positions[0,:]))
    print(f"\tAverage Energy: {np.average( energy_traj[:,0] )} " )
    print(f"\tSTD     Energy: {np.std( energy_traj[:,0] )} " )
    #print(f"\tVAR     Energy: {np.var( energy_traj[:,0] )} " )

def get_Ld( WFN, EDGES ):

    DX  = EDGES[1]  - EDGES[0]
    P   = WFN**2 / np.sum( WFN**2 ) # Define Normalized Prob
    IPR = 1 / np.sum( P**2 ) # Number of Sites
    Ld  = IPR * DX # Convert Number of Sites to Length
    return Ld

def get_Photon_Number( WFN, EDGES, FREQ ):
    NMAX  = 10
    N_AVE = 0
    OVLP  = np.zeros( (NMAX) )
    dq    = EDGES[1] - EDGES[0]
    for n in range( NMAX ):
        H_n   = scipy.special.hermite( n )
        PHI_n = FREQ**(1/4) * np.exp( -FREQ * EDGES**2 / 2) * H_n( FREQ**(1/2) * EDGES)
        PHI_n /= np.sqrt( np.sum(PHI_n**2) * dq )
        OVLP[n] = np.sum( WFN * PHI_n ) * dq

    N_AVE = np.average( np.arange(NMAX) * OVLP**2 )
    #print( "<N> = ", N_AVE )
    #print("Overlap:\n",OVLP)
    return N_AVE, OVLP

def plot( positions, TRAJ, energy_traj, PARAM, WFNs, production_flag=True ):
    
    R_NUC = PARAM["R_NUC"]

    if ( PARAM["DO_POLARITON"] == True ):
        EDGES, EL_WFN, PHOT_WFN = WFNs
    else:
        EDGES, EL_WFN = WFNs

    if ( production_flag == True ):
        OUT_NAME = "Production"
        num_steps = PARAM["num_steps_production"]
        time_step = PARAM["time_step_production"]
    else:
        OUT_NAME = "Equilibration"
        num_steps = PARAM["num_steps_equilibration"]
        time_step = PARAM["time_step_equilibration"]

    if ( PARAM["DATA_DIR"] is not None ):
        DATA_DIR = "DATA_" + PARAM["DATA_DIR"]
        sp.call(f"mkdir -p {DATA_DIR}",shell=True)
    else:
        DATA_DIR = "PLOT_DATA"
        sp.call(f"mkdir -p {DATA_DIR}",shell=True)

    dimension = PARAM["dimension"]
    particles = PARAM["particles"]
    interacting = PARAM["interacting"]

    E_AVE = np.average( energy_traj[:,0] )
    E_VAR = np.var(     energy_traj[:,0] )
    E_STD = np.std(     energy_traj[:,0] )

    # Save Energy Results
    np.savetxt( f"{DATA_DIR}/E_AVE_VAR_STD_{OUT_NAME}.dat", np.array([E_AVE,E_VAR,E_STD]), fmt="%1.8f", header="AVE, VAR, STD (of Energy)" )

    # DMQ Result
    EDGES = (EDGES[:-1] + EDGES[1:])/2
    dR    = EDGES[1] - EDGES[0]
    for d in range( dimension ):
        EL_WFN[:,d] = EL_WFN[:,d] / np.sqrt( np.sum( EL_WFN[:,d]**2 * dR ) )

    if ( PARAM["DO_POLARITON"] == True ):
        PHOT_WFN = PHOT_WFN / np.sqrt( np.sum( PHOT_WFN**2 * dR ) )


    # Compute Observables with DQMC Wavefunction
    X = np.linspace( EDGES[0],EDGES[-1],5000 )
    NX = len(EDGES)
    dX = EDGES[1] - EDGES[0]
    AVE_X  = np.zeros( (dimension) )
    AVE_X2 = np.zeros( (dimension) )
    Ld     = np.zeros( (dimension) )
    for d in range( dimension ):
        AVE_X[d]  = np.sum( EDGES    * EL_WFN[:,d]**2 ) * dX
        AVE_X2[d] = np.sum( EDGES**2 * EL_WFN[:,d]**2 ) * dX
        Ld[d]     = get_Ld( EL_WFN[:,d], EDGES )
        print( "\tdim = %1.0f, <x> = %1.4f, <x^2> = %3.4f, <x^2>-<x>^2 = %3.4f, Ld = %1.4f" % (d, AVE_X[d], AVE_X2[d], AVE_X2[d] - AVE_X[d]**2, Ld[d] ) )

    np.savetxt(f"{DATA_DIR}/X_AVE_VAR_STD_Ld_{OUT_NAME}.dat", np.c_[np.arange(dimension),AVE_X, AVE_X2, Ld], fmt="%1.6f", header="Dim, <X>, <X^2>, Ld" )

    if ( PARAM["DO_POLARITON"] == True ):
        AVE_QC               = np.sum( EDGES    * PHOT_WFN**2 ) * dX
        AVE_QC2              = np.sum( EDGES**2 * PHOT_WFN**2 ) * dX
        N_AVE, PHOT_WFN_FOCK = get_Photon_Number( PHOT_WFN, EDGES, PARAM["CAVITY_FREQ"] )
        print( "\t<QC> = %1.4f, <QC^2> = %1.4f, <QC^2>-<QC>^2 = %1.4f, <N> = %1.4f" % (AVE_QC, AVE_QC2, AVE_QC2 - AVE_QC**2, N_AVE ) )
        np.savetxt(f"{DATA_DIR}/N_AVE_{OUT_NAME}.dat", np.array([N_AVE]), fmt="%1.6f", header="<N>" )
        np.savetxt(f"{DATA_DIR}/PHOTON_WAVEFUNCTION_FOCK_BASIS_{OUT_NAME}.dat", np.c_[np.arange(len(PHOT_WFN_FOCK)), PHOT_WFN_FOCK], fmt="%1.6f", header="n, <n|PHOT>" )

    # Plot the Results
    for d in range( dimension ):
        POT = get_potential(X,PARAM)[:,d]
        FACTOR = np.max(np.abs(POT)) / np.max(EL_WFN[:,d])
        #plt.plot( EDGES, EL_WFN[:,d] * FACTOR + E_AVE, "-o", c="red", label="DQMQ" )
        plt.plot( EDGES, EL_WFN[:,d], "-", lw=6, c="red", label="DQMQ" )
        plt.plot( X, -1*POT * FACTOR**(-1), label="V(x)" )
        MAX_X = np.max( [abs(EDGES[0]),abs(EDGES[-1])] )
        #plt.xlim( EDGES[0], EDGES[-1])
        plt.xlim( -R_NUC[1,0]-5, R_NUC[1,0]+5)
        #plt.ylim( E_AVE*1.5, 0.1 )
        plt.ylim( 0 )
        plt.xlabel(f"Position Along Dimension {d}", fontsize=15)
        plt.ylabel("Wavefunction / Potential Energy", fontsize=15)
        plt.title(f"{dimension} Dimensions; {particles} Particles; Interacting: {interacting}",fontsize=15)
        plt.tight_layout()
        plt.savefig(f"{DATA_DIR}/WAVEFUNCTION_d{d}_N{particles}_INT_{interacting}_{OUT_NAME}.jpg",dpi=300)
        plt.clf()

        np.savetxt(f"{DATA_DIR}/WAVEFUNCTION_d{d}_N{particles}_INT_{interacting}_{OUT_NAME}.dat", np.c_[EDGES, EL_WFN[:,d]] )

    if ( PARAM["DO_POLARITON"] == True ):
        # Plot the Results
        QC_GRID = np.linspace( -4/PARAM["CAVITY_FREQ"], 4/PARAM["CAVITY_FREQ"],5000 )
        plt.plot( EDGES, PHOT_WFN / np.max(PHOT_WFN), "-o", c="red", label="DQMQ" )
        plt.plot( QC_GRID, 0.5 * PARAM["CAVITY_FREQ"]**2 * QC_GRID**2, label="V(QC)" )
        MAX_X = np.max( [abs(EDGES[0]),abs(EDGES[-1])] )
        plt.xlim( QC_GRID[0], QC_GRID[-1])
        plt.ylim( -0.5,2 )
        plt.xlabel("Position, QC",fontsize=15)
        plt.ylabel("Wavefunction / Potential Energy",fontsize=15)
        plt.title(f"{dimension} Dimensions; {particles} Particles; Interacting: {interacting}",fontsize=15)
        plt.savefig(f"{DATA_DIR}/PHOTON_WAVEFUNCTION_d{dimension}_N{particles}_INT_{interacting}_{OUT_NAME}.jpg",dpi=300)
        plt.clf()

        np.savetxt(f"{DATA_DIR}/PHOTON_WAVEFUNCTION_d{d}_N{particles}_INT_{interacting}_{OUT_NAME}.dat", np.c_[EDGES, PHOT_WFN] )


        # # Plot the photonic trajectory
        # plt.plot( PARAM["QC"][:], np.arange(num_steps)[::-1], "-o" )
        # plt.xlim( -12,12 )
        # plt.xlabel("Photinic Position, QC",fontsize=15)
        # plt.legend()
        # plt.ylabel("Simulation Step",fontsize=15)
        # plt.title(f"{dimension} Dimensions; {particles} Particles; Interacting: {interacting}",fontsize=15)
        # plt.savefig(f"{DATA_DIR}/PHOTON_TRAJECTORY_d{dimension}_N{particles}_INT_{interacting}_{OUT_NAME}.jpg",dpi=300)
        # plt.clf()

    # Plot the trajectory
    for p in range(particles):
        #COM_p = np.average( TRAJ[p,:,0,:], axis=0 )
        plt.plot( TRAJ[p,0,0,:], np.arange(num_steps)[::-1], "-o", label=f"Particle {p+1}" )
    plt.xlim( -12,12 )
    plt.xlabel("Position, X",fontsize=15)
    plt.legend()
    plt.ylabel("Simulation Step",fontsize=15)
    plt.title(f"{dimension} Dimensions; {particles} Particles; Interacting: {interacting}",fontsize=15)
    plt.savefig(f"{DATA_DIR}/TRAJECTORY_d{dimension}_N{particles}_INT_{interacting}_{OUT_NAME}.jpg",dpi=300)
    plt.clf()

    if ( PARAM["DO_POLARITON"] == True ):
        # Plot the trajectory
        #for w in range( len(PARAM["QC_TRAJ"]) ):
        for w in range( 10 ):
            plt.plot( PARAM["QC_TRAJ"][w,:], np.arange(num_steps)[::-1], "-o" )
        plt.xlim( -4/PARAM["CAVITY_FREQ"], 4/PARAM["CAVITY_FREQ"] )
        plt.xlabel("Position, qc",fontsize=15)
        plt.legend()
        plt.ylabel("Simulation Step",fontsize=15)
        plt.title(f"{dimension} Dimensions; {particles} Particles; Interacting: {interacting}",fontsize=15)
        plt.savefig(f"{DATA_DIR}/QC_TRAJECTORY_d{dimension}_N{particles}_INT_{interacting}_{OUT_NAME}.jpg",dpi=300)
        plt.clf()

    # Plot the trajectory of the energy
    AVE = np.average(energy_traj[:,0])
    STD = np.std(energy_traj[:,0])
    plt.plot( np.arange(num_steps)*time_step, energy_traj[:,0], "-o", c="black", label="<E_T> = %1.6f" % AVE )
    plt.plot( np.arange(num_steps)*time_step, energy_traj[:,0]*0 + AVE, "-", c="red" )
    plt.plot( np.arange(num_steps)*time_step, energy_traj[:,0]*0 + AVE - STD, "--", c="red", label="STD = %1.6f" % STD )
    plt.plot( np.arange(num_steps)*time_step, energy_traj[:,0]*0 + AVE + STD, "--", c="red" )
    plt.legend()
    plt.xlim( 0,num_steps*time_step )
    plt.xlabel("Simulation Time (a.u.)",fontsize=15)
    plt.ylabel("Energy",fontsize=15)
    plt.title(f"{dimension} Dimensions; {particles} Particles; Interacting: {interacting}",fontsize=15)
    plt.savefig(f"{DATA_DIR}/ENERGY_d{dimension}_N{particles}_INT_{interacting}_{OUT_NAME}.jpg",dpi=300)
    plt.clf()
    np.savetxt(f"{DATA_DIR}/ENERGY_d{dimension}_N{particles}_INT_{interacting}_{OUT_NAME}.dat", np.c_[np.arange(num_steps)*time_step, energy_traj[:,0]] )

    # Plot the trajectory of the trial energy
    AVE = np.average(energy_traj[:,3])
    STD = np.std(energy_traj[:,3])
    plt.plot( np.arange(num_steps)*time_step, energy_traj[:,3], "-o", c="black", label="<E_T> = %1.6f" % AVE )
    plt.plot( np.arange(num_steps)*time_step, energy_traj[:,3]*0 + AVE, "-", c="red" )
    plt.plot( np.arange(num_steps)*time_step, energy_traj[:,3]*0 + AVE - STD, "--", c="red", label="STD = %1.6f" % STD )
    plt.plot( np.arange(num_steps)*time_step, energy_traj[:,3]*0 + AVE + STD, "--", c="red" )
    plt.legend()
    plt.xlim( 0,num_steps*time_step )
    plt.xlabel("Simulation Time (a.u.)",fontsize=15)
    plt.ylabel("Energy",fontsize=15)
    plt.title(f"{dimension} Dimensions; {particles} Particles; Interacting: {interacting}",fontsize=15)
    plt.savefig(f"{DATA_DIR}/E_TRIAL_d{dimension}_N{particles}_INT_{interacting}_{OUT_NAME}.jpg",dpi=300)
    plt.clf()
    np.savetxt(f"{DATA_DIR}/E_TRIAL_d{dimension}_N{particles}_INT_{interacting}_{OUT_NAME}.dat", np.c_[np.arange(num_steps)*time_step, energy_traj[:,3]] )

    # Plot the trajectory of the trial energy
    AVE = np.average(energy_traj[:,4])
    STD = np.std(energy_traj[:,4])
    plt.plot( np.arange(num_steps)*time_step, energy_traj[:,4], "-o", c="black", label="<N_w> = %1.0f" % AVE )
    plt.plot( np.arange(num_steps)*time_step, energy_traj[:,4]*0 + AVE, "-", c="red" )
    plt.plot( np.arange(num_steps)*time_step, energy_traj[:,4]*0 + AVE - STD, "--", c="red", label="STD = %1.0f" % STD )
    plt.plot( np.arange(num_steps)*time_step, energy_traj[:,4]*0 + AVE + STD, "--", c="red" )
    plt.legend()
    plt.xlim( 0,num_steps*time_step )
    plt.xlabel("Simulation Time (a.u.)",fontsize=15)
    plt.ylabel("Number of Gaussian Walkers",fontsize=15)
    plt.title(f"{dimension} Dimensions; {particles} Particles; Interacting: {interacting}",fontsize=15)
    plt.savefig(f"{DATA_DIR}/Nw_d{dimension}_N{particles}_INT_{interacting}_{OUT_NAME}.jpg",dpi=300)
    plt.clf()
    np.savetxt(f"{DATA_DIR}/Nw_d{dimension}_N{particles}_INT_{interacting}_{OUT_NAME}.dat", np.c_[np.arange(num_steps)*time_step, energy_traj[:,3]] )
