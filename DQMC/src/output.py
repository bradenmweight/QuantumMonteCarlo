import numpy as np
from matplotlib import pyplot as plt
import subprocess as sp
from scipy.interpolate import interp1d

from potential import get_potential

def print_results( positions, energy_traj, PARAM ):

    # Print the results
    print("\n\tNumber of Walkers:", len(positions[0,:]))
    print("\n\tNumber of Particles:", len(positions))
    print("\tAverage position:", np.average( positions, axis=(0,1) ) )
    print(f"\tAverage energy: {np.average( energy_traj[:,0] )} " )
    print(f"\tVAR     energy: {np.var( energy_traj[:,0] )} " )

def plot( positions, TRAJ, energy_traj, PARAM, EDGES, WFN, production_flag=True ):

    if ( production_flag == True ):
        OUT_NAME = "Production"
        num_steps = PARAM["num_steps_production"]

    else:
        OUT_NAME = "Equilibration"
        num_steps = PARAM["num_steps_equilibration"]

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
    #WFN, EDGES = np.histogram( positions[:,:].flatten(), bins=100 )
    EDGES = (EDGES[:-1] + EDGES[1:])/2
    WFN = WFN / np.linalg.norm( WFN)

    # Exact Result
    E_EXACT = -0.500
    X = np.linspace( -12,12,5000 )
    PSI_0_EXACT = np.exp( -np.abs(X) / 2 ) #+ E_EXACT

    # Compute Observables with DQMC Wavefunction
    NX = len(EDGES)
    dX = EDGES[1] - EDGES[0]
    AVE_X  = np.sum( EDGES    * WFN**2 ) * dX
    AVE_X2 = np.sum( EDGES**2 * WFN**2 ) * dX
    print( "\t<x> = %1.4f, <x^2> = %1.4f, <x^2>-<x>^2 = %1.4f" % (AVE_X, AVE_X2, AVE_X2 - AVE_X**2 ) )

    # Plot the Results
    plt.plot( EDGES, WFN / np.max(WFN) + E_AVE, "-o", c="red", label="DQMQ" )
    #plt.plot( X, PSI_0_EXACT, label="Exact" )
    plt.plot( X, 0.1*get_potential(X,PARAM), label="V(x)/10" )
    MAX_X = np.max( [abs(EDGES[0]),abs(EDGES[-1])] )
    plt.xlim( -2, 2)
    plt.ylim( E_AVE*1.5, 0.1 )
    plt.xlabel("Position, X",fontsize=15)
    plt.ylabel("Wavefunction / Potential Energy",fontsize=15)
    plt.title(f"{dimension} Dimensions; {particles} Particles; Interacting: {interacting}",fontsize=15)
    plt.savefig(f"{DATA_DIR}/WAVEFUNCTION_d{dimension}_N{particles}_INT_{interacting}_{OUT_NAME}.jpg",dpi=300)
    plt.clf()

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

    # Plot the trajectory of the energy
    #if ( np.max(energy_traj[:,1]) < 10*np.max(np.abs(energy_traj[:,0])) ):
    #    plt.errorbar( np.arange(num_steps), energy_traj[:,0], yerr=energy_traj[:,1], fmt="b-o", ecolor="red", capsize=10, label="E, VAR" )
    #else:
    plt.plot( np.arange(num_steps), energy_traj[:,0], "-o", c="black" )
    
    plt.legend()
    plt.xlim( 0,num_steps )
    plt.xlabel("Simulation Step",fontsize=15)
    plt.ylabel("Energy",fontsize=15)
    plt.title(f"{dimension} Dimensions; {particles} Particles; Interacting: {interacting}",fontsize=15)
    plt.savefig(f"{DATA_DIR}/ENERGY_d{dimension}_N{particles}_INT_{interacting}_{OUT_NAME}.jpg",dpi=300)
    plt.clf()

