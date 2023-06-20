import numpy as np
from matplotlib import pyplot as plt

from potential import get_potential

def print_results( positions, PARAM ):

    # Print the results
    print("\n\tNumber of Walkers:", len(positions[0,:]))
    print("\n\tNumber of Particles:", len(positions))
    print("\tAverage position:", np.average( positions, axis=(0,1) ) )
    print(f"\tAverage energy: {np.average( get_potential(positions,PARAM) )} " )
    print(f"\tVAR     energy: {np.var( get_potential(positions,PARAM) )} " )

def plot( positions, TRAJ, energy_traj, PARAM ):

    dimension = PARAM["dimension"]
    particles = PARAM["particles"]
    interacting = PARAM["interacting"]
    num_steps = PARAM["num_steps"]




    E_AVE = np.average( get_potential(positions,PARAM) )
    E_VAR = np.var( get_potential(positions,PARAM) )
    E_STD = np.std( get_potential(positions,PARAM) )

    # Save Energy Results
    np.savetxt( "E_AVE_VAR_STD.dat", np.array([E_AVE,E_VAR,E_STD]).T )

    # DMQ Result
    PSI_0_DMQ_P0, EDGES = np.histogram( positions[:,:].flatten(), bins=100 )
    EDGES = (EDGES[:-1] + EDGES[1:])/2
    PSI_0_DMQ_P0 = PSI_0_DMQ_P0 / np.linalg.norm( PSI_0_DMQ_P0)

    # Exact Result
    E_EXACT = -0.600 # H2+ (Equilibrium, R12 ~ 2.0 --> E_0 = -0.6; Far-away, R12 ~ 10.0 --> E_0 = -0.5)
    X = np.linspace( -12,12,5000 )
    PSI_0_EXACT = np.exp( -np.abs(X) / 2 ) + E_EXACT

    # Compute Observables with DQMC Wavefunction
    NX = len(EDGES)
    dX = EDGES[1] - EDGES[0]
    AVE_X  = np.sum( EDGES    * PSI_0_DMQ_P0**2 ) * dX
    AVE_X2 = np.sum( EDGES**2 * PSI_0_DMQ_P0**2 ) * dX
    print( "\t<x> = %1.4f, <x^2> = %1.4f, <x^2>-<x>^2 = %1.4f" % (AVE_X, AVE_X2, AVE_X2 - AVE_X**2 ) )

    # Plot the Results
    plt.plot( EDGES, PSI_0_DMQ_P0/ np.max(PSI_0_DMQ_P0) + E_AVE, "-o", c="red", label="DQMQ" )
    plt.plot( X, PSI_0_EXACT, label="Exact" )
    plt.plot( X, get_potential(X,PARAM), label="V(x)" )
    MAX_X = np.max( [abs(EDGES[0]),abs(EDGES[-1])] )
    plt.xlim( -12, 12)
    #plt.ylim( -30, 0 )
    plt.xlabel("Position, X",fontsize=15)
    plt.ylabel("Wavefunction / Potential Energy",fontsize=15)
    plt.title(f"{dimension} Dimensions; {particles} Particles; Interacting: {interacting}",fontsize=15)
    plt.savefig(f"WAVEFUNCTION_d{dimension}_N{particles}_{interacting}.jpg",dpi=300)
    plt.clf()

    # Plot the trajectory
    for traj in range( 10 ):
        if ( particles == 1 ):
            plt.plot( TRAJ[0,traj,0,:], np.arange(num_steps)[::-1], "o", alpha=0.2, c="black" )
        if ( particles == 2 ):
            plt.plot( TRAJ[0,traj,0,:], np.arange(num_steps)[::-1], "o", alpha=0.2, c="black" )
            plt.plot( TRAJ[1,traj,0,:], np.arange(num_steps)[::-1], "o", alpha=0.2, c="red" )
    plt.xlim( -12,12 )
    plt.xlabel("Position, X",fontsize=15)
    if ( particles == 2 ): 
        plt.xlabel("Position, $|r_1 - r_2|$",fontsize=15)
        plt.xlim( -12,12 )

    plt.ylabel("Simulation Step",fontsize=15)
    plt.title(f"{dimension} Dimensions; {particles} Particles; Interacting: {interacting}",fontsize=15)
    plt.savefig(f"TRAJECTORY_d{dimension}_N{particles}_{interacting}.jpg",dpi=300)
    plt.clf()

    # Plot the trajectory of the energy
    #plt.plot( np.arange(num_steps), energy_traj[:,0], "-o", c="black" )
    #plt.errorbar( np.arange(num_steps), energy_traj[:,0], yerr=0.01*energy_traj[:,1], fmt="b-o", ecolor="red", capsize=10, label="E, VAR/100" )
    plt.errorbar( np.arange(num_steps), energy_traj[:,0], yerr=0.01*energy_traj[:,2], fmt="b-o", ecolor="red", capsize=10, label="E, STD/100" )
    plt.legend()
    plt.xlim( 0,num_steps )
    plt.xlabel("Simulation Step",fontsize=15)
    plt.ylabel("Energy",fontsize=15)
    plt.title(f"{dimension} Dimensions; {particles} Particles; Interacting: {interacting}",fontsize=15)
    plt.savefig(f"ENERGY_d{dimension}_N{particles}_{interacting}.jpg",dpi=300)
    plt.clf()

    # color_list = ['black','blue']
    # plt.errorbar( dimension_list, E_AVE[:,0], yerr=np.sqrt(E_VAR[:,0]), fmt="b-o", ecolor="blue", capsize=8, label=f"1 Particles" )
    # if ( particle_list[-1] >= 2 ):
    #     plt.errorbar( dimension_list, E_AVE[:,1], yerr=np.sqrt(E_VAR[:,1]), fmt="g-o", ecolor="green", capsize=8, label=f"2 Particles" )

    # plt.legend()
    # plt.xlabel("Dimension, d",fontsize=15)
    # plt.ylabel("Ground State Energy",fontsize=15)
    # plt.savefig(f"GS_ENERGY_d_p.jpg",dpi=300)
    # plt.clf()