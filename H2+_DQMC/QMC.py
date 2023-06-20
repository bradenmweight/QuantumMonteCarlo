import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def getGlobals():
    global num_walkers, num_steps, time_step
    # Parameters
    num_walkers = 10**6
    num_steps = 1000
    time_step = 0.05 # Choose such that: num_steps * time_step > 5

    global R_NUC, Z_NUC
    R_NUC = np.zeros( (2,3) ) # Two H, 3 dimensions
    R_NUC[0,0] = -6 # H
    R_NUC[0,0] =  6 # H
    Z_NUC = np.zeros( (2) )  # Two H
    Z_NUC[0] = 1  # One H
    Z_NUC[1] = 1  # One H

    global dimension, particles, interacting
    dimension   = 3 # Can do many dimensions
    particles   = 1 # Don't do more than 3 particles. Need to optimize code first.
    interacting = True

    global E_TRIAL_0
    E_TRIAL_0 = -1.0 # Choose best guess

def get_potential(x):
    if ( len(x.shape) == 3 ):
        shapes = x.shape
        V = np.zeros( (shapes[0], shapes[1]) )

        # Electron-Electron Correlation
        if ( interacting == True ):
            for p1 in range(shapes[0]):
                for p2 in range(shapes[0]):
                    if ( p1 != p2 ):
                        Ree = np.einsum( "wd->w", (x[p1,:,:]-x[p2,:,:])**2 ) ** (1/2)
                        V[p1,:] += 1/np.abs(Ree)

        # Electron-Nuclear Correlation
        for p1 in range(shapes[0]): # Loop over particles
            for Ni, R, in enumerate( R_NUC ):
                ReN = R_NUC[Ni,:] - x[p1,:,:]
                ReN = np.einsum( "wd->w", ReN**2 ) ** (1/2)
                V[p1,:] -= 1/np.abs(ReN)
        
        # Add all QM particles
        V = np.einsum("pw->w", V[:,:]) # Note: Shape Change

        # Nuclear-Nuclear Interaction
        V_NUC = 0
        if ( len(R_NUC) >= 2 ):
            for Ri1, R1, in enumerate( R_NUC ):
                for Ri2, R2, in enumerate( R_NUC ):
                    if ( Ri1 != Ri2 ):
                        R12 = np.linalg.norm( R1 - R2 )
                        V_NUC += Z_NUC[Ri1]*Z_NUC[Ri2]/R12

        return V + V_NUC
    else:
        V   = -0.001/np.abs(x+5) - 0.001/np.abs(x-5) # Electron-Nuclei Interaction
        return V

def DMC():

    # Initialize positions of the walkers around nuclei as gaussian of width 1
    #positions  = np.random.uniform(size=(particles,num_walkers,dimension))*2-1
    #positions *= 3 # Set uniform length around zero
    positions = np.zeros( (particles,num_walkers,dimension) )
    for Ri,R in enumerate( R_NUC ):
        for dim in range(dimension):
            NSTART = Ri*(num_walkers//len(R_NUC))
            NEND   = (Ri+1)*(num_walkers//len(R_NUC))
            positions[:,NSTART:NEND,dim]  = np.random.normal(R[dim], 2, size=(particles,NEND-NSTART))

    trajectory  = np.zeros( (particles,100,dimension,num_steps) ) # Store first 100 walker for visualization
    energy_traj = np.zeros( (num_steps,3) )

    # Initialize the weights of the walkers
    weights = np.ones(num_walkers)

    E_TRIAL = E_TRIAL_0

    for step in range(num_steps):

        energy_traj[step,0] = np.average( get_potential(positions) )
        energy_traj[step,1] = np.var( get_potential(positions) )
        energy_traj[step,2] = np.std( get_potential(positions) )

        if ( step % 50 == 0 ):
            print ( f"Step = {step}, N = {len(positions[0,:,0])}" )

        # Propagate the walkers
        positions += np.random.normal(0, np.sqrt(time_step), size=(particles,len(positions[0,:]),dimension))

        # Compute the potential energy for each walker
        potential_energies = get_potential(positions)

        # Compute the weights of the walkers
        alpha    = 0.5 # Arbitrary parameter -- Set to ~time-step for simplicity
        E_TRIAL  = E_TRIAL + alpha * np.log( num_walkers / len(positions[0,:]) )
        dE       = potential_energies - E_TRIAL
        POW_THRESH = 7 # This is somewhat arbitrary
        if ( (-time_step * dE).any() > POW_THRESH ):
            INDICES = [ j for j,de in enumerate(dE) if (-time_step * de) > POW_THRESH ]
            print( "HERE:", len(INDICES) )
            dE[ INDICES ] = -7 / 0.01
        weights  = np.exp( -time_step * dE )

        # Evaluate q = s + r
        s        = np.ones(( len(positions[0,:]) ), dtype=int)
        RAND     = np.random.uniform( size=len(positions[0,:]) )
        IND_ZERO = weights < RAND  # s --> 0
        IND_2    = weights - 1 > RAND # s --> 2
        s[ IND_ZERO ]  = -1 # Set to dummy labels for later killing
        s[ IND_2 ]     = -2 # Set to dummy labels for later doubling
        s[ s == 0 ]    = 1 # Keep everything else as is
        s[ IND_ZERO ]  = 0 # Kill these
        s[ IND_2 ]     = 2 # Duplicate these

        if ( np.sum( s ) < 1 ):
            print( "WARNING !!!!" )
            print( "Number of walkers went to zero..." )
            exit()
        #elif ( np.sum( s ) > 10**7 / len(positions[:]) / len(positions[0,:] / len(positions[0,0,:])) ):
        #elif ( np.sum( s ) > 10**7 / len(positions[0,0,:]) ):
        #    print( "WARNING !!!!" )
        #    print( "Too many walkers were spawned...", np.sum( s ) )
        #    exit()

        TMP = []
        for p in range( particles ):
            TMP.append( np.repeat( positions[p,:,:], s[:], axis=0 ) )
        positions = np.array(TMP).reshape( (particles,np.sum(s),dimension) )

        if ( len(positions[0]) > 100 ):
            trajectory[:,:,:,step] = positions[:,:100,:]

    return positions, trajectory, energy_traj

def print_results( positions ):

    # Print the results
    print("\n\tNumber of Walkers:", len(positions[0,:]))
    print("\n\tNumber of Particles:", len(positions))
    print("\tAverage position:", np.average( positions, axis=(0,1) ) )
    print(f"\tAverage energy: {np.average( get_potential(positions) )} " )
    print(f"\tVAR     energy: {np.var( get_potential(positions) )} " )

def plot( positions, TRAJ, energy_traj ):

    E_AVE = np.average( get_potential(positions) )
    E_VAR = np.var( get_potential(positions) )
    E_STD = np.std( get_potential(positions) )

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
    plt.plot( X, get_potential(X), label="V(x)" )
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

def main():
    getGlobals()
    positions, TRAJ, energy_traj = DMC() # This can be very easily parallelized.
    print_results( positions )
    plot(positions, TRAJ, energy_traj)

if ( __name__ == "__main__" ):
    main()
