import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

def V(x, wc, model, interacting):
    if ( len(x.shape) == 3 ):
        shapes = x.shape
        V = np.zeros( (shapes[0], shapes[1]) )

        # DO PAIRWISE COULOMB INTERACTION
        if ( interacting == True ):
            for p1 in range(shapes[0]):
                for p2 in range(shapes[0]):
                    if ( p1 != p2 ):
                        R = np.einsum( "wd->w", (x[p1,:,:]-x[p2,:,:])**2 ) ** (1/2)
                        V[p1,:] += 1/np.abs(R)
        
        # Here is harmonic oscillator potential of frequency wc
        V[:,:] += 0.5 * wc**2 * np.einsum( "pwd->pw", x * x )
        V = np.einsum( "pw->w", V[:,:] ) # Sum over particles
        #print( np.argmax( V ), V[ np.argmax( V ) ], np.max( V ) )
        return V
    else:
        r1_2 = x**2
        V   = 0.5 * wc**2 * r1_2
        return V

def DMC(num_walkers, num_steps, time_step, wc, E_TRIAL, dimension, model, particles, interacting):

    # Initialize positions of the walkers
    positions  = np.random.uniform(size=(particles,num_walkers,dimension))*2-1
    positions *= 1 # Set uniform length around zero

    trajectory  = np.zeros( (particles,100,dimension,num_steps) ) # Store first 100 walker for visualization
    energy_traj = np.zeros( (num_steps) )

    # Initialize the weights of the walkers
    weights = np.ones(num_walkers)

    for step in range(num_steps):

        #energy_traj[step] = E_TRIAL
        energy_traj[step] = np.average( V(positions, wc, model, interacting) )

        if ( step % 1000 == 0 ):
            print ( f"Step = {step}, N = {len(positions[0,:,0])}" )

        # Propagate the walkers
        positions += np.random.normal(0, np.sqrt(time_step), size=(particles,len(positions[0,:]),dimension))

        # Compute the potential energy for each walker
        potential_energies = V(positions, wc, model, interacting)

        # Compute the weights of the walkers
        alpha    = 1e-2 # Arbitrary parameter -- Set to ~time-step for simplicity
        E_TRIAL  = E_TRIAL + alpha * np.log( num_walkers / len(positions[0,:]) )
        weights  = np.exp(-time_step * (potential_energies - E_TRIAL))

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
        #elif ( np.sum( s ) > 10**6 ):
        #    print( "WARNING !!!!" )
        #    print( "Too many walkers were spawned..." )
        #    exit()

        TMP = []
        for p in range( particles ):
            TMP.append( np.repeat( positions[p,:,:], s[:], axis=0 ) )
        positions = np.array(TMP).reshape( (particles,np.sum(s),dimension) )

        if ( len(positions[0]) > 100 ):
            trajectory[:,:,:,step] = positions[:,:100 if len(positions[0,:]) >= 100 else len(positions[0,:]),:] # np.zeros( (particles,100,dimension,num_steps) )

    return positions, trajectory, energy_traj

# Parameters
num_walkers = 10000
num_steps = 500
time_step = 0.01 # Choose such that: num_steps * time_step > 5
wc = 1.0
model = "QHO"

dimension_list = [1,2,3,4,5,6,7,8,9,10] # Can do many of these, up to 10
particle_list  = [1,2] # Don't do more than 3 particles
interacting    = True

E_AVE = np.zeros( (len(dimension_list),len(particle_list)) )
E_VAR = np.zeros( (len(dimension_list),len(particle_list)) )
for dIND,dimension in enumerate(dimension_list):
    for pIND,particles in enumerate(particle_list):

        E_TRIAL = 0.5 * dimension * particles

        # Run the DMC simulation for GS
        positions, TRAJ, energy_traj = DMC(num_walkers, num_steps, time_step, wc, E_TRIAL, dimension, model, particles, interacting)

        # Print the results
        print("\n\tNumber of Walkers:", len(positions[0,:]))
        print("\n\tNumber of Particles:", len(positions))
        print("\tAverage position:", np.average( positions, axis=(0,1) ) )
        E_AVE[dIND,pIND] = np.average( V(positions, wc, model, interacting) )
        E_VAR[dIND,pIND] = np.var( V(positions, wc, model, interacting) )
        print(f"\tAverage energy: {E_AVE[dIND,pIND]} " )
        print(f"\tVAR     energy: {E_VAR[dIND,pIND]} " )

        # DMQ Result
        PSI_0_DMQ_P0, EDGES = np.histogram( positions[:,:].flatten(), bins=50 )
        EDGES = (EDGES[:-1] + EDGES[1:])/2
        PSI_0_DMQ_P0 = PSI_0_DMQ_P0 / np.linalg.norm( PSI_0_DMQ_P0)

        # Exact Result
        if ( model == "QHO" ):
            E_EXACT = 0.500 * dimension * particles # Many Non-interacting Particles
            X = np.linspace( -5,5,2000 )
            PSI_0_EXACT = np.exp( -X**2 / 2 ) + E_EXACT

        # Compute Observables with DQMC Wavefunction
        NX = len(EDGES)
        dX = EDGES[1] - EDGES[0]
        AVE_X  = np.sum( EDGES    * PSI_0_DMQ_P0**2 ) * dX
        AVE_X2 = np.sum( EDGES**2 * PSI_0_DMQ_P0**2 ) * dX
        print( "\t<x> = %1.4f, <x^2> = %1.4f, <x^2>-<x>^2 = %1.4f" % (AVE_X, AVE_X2, AVE_X2 - AVE_X**2 ) )

        # Plot the Results
        plt.plot( EDGES, PSI_0_DMQ_P0/ np.max(PSI_0_DMQ_P0) + E_AVE[dIND,pIND], "-o", c="red", label="DQMQ" )
        #plt.plot( EDGES, PSI_0_DMQ_P0/ np.max(PSI_0_DMQ_P0) + 0.5, "-o", c="red", label="DQMQ" )
        plt.plot( X, PSI_0_EXACT, "-", c="black", label="EXACT Non-interacting" )
        plt.plot( X, V(X,wc,model,interacting), label="V(x)" )
        plt.legend()
        MAX_X = np.max( [abs(EDGES[0]),abs(EDGES[-1])] )
        plt.xlim( -MAX_X*1.,MAX_X*1.)
        plt.ylim( 0, 1.5*(np.max(PSI_0_DMQ_P0/ np.max(PSI_0_DMQ_P0)) + E_AVE[dIND,pIND]) )
        plt.xlabel("Position, X",fontsize=15)
        plt.ylabel("Wavefunction / Potential Energy",fontsize=15)
        plt.title(f"Diffusion QMC Results: {model} {dimension}D\n Interacting: {interacting}",fontsize=15)
        plt.savefig(f"WAVEFUNCTION_d{dimension}_N{particles}_{interacting}.jpg",dpi=300)
        plt.clf()

        # Plot the trajectory
        for traj in range( 10 ):
            if ( particles == 1 ):
                plt.plot( TRAJ[0,traj,0,:], np.arange(num_steps)[::-1], "o", alpha=0.2, c="black" )
            if ( particles == 2 ):
                plt.plot( TRAJ[0,traj,0,:], np.arange(num_steps)[::-1], "o", alpha=0.2, c="black" )
                plt.plot( TRAJ[1,traj,0,:], np.arange(num_steps)[::-1], "o", alpha=0.2, c="red" )
        plt.xlim( -4,4 )
        plt.xlabel("Position, X",fontsize=15)
        if ( particles == 2 ): 
            plt.xlabel("Position, $|r_1 - r_2|$",fontsize=15)
            plt.xlim( -8,8 )

        plt.ylabel("Simulation Step",fontsize=15)
        plt.title(f"Diffusion QMC Results: {model} {dimension}D\n Interacting: {interacting}",fontsize=15)
        plt.savefig(f"TRAJECTORY_d{dimension}_N{particles}_{interacting}.jpg",dpi=300)
        plt.clf()

        # Plot the trajectory of the energy
        plt.plot( np.arange(num_steps), energy_traj[:], "-o", c="black" )
        plt.xlim( 0,num_steps )
        plt.xlabel("Simulation Step",fontsize=15)
        plt.ylabel("Energy",fontsize=15)
        plt.title(f"Diffusion QMC Results: {model} {dimension}D\n Interacting: {interacting}",fontsize=15)
        plt.savefig(f"ENERGY_d{dimension}_N{particles}_{interacting}.jpg",dpi=300)
        plt.clf()

        print("\n")





np.savetxt("E_AVE.dat", E_AVE)
np.savetxt("E_VAR.dat", E_VAR)

color_list = ['black','blue']
plt.errorbar( dimension_list, E_AVE[:,0], yerr=np.sqrt(E_VAR[:,0]), fmt="b-o", ecolor="blue", capsize=8, label=f"1 Particles" )
if ( particle_list[-1] >= 2 ):
    plt.errorbar( dimension_list, E_AVE[:,1], yerr=np.sqrt(E_VAR[:,1]), fmt="g-o", ecolor="green", capsize=8, label=f"2 Particles" )

plt.legend()
plt.xlabel("Dimension, d",fontsize=15)
plt.ylabel("Ground State Energy",fontsize=15)
plt.savefig(f"GS_ENERGY_d_p.jpg",dpi=300)
plt.clf()
